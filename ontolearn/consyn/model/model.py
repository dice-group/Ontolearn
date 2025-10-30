from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ontolearn.consyn.model.decoder import Decoder
from ontolearn.consyn.model.encoder import Encoder
from ontolearn.consyn.tokenizer import ConSynTokenizer
from ontolearn.consyn.utils import DecodingStrategy


class ConSynGeneratorModel(nn.Module):
    def __init__(self, tokenizer: Optional[ConSynTokenizer]=None, input_vocab_size: Optional[int]=None, target_vocab_size: Optional[int]=None, embed_dim: int = 512,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, num_heads: int = 8, ff_dim: int = 2048, dropout_prob: float = 0.1,
                 num_segments: int = None, use_checkpointing: bool = False, pre_norm: bool = False, triplet_margin: float = 1.0):
        super().__init__()
        
        self.encoder = Encoder(vocab_size=input_vocab_size, embed_dim=embed_dim, num_segments=num_segments, num_layers=num_encoder_layers,
                                num_heads=num_heads, ff_dim=ff_dim, dropout_prob=dropout_prob, use_checkpointing=use_checkpointing, pre_norm=pre_norm)

        self.decoder = Decoder(target_vocab_size=target_vocab_size, embed_dim=embed_dim, num_layers=num_decoder_layers, num_heads=num_heads,
                                ff_dim=ff_dim, dropout_prob=dropout_prob, use_checkpointing=use_checkpointing, pre_norm=pre_norm)
                                
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.vocab['[PAD]']
        self.cls_token_id = tokenizer.vocab['[CLS]']
        self.eos_token_id = tokenizer.vocab['[EOS]']

        self.triplet_margin = triplet_margin
        self.triplet_head = nn.Linear(embed_dim, embed_dim)

    def _get_semantic_content_and_eos_status(self, sequence_ids: List[int], eos_id: int, pad_id: int, bos_id: int) -> Tuple[List[int], bool]:
        if not sequence_ids:
            return [], False

        actual_end_idx = len(sequence_ids) - 1
        while actual_end_idx >= 0 and sequence_ids[actual_end_idx] == pad_id:
            actual_end_idx -= 1

        has_explicit_eos = False
        if actual_end_idx >= 0 and sequence_ids[actual_end_idx] == eos_id:
            has_explicit_eos = True
            actual_end_idx -= 1

        semantic_tokens = []
        for j in range(1, actual_end_idx + 1):
            if sequence_ids[j] not in [pad_id, eos_id, bos_id]: 
                semantic_tokens.append(sequence_ids[j])
                
        return actual_end_idx, semantic_tokens, has_explicit_eos

    def get_sequence_embeddings(self, generated_ids: torch.Tensor) -> torch.Tensor:
        device = generated_ids.device
        attention_mask = (generated_ids != self.pad_token_id).long().unsqueeze(1).unsqueeze(1)
        segment_ids = torch.zeros_like(generated_ids)
        
        encoder_output = self.encoder(generated_ids, segment_ids, attention_mask)
        semantic_mask = (generated_ids != self.pad_token_id) & (generated_ids != self.cls_token_id)
        
        sum_embeddings = torch.sum(encoder_output * semantic_mask.unsqueeze(-1), dim=1)
        num_semantic_tokens = torch.sum(semantic_mask, dim=1, keepdim=True)
        
        num_semantic_tokens = torch.where(num_semantic_tokens == 0, torch.tensor(1.0, device=device), num_semantic_tokens)
        sequence_embeddings = sum_embeddings / num_semantic_tokens

        return self.triplet_head(sequence_embeddings) #sequence_embeddings

    def _get_segment_embedding(self, encoder_output: torch.Tensor, start_indices: torch.Tensor, end_indices: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_output.size(0)
        embed_dim = encoder_output.size(-1)
        device = encoder_output.device
    
        embeddings = []
        for i in range(batch_size):
            start = start_indices[i].item() if i < start_indices.size(0) else 0
            end = end_indices[i].item() if i < end_indices.size(0) else start + 1
    
            segment = encoder_output[i, start:end, :]
            if segment.size(0) == 0:
                segment_embedding = torch.zeros(embed_dim, device=device)
            else:
                segment_embedding = segment.mean(dim=0)
            embeddings.append(segment_embedding)
    
        return self.triplet_head(torch.stack(embeddings))  # [B, D]

    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(1)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor,input_attention_mask: torch.Tensor,
                target_ids: Optional[torch.Tensor] = None, target_attention_mask: Optional[torch.Tensor] = None, use_triplet_loss: bool = False) -> torch.Tensor:
        
        encoder_output = self.encoder(input_ids, segment_ids, input_attention_mask)

        decoder_logits = None
        if target_ids is not None and target_attention_mask is not None:
            cross_mask = self.create_padding_mask(input_ids)
            decoder_logits = self.decoder(
                target_ids=target_ids, 
                encoder_output=encoder_output, 
                target_attention_mask=target_attention_mask, 
                encoder_attention_mask=cross_mask
            )
        
        triplet_loss = None
        if use_triplet_loss:
            anchor_embedding = self.triplet_head(encoder_output[:, 0, :])

            vocab = self.tokenizer.vocab
            pos_start, pos_end = (input_ids == vocab['[POS_START]']).nonzero(as_tuple=True)[1], \
                                 (input_ids == vocab['[POS_END]']).nonzero(as_tuple=True)[1]
            neg_start, neg_end = (input_ids == vocab['[NEG_START]']).nonzero(as_tuple=True)[1], \
                                 (input_ids == vocab['[NEG_END]']).nonzero(as_tuple=True)[1]
    
            pos = self._get_segment_embedding(encoder_output, pos_start, pos_end)
            neg = self._get_segment_embedding(encoder_output, neg_start, neg_end)
    
            triplet_loss = F.triplet_margin_loss(anchor_embedding, pos, neg, margin=self.triplet_margin)

        return decoder_logits, triplet_loss
    
    def generate_for_rl(self, input_ids, segment_ids, attention_mask, grammar_parser, 
                    max_length, k=1, decoding_strategy="multinomial", temperature=1.0, 
                    top_k=0, top_p=0.0):

        device = input_ids.device
        batch_size = input_ids.size(0)
    
        input_ids = input_ids.repeat_interleave(k, dim=0)
        segment_ids = segment_ids.repeat_interleave(k, dim=0)
        attention_mask = attention_mask.repeat_interleave(k, dim=0)
    
        encoder_output = self.encoder(input_ids, segment_ids, attention_mask)
        decoder_input = torch.full((batch_size * k, 1), self.cls_token_id, dtype=torch.long, device=device)
    
        finished = torch.zeros(batch_size * k, dtype=torch.bool, device=device)
        seq_log_probs = [[] for _ in range(batch_size * k)]
        semantic_tokens_flat = [[] for _ in range(batch_size * k)]
        has_explicit_eos = [False] * (batch_size * k)
        is_invalid = [False] * (batch_size * k)
    
        for step in range(max_length - 1):
            tgt_len = decoder_input.size(1)
            causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), 1).bool()
            causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(0)
    
            logits = self.decoder(decoder_input, encoder_output, causal_mask, attention_mask)
            logits = logits[:, -1, :]
    
            active = ~finished
            if not active.any():
                break
    
            active_indices = torch.nonzero(active).squeeze(1)
            partial_sequences = decoder_input[active].tolist()
            valid_next_batch = grammar_parser.get_valid_next_token_ids_batch(partial_sequences)
    
            grammar_mask = torch.full_like(logits, float('-inf'))
            for i, valid_set in zip(active_indices.tolist(), valid_next_batch):
                grammar_mask[i, list(valid_set)] = 0.0
    
            masked_logits = logits + grammar_mask
            next_tokens = torch.full((batch_size * k,), self.pad_token_id, device=device)
            
            for i in range(batch_size * k):
                if finished[i]:
                    seq_log_probs[i].append(torch.tensor(0.0, device=device).unsqueeze(0))
                    continue
    
                selected_token, log_prob = DecodingStrategy.select_next_token(
                    decoding_strategy, masked_logits[i], temperature
                )
    
                if selected_token == -1:
                    selected_token = self.eos_token_id
                    log_prob = torch.tensor(float('-inf'), device=device)
    
                next_tokens[i] = selected_token
                seq_log_probs[i].append(log_prob.unsqueeze(0))
    
            just_finished = (next_tokens == self.eos_token_id) & (~finished)
            finished |= just_finished
            next_tokens = torch.where(just_finished, self.pad_token_id, next_tokens)
            decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
    
        for i in range(batch_size * k):
            _, sematic_tokens, eos_flag = self._get_semantic_content_and_eos_status(
                decoder_input[i].cpu().tolist(), self.eos_token_id, self.pad_token_id, self.cls_token_id
            )
            semantic_tokens_flat[i] = sematic_tokens
            has_explicit_eos[i] = eos_flag
            is_invalid[i] = not grammar_parser.is_expression_grammatical_complete(sematic_tokens)
    
        return decoder_input, seq_log_probs, is_invalid, semantic_tokens_flat, has_explicit_eos
