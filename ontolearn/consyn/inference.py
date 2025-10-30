from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ontolearn.consyn.grammar import ConSynGrammarParser
from ontolearn.consyn.model.model import ConSynGeneratorModel
from ontolearn.consyn.reward import ConSynRewardFunction
from ontolearn.consyn.tokenizer import ConSynTokenizer
from ontolearn.consyn.utils import ConSynHypothesisSpace
from ontolearn.heuristics import ConSynHeuristic

class ConSynInference(nn.Module):
    def __init__(self, model: ConSynGeneratorModel, tokenizer: ConSynTokenizer, grammar_parser: ConSynGrammarParser,
        reward_function: ConSynRewardFunction, max_decoder_seq_len: int, device: torch.device = None, verbose:bool=False):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.grammar_parser = grammar_parser
        self.reward_function = reward_function
        self.max_decoder_seq_len = max_decoder_seq_len
        self.device = device
        self.verbose = verbose
        self.pad_token_id = self.tokenizer.vocab['[PAD]']
        
        # self.model.to(self.device)
        
    @torch.no_grad()
    def process_generated_output(self, generated_ids_flat: torch.Tensor, per_token_log_probs_flat: List[List[torch.Tensor]],
                                is_grammatically_invalid_flat: List[bool], semantic_tokens_flat: List[List[int]], has_explicit_eos_flat: List[bool],
                                batch_size: int, k: int, device:Optional[torch.device] = None) -> Tuple[List[List[torch.Tensor]], List[List[str]],
                                                  List[List[List[torch.Tensor]]], List[List[bool]],
                                                  List[List[List[int]]], List[List[bool]]]:

        device = device or self.device
       
        all_cumulative_log_probs_flat_list = [
            torch.stack(seq_log_probs).sum().item() if seq_log_probs else -float('inf')
            for seq_log_probs in per_token_log_probs_flat
        ]
        all_cumulative_log_probs_flat_tensor = torch.tensor(
            all_cumulative_log_probs_flat_list, device=device, dtype=torch.float
        )

        actual_seq_len = generated_ids_flat.size(1)

        reshaped_generated_ids = generated_ids_flat.view(batch_size, k, actual_seq_len)
        reshaped_cumulative_log_probs_tensor = all_cumulative_log_probs_flat_tensor.view(batch_size, k)
        reshaped_is_grammatically_invalid = torch.tensor(is_grammatically_invalid_flat, dtype=torch.bool, device=device).view(batch_size, k)

        reshaped_decoded_concepts: List[List[str]] = []
        for i in range(batch_size * k):
            if i % k == 0: reshaped_decoded_concepts.append([])
            decoded_text = self.tokenizer.decode(
                generated_ids_flat[i][generated_ids_flat[i] != self.pad_token_id].tolist(),
                skip_special_tokens=True
            )
            reshaped_decoded_concepts[-1].append(decoded_text)

        reshaped_semantic_tokens: List[List[List[int]]] = []
        for i in range(batch_size * k):
            if i % k == 0: reshaped_semantic_tokens.append([])
            reshaped_semantic_tokens[-1].append(semantic_tokens_flat[i])

        reshaped_has_explicit_eos: List[List[bool]] = []
        for i in range(batch_size * k):
            if i % k == 0: reshaped_has_explicit_eos.append([])
            reshaped_has_explicit_eos[-1].append(has_explicit_eos_flat[i])

        reshaped_per_token_log_probs: List[List[List[torch.Tensor]]] = []
        for i in range(batch_size * k):
            if i % k == 0: reshaped_per_token_log_probs.append([])
            reshaped_per_token_log_probs[-1].append(per_token_log_probs_flat[i])


        final_generated_ids_grouped: List[List[torch.Tensor]] = []
        final_decoded_concepts_grouped: List[List[str]] = []
        final_per_token_log_probs_grouped: List[List[List[torch.Tensor]]] = []
        final_is_grammatically_invalid_grouped: List[List[bool]] = []
        final_semantic_tokens_grouped: List[List[List[int]]] = []
        final_has_explicit_eos_grouped: List[List[bool]] = []

        sorted_log_probs_tensor, sort_indices = reshaped_cumulative_log_probs_tensor.sort(dim=1, descending=True)
        
        for i in range(batch_size):
            current_sort_indices_list = sort_indices[i].tolist()

            current_sorted_ids = torch.gather(
                reshaped_generated_ids[i], 0, sort_indices[i].unsqueeze(-1).expand(-1, actual_seq_len)
            )
            final_generated_ids_grouped.append(current_sorted_ids)

            final_decoded_concepts_grouped.append([reshaped_decoded_concepts[i][j] for j in current_sort_indices_list])
            final_per_token_log_probs_grouped.append([reshaped_per_token_log_probs[i][j] for j in current_sort_indices_list])
            final_is_grammatically_invalid_grouped.append([reshaped_is_grammatically_invalid[i][j].item() for j in current_sort_indices_list])
            final_semantic_tokens_grouped.append([reshaped_semantic_tokens[i][j] for j in current_sort_indices_list])
            final_has_explicit_eos_grouped.append([reshaped_has_explicit_eos[i][j] for j in current_sort_indices_list])
            
        return (final_generated_ids_grouped, final_decoded_concepts_grouped,
                final_per_token_log_probs_grouped, final_is_grammatically_invalid_grouped,
                final_semantic_tokens_grouped, final_has_explicit_eos_grouped)


    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, heuristic_function:ConSynHeuristic, cshs: ConSynHypothesisSpace, device:Optional[torch.device] = None, decoding_strategy: str = "multinomial", temperature: float = 1.0, 
                top_k: int = 0, top_p: float = 0.0, num_k_predictions: int = 1, hypothesis_threshold_score: float = 0.65, fit_mode:Optional[bool] = None) -> float:
        
        device = device or self.device
    
        self.model.to(device)
        self.model.eval()
        total_rewards = 0.0
        num_batches = 0
        
        eval_batched_hypothesis = []
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            enc_mask = batch['attention_mask'].to(device)
            
            generated_ids_flat, per_token_log_probs_flat, \
            is_grammatically_invalid_flat, semantic_tokens_flat, \
            has_explicit_eos_flat = self.model.generate_for_rl(input_ids, segment_ids, enc_mask,
                self.grammar_parser, self.max_decoder_seq_len, k=num_k_predictions, decoding_strategy=decoding_strategy,
                temperature=temperature, top_k=top_k, top_p=top_p
            )

            (generated_ids_grouped, decoded_concepts_grouped,
             per_token_log_probs_grouped, is_grammatically_invalid_grouped,
             semantic_tokens_grouped, has_explicit_eos_grouped) = \
                self.process_generated_output(
                    generated_ids_flat, per_token_log_probs_flat,
                    is_grammatically_invalid_flat, semantic_tokens_flat,
                    has_explicit_eos_flat, input_ids.size(0), num_k_predictions, device
                )
            
            rewards, hypothesis = self.reward_function.forward(
                semantic_tokens_grouped,
                batch,
                is_grammatically_invalid_grouped,
                has_explicit_eos_grouped,
                hypothesis_threshold_score,
                heuristic_function,
                cshs,
                fit_mode,
                self.verbose
            )

            if hypothesis:
                eval_batched_hypothesis.append(hypothesis)
            
            total_rewards += rewards.mean().item()
            num_batches += 1

        avg_reward = total_rewards / num_batches
        return avg_reward, eval_batched_hypothesis