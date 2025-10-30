import torch
import torch.nn as nn

from ontolearn.consyn.model.embeddings import InputEmbeddingLayer
from ontolearn.consyn.model.attention import MultiHeadCrossAttention, MultiHeadSelfAttention

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_prob: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_prob)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout_prob)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(ff_dim, embed_dim),
        )

        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                target_attention_mask: torch.Tensor = None,
                encoder_attention_mask: torch.Tensor = None) -> torch.Tensor:

        if self.pre_norm:
            # Pre-LN version
            residual = x
            x = self.norm1(x)
            x = residual + self.dropout1(self.self_attn(x, target_attention_mask))

            residual = x
            x = self.norm2(x)
            x = residual + self.dropout2(self.cross_attn(x, encoder_output, encoder_attention_mask))

            residual = x
            x = self.norm3(x)
            x = residual + self.dropout3(self.ffn(x))

        else:
            # Post-LN version
            attn_output = self.self_attn(x, target_attention_mask)
            x = self.norm1(x + self.dropout1(attn_output))

            cross_attn_output = self.cross_attn(x, encoder_output, encoder_attention_mask)
            x = self.norm2(x + self.dropout2(cross_attn_output))

            ffn_output = self.ffn(x)
            x = self.norm3(x + self.dropout3(ffn_output))

        return x
    

class Decoder(nn.Module):
    def __init__(self, target_vocab_size: int, embed_dim: int, num_layers: int = 6, num_heads: int = 8,
                 ff_dim: int = 2048, dropout_prob: float = 0.1, use_checkpointing: bool = False,
                 pre_norm: bool = False):
        super().__init__()
        self.embedding_layer = InputEmbeddingLayer(target_vocab_size, embed_dim, num_segments=None, dropout_prob=dropout_prob)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout_prob, pre_norm=pre_norm)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, target_vocab_size)

        self.use_checkpointing = use_checkpointing

    def forward(self, target_ids: torch.Tensor, encoder_output: torch.Tensor,
                target_attention_mask: torch.Tensor = None,
                encoder_attention_mask: torch.Tensor = None) -> torch.Tensor:

        x = self.embedding_layer(target_ids)

        for layer in self.decoder_layers:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, encoder_output, target_attention_mask, encoder_attention_mask)
            else:
                x = layer(x, encoder_output, target_attention_mask, encoder_attention_mask)

        x = self.norm(x)
        return self.output_proj(x)