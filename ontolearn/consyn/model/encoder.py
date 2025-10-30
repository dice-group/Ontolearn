import torch
import torch.nn as nn

from ontolearn.consyn.model.embeddings import InputEmbeddingLayer
from ontolearn.consyn.model.attention import MultiHeadSelfAttention


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_prob: float = 0.1, pre_norm: bool = False):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_prob)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob), 
            nn.Linear(ff_dim, embed_dim),
        )

        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if self.pre_norm:
            # Pre-LN: norm -> sublayer -> residual
            x_norm = self.norm1(x)
            x = x + self.dropout1(self.self_attn(x_norm, attention_mask))

            x_norm = self.norm2(x)
            x = x + self.dropout2(self.ffn(x_norm))
        else:
            # Post-LN: sublayer -> residual -> norm
            attn_output = self.self_attn(x, attention_mask)
            x = self.norm1(x + self.dropout1(attn_output))

            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_output))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_segments: int = None, num_layers: int = 6,
                 num_heads: int = 8, ff_dim: int = 2048, dropout_prob: float = 0.1,
                 use_checkpointing: bool = False, pre_norm: bool = False):
        super().__init__()
        self.embedding_layer = InputEmbeddingLayer(vocab_size, embed_dim, num_segments, dropout_prob)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim, dropout_prob, pre_norm)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.use_checkpointing = use_checkpointing

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding_layer(input_ids, segment_ids)

        for layer in self.encoder_layers:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask)
            else:
                x = layer(x, attention_mask)

        x = self.norm(x)
        return x