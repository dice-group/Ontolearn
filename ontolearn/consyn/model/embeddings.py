import math
import torch
import torch.nn as nn

class InputEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_segments: int = None, dropout_prob: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.segment_embeddings = nn.Embedding(num_segments, embed_dim) if num_segments is not None else None
        self.dropout = nn.Dropout(dropout_prob)
        self.embed_dim = embed_dim

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor = None) -> torch.Tensor:
        token_embeds = self.token_embeddings(input_ids)
        embeddings = token_embeds

        if self.segment_embeddings is not None:
            if segment_ids is None:
                raise ValueError("segment_ids must be provided if num_segments was specified.")
            if segment_ids.shape != input_ids.shape:
                raise ValueError("segment_ids shape mismatch with input_ids")
            segment_embeds = self.segment_embeddings(segment_ids)
            embeddings = embeddings + segment_embeds

        embeddings = embeddings * math.sqrt(self.embed_dim)  # Optional scaling
        return self.dropout(embeddings)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be an even number.")
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0] and \
           self.cos_cached.device == device and self.cos_cached.dtype == dtype:
            return

        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))
        emb = torch.cat([freqs, freqs], dim=-1)

        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x: torch.Tensor, seq_len: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]  # sequence length dimension

        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    def rotate_half(x_val):
        x1, x2 = x_val[..., :x_val.shape[-1] // 2], x_val[..., x_val.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    return (x * cos[None, None, :, :]) + (rotate_half(x) * sin[None, None, :, :])