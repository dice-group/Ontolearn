"""
concept_aggregators.py
======================
Three aggregation strategies for mapping a variable-length set of entity
embeddings to a fixed-size vector, and the unified ConceptVNet architecture
used by both train_vocell_v_net.py (offline training) and vocell.py (inference).

Aggregation strategies
----------------------
  'mean'           — simple average (no learnable parameters)
  'deepsets'       — DeepSets (Zaheer et al. 2017): φ → mean-pool → ρ
  'settransformer' — Simplified SetTransformer (Lee et al. 2019): SAB + PMA

All aggregators
  Input : (|S|, dim)  — embedding matrix of the set; (0, dim) tolerated
  Output: (dim,)      — fixed-size representation

ConceptVNet
  Predicts best-reachable F1 for a concept C given E+ and E−.
  Uses the chosen aggregator for all three sets (I(C), E+, E−), then
  passes the concatenated (3·dim,) vector through an MLP head.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn

AGG_TYPES = ('mean', 'deepsets', 'settransformer')


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_instance_emb_matrix(iris: List[str], df: pd.DataFrame) -> torch.Tensor:
    """
    Look up entity embeddings for a list of IRI strings.
    Returns  (|valid|, dim)  float32.
    Returns  (0, dim)        if no IRI is found (handled gracefully by all aggregators).
    """
    valid = [i for i in iris if i in df.index]
    dim   = df.shape[1]
    if not valid:
        return torch.zeros(0, dim, dtype=torch.float32)
    return torch.tensor(df.loc[valid].values.astype('float32'))


def get_mean_emb(iris: List[str], df: pd.DataFrame) -> torch.Tensor:
    """Mean embedding. Returns (1, 1, dim) for backward compatibility."""
    dim   = df.shape[1]
    valid = [i for i in iris if i in df.index]
    if not valid:
        return torch.zeros(1, 1, dim, dtype=torch.float32)
    arr = df.loc[valid].values.astype('float32')
    return torch.tensor(arr.mean(axis=0)).view(1, 1, dim)


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregator modules
# ─────────────────────────────────────────────────────────────────────────────

class MeanAggregator(nn.Module):
    """
    Simple mean pooling — no learnable parameters.
    (|S|, dim) → (dim,)
    """

    def __init__(self, dim: int, device: str = 'cpu'):
        super().__init__()
        self.dim = dim
        # Dummy buffer so .to(device) propagates uniformly across all aggregators
        self.register_buffer('_dummy', torch.zeros(1, device=device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[0] == 0:
            return torch.zeros(self.dim, device=self._dummy.device)
        return X.mean(dim=0)


class DeepSetsAggregator(nn.Module):
    """
    DeepSets (Zaheer et al. 2017):  ρ( mean_{x ∈ S}  φ(x) )

    φ: per-element encoder  (dim → dim)
    ρ: set-level decoder    (dim → dim)

    (|S|, dim) → (dim,)
    """

    def __init__(self, dim: int, device: str = 'cpu'):
        super().__init__()
        self.dim = dim
        self.phi = nn.Sequential(
            nn.Linear(dim, dim, device=device),
            nn.LayerNorm(dim, device=device),
            nn.ReLU(),
            nn.Linear(dim, dim, device=device),
        )
        self.rho = nn.Sequential(
            nn.Linear(dim, dim, device=device),
            nn.LayerNorm(dim, device=device),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[0] == 0:
            dev = next(self.parameters()).device
            return torch.zeros(self.dim, device=dev)
        return self.rho(self.phi(X).mean(dim=0))  # (dim,)


class SetTransformerAggregator(nn.Module):
    """
    Simplified SetTransformer (Lee et al. 2019):
      SAB (self-attention block) → PMA (pooling by multi-head attention)

    SAB : X  → MHA(X, X, X) + residual + LN            → (|S|, dim)
    PMA : seed → MHA(seed, SAB(X), SAB(X)) + residual + LN → FF → (dim,)

    (|S|, dim) → (dim,)
    """

    def __init__(self, dim: int, num_heads: int = 4, device: str = 'cpu'):
        super().__init__()
        # Ensure dim is divisible by num_heads
        print(f"Initializing SetTransformerAggregator with dim={dim} and num_heads={num_heads}")
        while num_heads > 1 and dim % num_heads != 0:
            num_heads -= 1
        self.dim = dim

        self.sab_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, device=device)
        self.sab_norm = nn.LayerNorm(dim, device=device)

        # Learned seed vector for PMA
        self.seed = nn.Parameter(torch.randn(1, 1, dim, device=device))

        self.pma_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, device=device)
        self.pma_norm = nn.LayerNorm(dim, device=device)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim, device=device),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[0] == 0:
            return torch.zeros(self.dim, device=self.seed.device)

        X   = X.unsqueeze(0)                         # (1, |S|, dim)
        # SAB
        enc, _ = self.sab_attn(X, X, X)             # (1, |S|, dim)
        enc    = self.sab_norm(enc + X)              # residual + LN
        # PMA
        seed   = self.seed                           # (1, 1, dim)
        out, _ = self.pma_attn(seed, enc, enc)       # (1, 1, dim)
        out    = self.pma_norm(out + seed)           # residual + LN
        return self.ff(out.squeeze(0).squeeze(0))    # (dim,)


def build_aggregator(agg_type: str, dim: int, device: str = 'cpu') -> nn.Module:
    """Factory: returns the aggregator module for the given agg_type."""
    agg_type = agg_type.lower()
    if agg_type == 'mean':
        return MeanAggregator(dim, device)
    elif agg_type == 'deepsets':
        return DeepSetsAggregator(dim, device)
    elif agg_type in ('settransformer', 'transformer'):
        return SetTransformerAggregator(dim=dim, device=device)
    else:
        raise ValueError(f"Unknown agg_type '{agg_type}'. Valid options: {AGG_TYPES}")


# ─────────────────────────────────────────────────────────────────────────────
#  ConceptVNet — shared architecture for training and inference
# ─────────────────────────────────────────────────────────────────────────────

class ConceptVNet(nn.Module):
    """
    Predicts best-reachable F1 from the embedding context of a concept.

    Architecture
    ------------
    1. Three variable-length sets (I(C), E+, E−) are each aggregated
       to a fixed (embedding_dim,) vector by self.aggregator.
    2. The three vectors are concatenated → (3 · embedding_dim,).
    3. An MLP prediction head maps this to a scalar in [0, 1].

    agg_type options
    ----------------
      'mean'           — simple average (no learnable parameters in aggregator)
      'deepsets'       — DeepSets (Zaheer et al. 2017)
      'settransformer' — Simplified SetTransformer (Lee et al. 2019)

    This class is used identically by:
      - train_vocell_v_net.py  (offline training)
      - vocell.py              (inference-time beam ranking)
    """

    def __init__(self, embedding_dim: int, agg_type: str = 'mean', device: str = 'cpu'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.agg_type      = agg_type
        self.aggregator    = build_aggregator(agg_type, embedding_dim, device)

        in_dim = 3 * embedding_dim
        h      = 2 * embedding_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, h, device=device),
            nn.LayerNorm(h, device=device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, embedding_dim, device=device),
            nn.LayerNorm(embedding_dim, device=device),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, device=device),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def aggregate(self, emb_matrix: torch.Tensor) -> torch.Tensor:
        """(|S|, dim) → (dim,)"""
        return self.aggregator(emb_matrix)

    def score_candidates(
        self,
        candidate_mats: List[torch.Tensor],
        pos_mat:        torch.Tensor,
        neg_mat:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a list of candidate concepts in a single forward pass.

        candidate_mats : List of N tensors, each (|S_i|, dim)
        pos_mat        : (|E+|, dim)
        neg_mat        : (|E−|, dim)

        Returns: (N,) score tensor in [0, 1]
        """
        N     = len(candidate_mats)
        p_agg = self.aggregate(pos_mat)                               # (dim,)
        n_agg = self.aggregate(neg_mat)                               # (dim,)
        c_aggs = torch.stack([self.aggregate(m) for m in candidate_mats])  # (N, dim)
        X = torch.cat(
            [c_aggs, p_agg.expand(N, -1), n_agg.expand(N, -1)],
            dim=1,
        )                                                              # (N, 3·dim)
        return self.head(X).squeeze(-1)                               # (N,)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Legacy interface: X is (N, 3, dim) pre-aggregated (used for 'mean' mode
        in online V-learning replay where experiences store mean embeddings).
        """
        return self.head(X.flatten(start_dim=1)).squeeze(-1)
