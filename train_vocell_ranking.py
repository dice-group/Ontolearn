"""
train_vocell_ranking.py
=======================
V-Net training with **MarginRankingLoss** — framing concept scoring as a
pairwise ranking task instead of F1 regression (MSE).

Why ranking?
------------
During beam search we only need to know *which concept to expand next*,
not its exact future F1 value.  MarginRankingLoss directly optimises the
ordering:

    loss(s_i, s_j, y) = max(0, -y * (s_i - s_j) + margin)

where  y = +1  if  f1_i > f1_j  (concept i is better)
       y = -1  otherwise

Two architectures are trained:

  ConceptVNetRanking      — identical structure to the existing ConceptVNet
                            (same 3-layer MLP, same param count) so that
                            checkpoints are drop-in compatible with vocell.py

  ConceptVNetRankingLarge — compact ~10K-parameter network (single hidden
                            layer of width 16 by default) that is much
                            smaller than ConceptVNetRanking yet still
                            learns a ranking over concepts

Usage
-----
# Family loocv (same arch):
python train_vocell_ranking.py --strategy loocv --arch same

# Family loocv (large arch, 10k hidden):
python train_vocell_ranking.py --strategy loocv --arch large

# Bootstrap (same arch):
python train_vocell_ranking.py \\
    --lps_file LPs/Carcinogenesis/lps.json \\
    --dataset_file vnet_search_data_carcinogenesis.json \\
    --embeddings  ../Ontolearn_ISWC/datasets/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv \\
    --strategy bootstrap --arch same --output_dir Carcinogenesis_ranking

Checkpoint format
-----------------
The saved .pt files contain:
  model_state_dict  – state_dict() of the trained network
  embedding_dim     – int, inferred from the embeddings CSV
  arch              – 'same' | 'large'
  loss              – 'ranking'
  margin            – float, the MarginRankingLoss margin used
  training_strategy – 'loocv' | 'bootstrap'
  …plus strategy-specific metadata (lp_name, training_lps, best_f1, etc.)

Loading in vocell.py
--------------------
Both architectures expose the same forward(X) → (N,) interface as
ConceptVNet, so the existing BeamVNet.load() path works unchanged when
arch='same'.  For arch='large' use ConceptVNetRankingLarge directly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

# ── Re-use all dataset helpers from the original trainer ─────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from train_vocell_v_net import (
    get_mean_emb,
    compute_best_reachable_from_nodes,
    build_dataset_from_nodes,
    build_bootstrap_datasets,
)


# =============================================================================
#  Architecture 1 — Same structure as ConceptVNet  (drop-in compatible)
# =============================================================================

class ConceptVNetRanking(nn.Module):
    """
    Identical layer structure to ConceptVNet in train_vocell_v_net.py.

    Input : (N, 3, embedding_dim)
    Output: (N,)  — raw scores (no Sigmoid, not constrained to [0,1])

    Trained with MarginRankingLoss instead of MSELoss.
    The raw (unbounded) output is fine for ranking; Sigmoid is dropped so the
    gradients from the ranking loss flow more freely.

    Parameter count (embedding_dim=200):
        Linear(600→400)   240 400
        Linear(400→200)    80 200
        Linear(200→1)         201
        LayerNorm ×2          800
        ─────────────────────────
        Total             ~321 601
    """

    def __init__(self, embedding_dim: int, device: str = 'cpu',
                 margin: float = 0.05):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.margin        = margin

        in_dim = 3 * embedding_dim
        h      = 2 * embedding_dim

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, h, device=device),
            nn.LayerNorm(h, device=device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, embedding_dim, device=device),
            nn.LayerNorm(embedding_dim, device=device),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, device=device),
            # No Sigmoid — raw scores for ranking
        )
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, 3, embedding_dim) → (N,)"""
        return self.net(X).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
#  Architecture 2 — ~10K-parameter compact network
# =============================================================================

class ConceptVNetRankingLarge(nn.Module):
    """
    Compact network targeting ~10 000 trainable parameters.

    For embedding_dim=200 the input is 3×200=600.  A single hidden layer
    of width h gives  (600+1)·h + (h+1) = 602h + 1  parameters.  Setting
    h=16 gives  602×16 + 1 = 9 633 ≈ 10K.

    Architecture:
        Flatten
        Linear(3·dim → hidden_dim)   default hidden_dim=16
        LayerNorm(hidden_dim)
        ReLU
        Linear(hidden_dim → 1)       no Sigmoid — raw ranking scores

    Parameter count (embedding_dim=200, hidden_dim=16):
        Linear(600→16)    9 616
        LayerNorm(16)        32
        Linear(16→1)         17
        ──────────────────────
        Total             9 665
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 16,
                 device: str = 'cpu', margin: float = 0.05):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.margin        = margin

        in_dim = 3 * embedding_dim

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim,     hidden_dim, device=device),
            nn.LayerNorm(hidden_dim,           device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1,           device=device),
            # No Sigmoid — raw scores for ranking
        )
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, 3, embedding_dim) → (N,)"""
        return self.net(X).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
#  Pairwise dataset builder
# =============================================================================

def build_ranking_pairs(
    X: torch.Tensor,
    y: torch.Tensor,
    max_pairs: int = 50_000,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build pairwise training examples for MarginRankingLoss from (X, y).

    For each pair (i, j) where y[i] ≠ y[j]:
        X1[k] = X[i],   X2[k] = X[j],   target[k] = +1 if y[i] > y[j] else -1

    To keep training tractable we sample at most `max_pairs` pairs
    (stratified: half where y[i]>y[j], half where y[i]<y[j]).

    Returns
    -------
    X1     : (K, 3, dim)
    X2     : (K, 3, dim)
    target : (K,)   dtype=float32, values ∈ {+1, -1}
    """
    rng = random.Random(seed)
    N   = len(y)

    # Build index groups by quantised F1 bucket (avoid float equality issues)
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, val in enumerate(y.tolist()):
        buckets[round(val * 1000)].append(idx)   # 0.001 resolution

    bucket_keys = sorted(buckets.keys())

    if len(bucket_keys) < 2:
        raise ValueError(
            f"build_ranking_pairs: all {N} concepts have the same F1 value "
            f"({bucket_keys[0]/1000:.3f}) — cannot build ranking pairs. "
            "Check that the dataset contains diverse F1 scores."
        )

    X1_list:  List[torch.Tensor] = []
    X2_list:  List[torch.Tensor] = []
    tgt_list: List[float]        = []

    half = max_pairs // 2

    # Positive pairs: y[i] > y[j]  → target = +1
    for _ in range(half):
        # Pick two different buckets
        hi_key = rng.choice(bucket_keys[1:])  # at least bucket index 1
        lo_candidates = [k for k in bucket_keys if k < hi_key]
        if not lo_candidates:
            continue
        lo_key = rng.choice(lo_candidates)
        i = rng.choice(buckets[hi_key])
        j = rng.choice(buckets[lo_key])
        X1_list.append(X[i])
        X2_list.append(X[j])
        tgt_list.append(1.0)

    # Negative pairs: y[i] < y[j]  → target = -1  (mirror)
    for _ in range(half):
        hi_key = rng.choice(bucket_keys[1:])
        lo_candidates = [k for k in bucket_keys if k < hi_key]
        if not lo_candidates:
            continue
        lo_key = rng.choice(lo_candidates)
        i = rng.choice(buckets[lo_key])
        j = rng.choice(buckets[hi_key])
        X1_list.append(X[i])
        X2_list.append(X[j])
        tgt_list.append(-1.0)

    if not X1_list:
        raise ValueError("Could not build any ranking pairs — all concepts "
                         "have the same best-reachable F1.")

    X1     = torch.stack(X1_list, dim=0)
    X2     = torch.stack(X2_list, dim=0)
    target = torch.tensor(tgt_list, dtype=torch.float32)
    return X1, X2, target


# =============================================================================
#  Training loop
# =============================================================================

def train_ranking(
    X:          torch.Tensor,
    y:          torch.Tensor,
    arch:       str   = 'same',
    epochs:     int   = 200,
    batch_size: int   = 512,
    lr:         float = 1e-3,
    margin:     float = 0.05,
    hidden_dim: int   = 10_000,
    max_pairs:  int   = 50_000,
    device:     str   = 'cpu',
    seed:       int   = 42,
) -> nn.Module:
    """
    Train a ranking V-Net on the (X, y) dataset.

    Parameters
    ----------
    X          : (N, 3, embedding_dim) concept feature tensors
    y          : (N,) best-reachable F1 targets (used *only* to define pair ordering)
    arch       : 'same'  → ConceptVNetRanking  (same size as ConceptVNet)
                 'large' → ConceptVNetRankingLarge (hidden_dim wide first layer)
    epochs     : number of training epochs
    batch_size : pairs per gradient step
    lr         : Adam learning rate
    margin     : MarginRankingLoss margin (concepts that differ by less than
                 this are not penalised)
    hidden_dim : hidden-layer width for arch='large' (default 16 → ~10K params)
    max_pairs  : total pairwise training examples to sample
    device     : torch device string
    seed       : RNG seed for reproducibility

    Returns
    -------
    Trained model in eval mode.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    _, _, embedding_dim = X.shape

    # Instantiate architecture
    if arch == 'same':
        net = ConceptVNetRanking(embedding_dim, device=device, margin=margin)
    elif arch == 'large':
        net = ConceptVNetRankingLarge(embedding_dim, hidden_dim=hidden_dim,
                                      device=device, margin=margin)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'same' or 'large'.")

    n_params = net.count_parameters()
    print(f"\n{'='*60}")
    print(f"Ranking V-Net  |  arch={arch}  |  params={n_params:,}")
    print(f"embedding_dim={embedding_dim}  "
          f"hidden_dim={hidden_dim if arch=='large' else f'2x{embedding_dim} (same arch)'}  "
          f"margin={margin}  params={n_params:,}")
    print(f"{'='*60}")

    # Build pairwise dataset
    print(f"Building ranking pairs from {len(y)} concepts …")
    X1, X2, target = build_ranking_pairs(X, y, max_pairs=max_pairs, seed=seed)
    print(f"  Generated {len(target):,} pairs  "
          f"(+1: {(target>0).sum().item():,}  -1: {(target<0).sum().item():,})")

    X1     = X1.to(device)
    X2     = X2.to(device)
    target = target.to(device)
    K      = len(target)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=15)

    print(f"\n{'Epoch':>6}  {'RankLoss':>12}  {'LR':>10}")
    print("-" * 34)

    loss_history = []
    for epoch in range(1, epochs + 1):
        net.train()
        perm       = torch.randperm(K, device=device)
        epoch_loss = 0.0
        steps      = 0

        for start in range(0, K, batch_size):
            idx  = perm[start:start + batch_size]
            x1b  = X1[idx]
            x2b  = X2[idx]
            tb   = target[idx]

            s1   = net(x1b)   # (B,)
            s2   = net(x2b)   # (B,)
            loss = net.loss_fn(s1, s2, tb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item()
            steps      += 1

        avg = epoch_loss / max(steps, 1)
        sch.step(avg)
        current_lr = opt.param_groups[0]['lr']
        loss_history.append({'epoch': epoch, 'loss': avg, 'lr': current_lr})
        print(f"{epoch:>6}  {avg:>12.6f}  {current_lr:>10.2e}")

    net.eval()
    return net, loss_history


# =============================================================================
#  LOOCV and bootstrap wrappers  (mirror train_vocell_v_net.py structure)
# =============================================================================

def train_ranking_for_lp(
    lp_name: str,
    dataset: Dict,
    df:      pd.DataFrame,
    epochs:  int,
    device:  str,
    arch:    str   = 'same',
    margin:  float = 0.05,
    hidden_dim: int = 10_000,
    max_pairs: int = 50_000,
) -> Tuple[nn.Module, List[str], float]:
    """Leave-one-LP-out: train on all LPs except `lp_name`."""
    training_lps = {n: d for n, d in dataset.items() if n != lp_name}

    X_all, y_all = [], []
    for name, lp_data in training_lps.items():
        nodes    = lp_data['nodes']
        pos_iris = lp_data['positive_examples']
        neg_iris = lp_data['negative_examples']
        emb_pos  = get_mean_emb(pos_iris, df)
        emb_neg  = get_mean_emb(neg_iris, df)
        best_r   = compute_best_reachable_from_nodes(nodes)
        Xep, yep = build_dataset_from_nodes(nodes, best_r, emb_pos, emb_neg, df)
        X_all.append(Xep)
        y_all.append(yep)

    X = torch.cat(X_all, dim=0)
    y = torch.cat(y_all, dim=0)
    net, loss_history = train_ranking(X, y, arch=arch, epochs=epochs, device=device,
                        margin=margin, hidden_dim=hidden_dim, max_pairs=max_pairs)
    return net, list(training_lps.keys()), float(y.max()), loss_history


def train_ranking_bootstrap(
    dataset:        Dict,
    lp_names:       List[str],
    df:             pd.DataFrame,
    epochs:         int,
    device:         str,
    arch:           str   = 'same',
    margin:         float = 0.05,
    hidden_dim:     int   = 10_000,
    sample_lp_frac: float = 0.5,
    sample_ex_frac: float = 0.5,
    n_rounds:       int   = 5,
    max_pairs:      int   = 50_000,
    seed:           int   = 42,
) -> Tuple[nn.Module, float]:
    """Bootstrap strategy: pool multiple subsampled LP rounds."""
    n = len(lp_names)
    print(f"\nBootstrap ranking: {n} LP{'s' if n != 1 else ''}  |  "
          f"{n_rounds} rounds  |  LP={sample_lp_frac*100:.0f}%  "
          f"ex={sample_ex_frac*100:.0f}%")

    # Re-use the mean-path from train_vocell_v_net (returns X, y tensors)
    X, y = build_bootstrap_datasets(
        dataset, lp_names, sample_lp_frac, sample_ex_frac,
        n_rounds, df, agg_type='mean', seed=seed,
    )
    net, loss_history = train_ranking(X, y, arch=arch, epochs=epochs, device=device,
                            margin=margin, hidden_dim=hidden_dim,
                            max_pairs=max_pairs, seed=seed)
    best_f1 = float(y.max()) if y.numel() > 0 else 0.0
    return net, best_f1, loss_history


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VOCELL V-Net training with MarginRankingLoss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--lps_file',    default='LPs/Family/lps_difficult.json')
    parser.add_argument('--dataset_file',default='vnet_search_data_difficult.json')
    parser.add_argument('--embeddings',  default='Experiments/embeddings/Keci_entity_embeddings.csv')
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--device',      default='cpu')
    parser.add_argument('--arch',        default='same', choices=['same', 'large'],
                        help=("'same' = identical size to ConceptVNet; "
                              "'large' = 10K-wide first hidden layer"))
    parser.add_argument('--hidden_dim',  type=int,   default=16,
                        help='Hidden-layer width for arch=large (default 16 → ~10K params for dim=200).')
    parser.add_argument('--margin',      type=float, default=0.05,
                        help='MarginRankingLoss margin.')
    parser.add_argument('--max_pairs',   type=int,   default=50_000,
                        help='Max pairwise training samples per epoch.')
    parser.add_argument('--strategy',    default='auto',
                        choices=['auto', 'loocv', 'bootstrap'])
    parser.add_argument('--loocv_threshold', type=int, default=10)
    parser.add_argument('--sample_lp_frac',  type=float, default=0.5)
    parser.add_argument('--sample_ex_frac',  type=float, default=0.5)
    parser.add_argument('--n_rounds',        type=int,   default=5)
    parser.add_argument('--output_dir',      default=None,
                        help='Checkpoint directory (default: Family_ranking_<arch>).')
    args = parser.parse_args()

    output_dir = args.output_dir or f'Family_ranking_{args.arch}'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("VOCELL Ranking V-Net training")
    print(f"  arch       : {args.arch}")
    print(f"  strategy   : {args.strategy}")
    print(f"  margin     : {args.margin}")
    print(f"  hidden_dim : {args.hidden_dim}  (only used for arch=large)")
    print(f"  max_pairs  : {args.max_pairs:,}")
    print(f"  epochs     : {args.epochs}")
    print(f"  device     : {args.device}")
    print(f"  lps_file   : {args.lps_file}")
    print(f"  dataset    : {args.dataset_file}")
    print(f"  embeddings : {args.embeddings}")
    print(f"  output_dir : {output_dir}")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────
    with open(args.lps_file) as f:
        _raw = json.load(f)
    lps_data     = _raw.get('problems', _raw) if isinstance(_raw, dict) else {}
    all_lp_names = list(lps_data.keys())

    with open(args.dataset_file) as f:
        dataset = json.load(f)

    all_lp_names = [n for n in all_lp_names if n in dataset]
    print(f"  LPs available: {len(all_lp_names)}")

    df = pd.read_csv(args.embeddings, index_col=0).astype('float32')
    embedding_dim = df.shape[1]
    print(f"  Embedding dim: {embedding_dim}")

    # ── Strategy ─────────────────────────────────────────────
    if args.strategy == 'loocv':
        use_loocv = True
    elif args.strategy == 'bootstrap':
        use_loocv = False
    else:
        use_loocv = 1 < len(all_lp_names) <= args.loocv_threshold

    # ── Common checkpoint metadata ───────────────────────────
    base_meta = dict(
        embedding_dim     = embedding_dim,
        arch              = args.arch,
        hidden_dim        = args.hidden_dim,
        loss              = 'ranking',
        margin            = args.margin,
    )

    if use_loocv:
        print(f"\nStrategy: Leave-One-Out  ({len(all_lp_names)} LPs)")
        for lp_name in all_lp_names:
            print(f"\n{'='*60}\nHeld-out LP: {lp_name}\n{'='*60}")
            net, training_lps, best_f1, loss_history = train_ranking_for_lp(
                lp_name, dataset, df,
                epochs     = args.epochs,
                device     = args.device,
                arch       = args.arch,
                margin     = args.margin,
                hidden_dim = args.hidden_dim,
                max_pairs  = args.max_pairs,
            )
            save_path = os.path.join(
                output_dir, f'vocell_v_net_{lp_name}_ranking_{args.arch}.pt')
            torch.save({
                **base_meta,
                'model_state_dict'  : net.state_dict(),
                'training_strategy' : 'loocv',
                'lp_name'           : lp_name,
                'training_lps'      : training_lps,
                'best_f1_in_training': best_f1,
            }, save_path)
            loss_path = save_path.replace('.pt', '_loss.json')
            with open(loss_path, 'w') as _f:
                json.dump(loss_history, _f, indent=2)
            print(f"Saved → {save_path}")
            print(f"Saved → {loss_path}")

    else:
        print(f"\nStrategy: Bootstrap  ({len(all_lp_names)} LPs)")
        net, best_f1, loss_history = train_ranking_bootstrap(
            dataset, all_lp_names, df,
            epochs         = args.epochs,
            device         = args.device,
            arch           = args.arch,
            margin         = args.margin,
            hidden_dim     = args.hidden_dim,
            sample_lp_frac = args.sample_lp_frac,
            sample_ex_frac = args.sample_ex_frac,
            n_rounds       = args.n_rounds,
            max_pairs      = args.max_pairs,
        )
        save_path = os.path.join(
            output_dir, f'vocell_v_net_bootstrap_ranking_{args.arch}.pt')
        torch.save({
            **base_meta,
            'model_state_dict'   : net.state_dict(),
            'training_strategy'  : 'bootstrap',
            'all_lp_names'       : all_lp_names,
            'sample_lp_frac'     : args.sample_lp_frac,
            'sample_ex_frac'     : args.sample_ex_frac,
            'n_rounds'           : args.n_rounds,
            'best_f1_in_training': best_f1,
        }, save_path)
        loss_path = save_path.replace('.pt', '_loss.json')
        with open(loss_path, 'w') as _f:
            json.dump(loss_history, _f, indent=2)
        print(f"Saved → {save_path}")
        print(f"Saved → {loss_path}")


if __name__ == '__main__':
    main()

# ── Quick-run examples ─────────────────────────────────────────────────────
# Family LOOCV — same arch:
#   python train_vocell_ranking.py --strategy loocv --arch same
#
# Family LOOCV — ~10K-param arch:
#   python train_vocell_ranking.py --strategy loocv --arch large
#
# Carcinogenesis bootstrap — same arch:
#   python train_vocell_ranking.py \
#       --lps_file LPs/Carcinogenesis/lps.json \
#       --dataset_file vnet_search_data_carcinogenesis.json \
#       --embeddings ../Ontolearn_ISWC/datasets/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv \
#       --strategy bootstrap --arch same --output_dir Carcinogenesis_ranking
#
# Carcinogenesis bootstrap — ~10K-param arch:
#   python train_vocell_ranking.py \
#       --lps_file LPs/Carcinogenesis/lps.json \
#       --dataset_file vnet_search_data_carcinogenesis.json \
#       --embeddings ../Ontolearn_ISWC/datasets/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv \
#       --strategy bootstrap --arch large --output_dir Carcinogenesis_ranking_large
