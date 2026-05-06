"""
train_vocell_v_net.py  —  Offline V-Net training for VOCELL
============================================================

Goal
----
Train a V-network that, given a concept C and the positive/negative example
sets (E+, E-), predicts the BEST F1 score reachable by continuing the beam
search from C.  We call this  y_C = max_{D reachable from C} F1(D, E+, E-).

This is used at inference time (vocell.py) to guide BEAM SELECTION:
instead of always expanding the top-beam_width concepts by their own F1,
we use the V-net to pick which concepts to expand next — preferring those
from which a high F1 is achievable deeper in the search tree.  This reduces
the total number of concepts explored while preserving solution quality.

Note: PruneCELBasedRefinement already evaluates every concept via SPARQL
during refine(), so filtering BEFORE evaluate() saves nothing.  The savings
come entirely from smarter beam expansion decisions.

Three-file pipeline
-------------------
  Step 1 — generate_vnet_dataset.py
    Run beam search on every LP, record the search tree
    (concept_str, f1, depth, parent_str, instance_iris) → JSON.
    Expensive one-time step that needs SPARQL + reasoner.

  Step 2 — this file
    Leave-one-LP-out training:
      • For each training LP load nodes from JSON
      • y_C = bottom-up DP max over best F1 reachable from C
      • x_C = [mean_emb(instance_iris_C), mean_emb(E+), mean_emb(E-)]
      • MSE training of ConceptVNet
    No KB / SPARQL queries needed here — instance_iris already in JSON.

  Step 3 — vocell.py
    Load checkpoint, run beam search, use V-net to select which top-F1
    concepts to expand at each depth (hybrid guaranteed + exploratory beam).
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from concept_aggregators import (
    ConceptVNet as AggConceptVNet,
    get_instance_emb_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
#  V-Net architecture   (3-channel: concept, pos_mean, neg_mean)
#  *** MUST match BeamVNet in vocell.py ***
# ─────────────────────────────────────────────────────────────────────────────

class ConceptVNet(nn.Module):
    """
    Predicts best-reachable F1 from the embedding context of a concept.

    Input : (N, 3, embedding_dim)
              channel 0 → mean embedding of I(C)
              channel 1 → mean embedding of E+
              channel 2 → mean embedding of E-
    Output: (N,)  values in [0, 1]
    """

    def __init__(self, embedding_dim: int, device: str = 'cpu'):
        super().__init__()
        self.embedding_dim = embedding_dim
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_mean_emb(iris: List[str], df: pd.DataFrame) -> torch.Tensor:
    """Mean embedding of a list of entity IRIs.  Returns (1, 1, dim) float32."""
    dim   = df.shape[1]
    valid = [i for i in iris if i in df.index]
    if not valid:
        return torch.zeros(1, 1, dim, dtype=torch.float32)
    arr  = df.loc[valid].values.astype('float32')
    mean = arr.mean(axis=0)
    return torch.tensor(mean).view(1, 1, dim)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: Best-reachable F1 — bottom-up DP on recorded search tree
# ─────────────────────────────────────────────────────────────────────────────

def compute_best_reachable_from_nodes(nodes: List[Dict]) -> Dict[str, float]:
    """
    Given the list of recorded search-tree nodes from generate_vnet_dataset.py,
    compute  y_C = max F1 reachable from C  via bottom-up DP.

    Each node dict must have: concept_str, f1, depth, parent_str.
    The tree structure is reconstructed from parent_str edges.
    """
    # Build children map and initialise best = own F1
    children_map: Dict[str, List[str]] = defaultdict(list)
    best: Dict[str, float] = {}

    for node in nodes:
        cstr   = node['concept_str']
        best[cstr] = node['f1']
        parent = node.get('parent_str')
        if parent is not None:
            children_map[parent].append(cstr)

    # Propagate deepest-first so every child is resolved before its parent
    ordered = sorted(nodes, key=lambda n: n['depth'], reverse=True)
    for node in ordered:
        cstr = node['concept_str']
        for child_str in children_map.get(cstr, []):
            if child_str in best and best[child_str] > best[cstr]:
                best[cstr] = best[child_str]

    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4b: Build (X, y) training dataset from recorded nodes
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_from_nodes(
    nodes:          List[Dict],
    best_reachable: Dict[str, float],
    emb_pos:        torch.Tensor,
    emb_neg:        torch.Tensor,
    df:             pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For every recorded node build one training sample:
        x_C = [mean_emb(instance_iris_C),  emb_pos,  emb_neg]   shape (1, 3, dim)
        y_C = best_reachable_f1 from C  (bottom-up DP target)

    instance_iris is read directly from the node dict written by
    generate_vnet_dataset.py — no KB queries needed here.
    If a node has no instance_iris (old dataset format), we fall back to
    get_iris() which uses a shared, memoised KB.

    Returns X (N, 3, dim),  y (N,)
    """
    dim = df.shape[1]
    X_list: List[torch.Tensor] = []
    y_list: List[float]        = []

    for node in nodes:
        cstr = node['concept_str']
        if cstr not in best_reachable:
            continue

        # Prefer pre-computed iris stored during dataset generation.
        # This is consistent with inference and avoids KB round-trips.
        iris = node.get('instance_iris')
        if iris is None:                  # backwards-compat with old JSON format
            iris = get_iris(cstr) or []

        emb_c = get_mean_emb(iris, df)                            # (1, 1, dim)
        x     = torch.cat([emb_c, emb_pos, emb_neg], dim=1)      # (1, 3, dim)
        X_list.append(x)
        y_list.append(best_reachable[cstr])

    X = torch.cat(X_list, dim=0)                                  # (N, 3, dim)
    y = torch.tensor(y_list, dtype=torch.float32)                 # (N,)
    return X, y


def build_raw_dataset_from_nodes_sets(
    nodes:          List[Dict],
    best_reachable: Dict[str, float],
    pos_iris:       List[str],
    neg_iris:       List[str],
    df:             pd.DataFrame,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Companion to build_dataset_from_nodes for DeepSets / SetTransformer training.

    Returns raw embedding matrices instead of pre-computed means:
      concept_mats : List[Tensor(K_i, dim)]  — one matrix per recorded node
      pos_mat      : Tensor(|E+|, dim)
      neg_mat      : Tensor(|E-|, dim)
      y            : Tensor(N,)  — best-reachable F1 targets

    Each concept_mat has K_i rows (one per instance in I(C_i)); K_i may be 0
    for concepts with no instances in the embedding table.
    """
    concept_mats: List[torch.Tensor] = []
    y_list:       List[float]        = []

    for node in nodes:
        cstr = node['concept_str']
        if cstr not in best_reachable:
            continue
        iris = node.get('instance_iris') or []
        concept_mats.append(get_instance_emb_matrix(iris, df))   # (K_i, dim)
        y_list.append(best_reachable[cstr])

    pos_mat = get_instance_emb_matrix(pos_iris, df)               # (|E+|, dim)
    neg_mat = get_instance_emb_matrix(neg_iris, df)               # (|E-|, dim)
    y       = torch.tensor(y_list, dtype=torch.float32)           # (N,)
    return concept_mats, pos_mat, neg_mat, y


# ─────────────────────────────────────────────────────────────────────────────
#  Step 5: Train
# ─────────────────────────────────────────────────────────────────────────────

def train_v_net(
    X:          torch.Tensor,
    y:          torch.Tensor,
    epochs:     int   = 200,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    device:     str   = 'cpu',
    seed:       int   = 42,
) -> ConceptVNet:
    """
    Train ConceptVNet to predict y from X.

    seed fixes torch / random state so repeated runs on the same data
    produce the same checkpoint — essential for reproducible experiments.
    """
    torch.manual_seed(seed)
    import random as _random
    _random.seed(seed)
    _, _, embedding_dim = X.shape
    net = ConceptVNet(embedding_dim, device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=15)

    X = X.to(device)
    y = y.to(device)
    N = len(y)

    print(f"\nTraining ConceptVNet  |  samples={N}  epochs={epochs}  "
          f"batch={batch_size}  lr={lr}")
    print(f"{'Epoch':>6}  {'Loss':>10}  {'LR':>10}")
    print("-" * 32)

    net.train()
    for epoch in range(1, epochs + 1):
        perm       = torch.randperm(N, device=device)
        epoch_loss = 0.0
        steps      = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            Xb, yb = X[idx], y[idx]
            pred = net(Xb)
            loss = net.loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()
            steps      += 1

        avg = epoch_loss / max(steps, 1)
        sch.step(avg)
        current_lr = opt.param_groups[0]['lr']
        print(f"{epoch:>6}  {avg:>10.6f}  {current_lr:>10.2e}")

    net.eval()
    return net


def train_v_net_agg(
    agg_type:        str,
    lp_datasets_raw: Dict[str, Tuple],
    epochs:          int   = 200,
    lr:              float = 1e-3,
    device:          str   = 'cpu',
    seed:            int   = 42,
) -> AggConceptVNet:
    """
    Train ConceptVNet with a DeepSets or SetTransformer aggregator.

    lp_datasets_raw maps each training-LP name to
        (concept_mats, pos_mat, neg_mat, y)
    where concept_mats is a List[Tensor(K_i, dim)] of raw embedding matrices
    and pos_mat / neg_mat are (|E+|, dim) / (|E-|, dim) tensors.

    One gradient step is taken per LP per epoch (all concepts of that LP
    are forwarded together via AggConceptVNet.score_candidates).
    """
    torch.manual_seed(seed)
    import random as _random
    _random.seed(seed)

    # Determine embedding dim from first available positive-example matrix
    embedding_dim = None
    for _, pos_mat, _, _ in lp_datasets_raw.values():
        if pos_mat.shape[0] > 0:
            embedding_dim = pos_mat.shape[1]
            break
    if embedding_dim is None:
        raise ValueError("All LP positive-example matrices are empty — "
                         "cannot determine embedding_dim.")

    net = AggConceptVNet(embedding_dim, agg_type, device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=15)

    total_samples = sum(len(v[3]) for v in lp_datasets_raw.values())
    print(f"\nTraining ConceptVNet [{agg_type.upper()}]  |  "
          f"samples={total_samples}  epochs={epochs}  lr={lr}")
    print(f"{'Epoch':>6}  {'Loss':>10}  {'LR':>10}")
    print("-" * 32)

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0.0
        epoch_n    = 0

        for lp_name, (concept_mats, pos_mat, neg_mat, y) in lp_datasets_raw.items():
            if len(concept_mats) == 0:
                continue
            pos_d  = pos_mat.to(device)
            neg_d  = neg_mat.to(device)
            mats_d = [m.to(device) for m in concept_mats]
            y_d    = y.to(device)

            scores = net.score_candidates(mats_d, pos_d, neg_d)   # (N,)
            loss   = net.loss_fn(scores, y_d)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item() * len(concept_mats)
            epoch_n    += len(concept_mats)

        avg        = epoch_loss / max(epoch_n, 1)
        sch.step(avg)
        current_lr = opt.param_groups[0]['lr']
        print(f"{epoch:>6}  {avg:>10.6f}  {current_lr:>10.2e}")

    net.eval()
    return net


def build_bootstrap_datasets(
    dataset:        Dict,
    lp_names:       List[str],
    sample_lp_frac: float,
    sample_ex_frac: float,
    n_rounds:       int,
    df:             pd.DataFrame,
    agg_type:       str,
    seed:           int = 42,
):
    """
    Build training data by bootstrap sampling over LPs and their examples.

    For each of `n_rounds` rounds:
      - Sample max(1, floor(sample_lp_frac * |lp_names|)) LPs
      - For each sampled LP subsample floor(sample_ex_frac * |pos|) positives
        and floor(sample_ex_frac * |neg|) negatives
      - Build concept features using those subsampled pos/neg as E+/E- context

    The search-tree nodes (and their instance_iris / y targets) are taken
    from the full recorded search trees — only the example *context* changes.

    Returns
    -------
    'mean'                → (X, y)           tensors (all rounds concatenated)
    'deepsets' / 'st'     → lp_datasets_raw  dict keyed by "{lp}_r{round}"
    """
    rng              = random.Random(seed)
    n_lps_to_sample  = max(1, int(sample_lp_frac * len(lp_names)))

    if agg_type == 'mean':
        X_all, y_all = [], []

        for round_idx in range(n_rounds):
            sampled = rng.sample(lp_names, min(n_lps_to_sample, len(lp_names)))
            for lp_name in sampled:
                lp_data  = dataset[lp_name]
                nodes    = lp_data['nodes']
                pos_iris = lp_data['positive_examples']
                neg_iris = lp_data['negative_examples']

                n_pos   = max(1, int(sample_ex_frac * len(pos_iris)))
                n_neg   = max(1, int(sample_ex_frac * len(neg_iris)))
                sub_pos = rng.sample(pos_iris, min(n_pos, len(pos_iris)))
                sub_neg = rng.sample(neg_iris, min(n_neg, len(neg_iris)))

                emb_pos        = get_mean_emb(sub_pos, df)
                emb_neg        = get_mean_emb(sub_neg, df)
                best_reachable = compute_best_reachable_from_nodes(nodes)
                X_ep, y_ep     = build_dataset_from_nodes(
                    nodes, best_reachable, emb_pos, emb_neg, df
                )
                if X_ep.shape[0] == 0:
                    continue
                X_all.append(X_ep)
                y_all.append(y_ep)

        X = torch.cat(X_all, dim=0)
        y = torch.cat(y_all, dim=0)
        return X, y

    else:
        lp_datasets_raw: Dict[str, tuple] = {}

        for round_idx in range(n_rounds):
            sampled = rng.sample(lp_names, min(n_lps_to_sample, len(lp_names)))
            for lp_name in sampled:
                lp_data  = dataset[lp_name]
                nodes    = lp_data['nodes']
                pos_iris = lp_data['positive_examples']
                neg_iris = lp_data['negative_examples']

                n_pos   = max(1, int(sample_ex_frac * len(pos_iris)))
                n_neg   = max(1, int(sample_ex_frac * len(neg_iris)))
                sub_pos = rng.sample(pos_iris, min(n_pos, len(pos_iris)))
                sub_neg = rng.sample(neg_iris, min(n_neg, len(neg_iris)))

                best_reachable                    = compute_best_reachable_from_nodes(nodes)
                concept_mats, pos_mat, neg_mat, y = build_raw_dataset_from_nodes_sets(
                    nodes, best_reachable, sub_pos, sub_neg, df
                )
                if not concept_mats:
                    continue
                key = f"{lp_name}_r{round_idx}"
                lp_datasets_raw[key] = (concept_mats, pos_mat, neg_mat, y)

        return lp_datasets_raw


def train_bootstrap(
    dataset:        Dict,
    lp_names:       List[str],
    df:             pd.DataFrame,
    epochs:         int,
    device:         str,
    agg_type:       str   = 'mean',
    sample_lp_frac: float = 0.5,
    sample_ex_frac: float = 0.5,
    n_rounds:       int   = 5,
    seed:           int   = 42,
):
    """
    Bootstrap training strategy for large (or single) LP sets.

    Pools data from `n_rounds` bootstrap rounds (each with randomly
    subsampled LPs and example sets) and trains a single shared model.

    Returns (net, best_f1_in_training).
    """
    n   = len(lp_names)
    print(f"\nBootstrap training: {n} LP{'s' if n != 1 else ''}  |  "
          f"{n_rounds} rounds  |  "
          f"LP sample={sample_lp_frac*100:.0f}%  |  "
          f"ex sample={sample_ex_frac*100:.0f}%")

    data = build_bootstrap_datasets(
        dataset, lp_names, sample_lp_frac, sample_ex_frac,
        n_rounds, df, agg_type, seed,
    )

    if agg_type == 'mean':
        X, y   = data
        net    = train_v_net(X, y, epochs=epochs, device=device, seed=seed)
        best_f1 = float(y.max()) if y.numel() > 0 else 0.0
    else:
        lp_datasets_raw = data
        net    = train_v_net_agg(agg_type, lp_datasets_raw,
                                 epochs=epochs, device=device, seed=seed)
        best_f1 = max(
            (float(v[3].max()) for v in lp_datasets_raw.values() if v[3].numel() > 0),
            default=0.0,
        )

    return net, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# Module-level cache: shared KB + parser created once, results memoised.
# This ensures get_iris() returns the same sorted list every call and
# avoids loading the OWL file hundreds of times during dataset construction.
_IRIS_CACHE: Dict[str, List[str]] = {}
_SHARED: Dict[str, object] = {}


def get_iris(concept_str: str) -> List[str]:
    """Translate a DL concept string to a sorted list of instance IRIs.

    Uses one shared KnowledgeBase + DLSyntaxParser (lazy-init, memoised).
    Returns [] for concepts that cannot be parsed or have no instances.
    """
    if concept_str in _IRIS_CACHE:
        return _IRIS_CACHE[concept_str]

    if 'kb' not in _SHARED:
        from ontolearn.knowledge_base import KnowledgeBase
        from owlapy.parser import DLSyntaxParser
        kb  = KnowledgeBase(path="KGs/Family/family.owl")
        ns  = list(kb.ontology.classes_in_signature())[0].iri.get_namespace()
        _SHARED['kb']     = kb
        _SHARED['parser'] = DLSyntaxParser(namespace=ns)

    try:
        owl_expr = _SHARED['parser'].parse(concept_str)
        iris     = sorted(i.str for i in _SHARED['kb'].individuals(owl_expr))
    except Exception:
        iris = []

    _IRIS_CACHE[concept_str] = iris
    return iris


def train_for_lp(lp_name, dataset, df, EPOCHS, DEVICE, agg_type='mean'):
    training_lps = {n: d for n, d in dataset.items() if n != lp_name}

    if agg_type == 'mean':
        # ── existing mean-pooling path (unchanged) ──────────────────────────
        X_all, y_all = [], []

        for name, lp_data in training_lps.items():
            nodes    = lp_data['nodes']
            pos_iris = lp_data['positive_examples']
            neg_iris = lp_data['negative_examples']

            emb_pos = get_mean_emb(pos_iris, df)
            emb_neg = get_mean_emb(neg_iris, df)

            best_reachable = compute_best_reachable_from_nodes(nodes)

            X_ep, y_ep = build_dataset_from_nodes(
                nodes, best_reachable, emb_pos, emb_neg, df
            )

            X_all.append(X_ep)
            y_all.append(y_ep)

        X = torch.cat(X_all, dim=0)
        y = torch.cat(y_all, dim=0)

        net = train_v_net(X, y, epochs=EPOCHS, device=DEVICE)
        return net, list(training_lps.keys()), float(y.max())

    else:
        # ── DeepSets / SetTransformer path ─────────────────────────────────
        lp_datasets_raw: Dict[str, Tuple] = {}
        best_f1_all = 0.0

        for name, lp_data in training_lps.items():
            nodes    = lp_data['nodes']
            pos_iris = lp_data['positive_examples']
            neg_iris = lp_data['negative_examples']

            best_reachable = compute_best_reachable_from_nodes(nodes)
            concept_mats, pos_mat, neg_mat, y = build_raw_dataset_from_nodes_sets(
                nodes, best_reachable, pos_iris, neg_iris, df
            )
            if not concept_mats:
                continue

            lp_datasets_raw[name] = (concept_mats, pos_mat, neg_mat, y)
            if y.numel() > 0:
                best_f1_all = max(best_f1_all, float(y.max()))

        net = train_v_net_agg(agg_type, lp_datasets_raw, epochs=EPOCHS, device=DEVICE)
        return net, list(training_lps.keys()), best_f1_all


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline V-Net training for VOCELL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--lps_file',    default='LPs/Family/lps_difficult.json',
                        help='Path to the LP JSON file.')
    parser.add_argument('--dataset_file',default='vnet_search_data_difficult.json',
                        help='Path to the search-tree dataset JSON.')
    parser.add_argument('--embeddings',  default='Experiments/embeddings/Keci_entity_embeddings.csv',
                        help='Path to entity embeddings CSV.')
    parser.add_argument('--epochs',      type=int, default=200,
                        help='Training epochs.')
    parser.add_argument('--device',      default='cpu',
                        help='Torch device (cpu | cuda | cuda:0 …).')
    parser.add_argument('--agg_type',    default='mean',
                        choices=['mean', 'deepsets', 'settransformer'],
                        help='Concept-set aggregation strategy.')
    parser.add_argument('--strategy',    default='auto',
                        choices=['auto', 'loocv', 'bootstrap'],
                        help=('Training strategy. "auto" picks loocv when '
                              '1 < N_LPS <= --loocv_threshold, else bootstrap.'))
    parser.add_argument('--loocv_threshold', type=int, default=10,
                        help='Max number of LPs for automatic loocv selection.')
    parser.add_argument('--sample_lp_frac',  type=float, default=0.5,
                        help='[bootstrap] Fraction of LPs sampled per round.')
    parser.add_argument('--sample_ex_frac',  type=float, default=0.5,
                        help='[bootstrap] Fraction of pos/neg examples sampled per LP per round.')
    parser.add_argument('--n_rounds',        type=int,   default=5,
                        help='[bootstrap] Number of bootstrap rounds.')
    parser.add_argument('--output_dir',  default=None,
                        help='Directory for saved checkpoints. '
                             'Defaults to Family_{agg_type}.')
    args = parser.parse_args()

    # ── Resolve config from parsed args ──────────────────────
    LPS_FILE           = args.lps_file
    DATASET_FILE       = args.dataset_file
    EMBEDDINGS         = args.embeddings
    EPOCHS             = args.epochs
    DEVICE             = args.device
    AGG_TYPE           = args.agg_type
    LOOCV_THRESHOLD    = args.loocv_threshold
    SAMPLE_LP_FRAC     = args.sample_lp_frac
    SAMPLE_EX_FRAC     = args.sample_ex_frac
    N_BOOTSTRAP_ROUNDS = args.n_rounds
    output_dir         = args.output_dir or f'Family_{AGG_TYPE}'

    print("=" * 60)
    print("VOCELL V-Net training")
    print(f"  agg_type   : {AGG_TYPE}")
    print(f"  strategy   : {args.strategy}")
    print(f"  epochs     : {EPOCHS}")
    print(f"  device     : {DEVICE}")
    print(f"  lps_file   : {LPS_FILE}")
    print(f"  dataset    : {DATASET_FILE}")
    print(f"  embeddings : {EMBEDDINGS}")
    print(f"  output_dir : {output_dir}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    # ── Load LPs ────────────────────────────────────────────
    with open(LPS_FILE) as f:
        _raw = json.load(f)
    # Support both {"problems": {...}} and flat {"LP_name": {...}} formats.
    lps_data = _raw.get('problems', _raw) if isinstance(_raw, dict) else {}

    all_lp_names = list(lps_data.keys())

    # ── Load dataset (search trees) ─────────────────────────
    with open(DATASET_FILE) as f:
        dataset = json.load(f)

    # Only train on LPs that are present in the dataset (generate_vnet_dataset.py
    # may have been run with --lps / --num_lps, so the dataset can be a subset).
    available_lp_names = [n for n in all_lp_names if n in dataset]
    missing = len(all_lp_names) - len(available_lp_names)
    if missing:
        print(f"  ⚠ {missing} LP(s) from the LP file are not in the dataset "
              f"and will be skipped. Run generate_vnet_dataset.py to add them.")
    all_lp_names = available_lp_names
    print(f"  LPs available for training: {len(all_lp_names)}")

    # ── Load embeddings ─────────────────────────────────────
    df = pd.read_csv(EMBEDDINGS, index_col=0).astype('float32')

    # ── Choose and run training strategy ────────────────────────────
    if args.strategy == 'loocv':
        use_loocv = True
    elif args.strategy == 'bootstrap':
        use_loocv = False
    else:  # 'auto'
        use_loocv = 1 < len(all_lp_names) <= LOOCV_THRESHOLD

    if use_loocv:
        # ── Leave-One-Out: one checkpoint per LP ─────────────
        print(f"Strategy: Leave-One-Out  ({len(all_lp_names)} LPs)")

        for lp_name in all_lp_names:
            print("\n" + "=" * 60)
            print(f"Held-out LP: {lp_name}")
            print("=" * 60)

            net, training_lps, best_f1 = train_for_lp(
                lp_name, dataset, df, EPOCHS, DEVICE, agg_type=AGG_TYPE
            )
            save_path = os.path.join(output_dir, f'vocell_v_net_{lp_name}_{AGG_TYPE}.pt')
            torch.save({
                'model_state_dict':    net.state_dict(),
                'embedding_dim':       df.shape[1],
                'agg_type':            AGG_TYPE,
                'training_strategy':   'loocv',
                'lp_name':             lp_name,
                'training_lps':        training_lps,
                'best_f1_in_training': best_f1,
            }, save_path)
            print(f"Saved → {save_path}")

    else:
        # ── Bootstrap: one shared checkpoint ─────────────────
        n = len(all_lp_names)
        print(f"Strategy: Bootstrap  ({n} LP{'s' if n != 1 else ''})")

        net, best_f1 = train_bootstrap(
            dataset, all_lp_names, df, EPOCHS, DEVICE,
            agg_type       = AGG_TYPE,
            sample_lp_frac = SAMPLE_LP_FRAC,
            sample_ex_frac = SAMPLE_EX_FRAC,
            n_rounds       = N_BOOTSTRAP_ROUNDS,
        )
        save_path = os.path.join(output_dir, f'vocell_v_net_bootstrap_{AGG_TYPE}.pt')
        torch.save({
            'model_state_dict':    net.state_dict(),
            'embedding_dim':       df.shape[1],
            'agg_type':            AGG_TYPE,
            'training_strategy':   'bootstrap',
            'all_lp_names':        all_lp_names,
            'sample_lp_frac':      SAMPLE_LP_FRAC,
            'sample_ex_frac':      SAMPLE_EX_FRAC,
            'n_rounds':            N_BOOTSTRAP_ROUNDS,
            'best_f1_in_training': best_f1,
        }, save_path)
        print(f"Saved → {save_path}")


if __name__ == '__main__':
    main()
# Family: python train_vocell_v_net.py --strategy loocv
# Carcinogenesis: python train_vocell_v_net.py --lps_file LPs/Carcinogenesis/lps.json --dataset_file vnet_search_data_carcinogenesis.json --strategy bootstrap --output_dir Carcinogenesis_mean --epochs 200 --embeddings ../Ontolearn_ISWC/datasets/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv
# Mutagenesis:  python train_vocell_v_net.py --lps_file LPs/Mutagenesis/lps.json --dataset_file vnet_search_data_mutagenesis.json --strategy bootstrap --output_dir Mutagegenesis_mean --epochs 200 --embeddings ../Ontolearn_ISWC/datasets/mutagenesis/embeddings/DeCaL_entity_embeddings.csv
# Animal: python train_vocell_v_net.py   --lps_file ../Ontolearn_ISWC/datasets/animals/training_data/training_data_prep.json   --dataset_file /tmp/vnet_animals_test.json   --strategy bootstrap --epochs 50 --output_dir animals_mean --embeddings ../Ontolearn_ISWC/datasets/animals/embeddings/DeCaL_entity_embeddings.csv