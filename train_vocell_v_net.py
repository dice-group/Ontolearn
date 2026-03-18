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

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn


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

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    # Target LP: held out from training, used only for evaluation.
    LP_NAME      = 'Aunt'   

    # Pre-computed search data produced by generate_vnet_dataset.py.
    DATASET_FILE = 'vnet_search_data_difficult.json'

    EMBEDDINGS   = 'Experiments/embeddings/Keci_entity_embeddings.csv'
    SAVE_PATH    = f'vocell_v_net_{LP_NAME}.pt'
    EPOCHS       = 200
    DEVICE       = 'cpu'

    print("=" * 60)
    print(f"V-Net training  —  Leave-one-LP-out  |  Target: {LP_NAME}")
    print(f"  Dataset : {DATASET_FILE}")
    print("=" * 60)

    # ── Load pre-computed search dataset ──────────────────────────────────────
    print(f"\n[1/4] Loading dataset from '{DATASET_FILE}'...")
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"{DATASET_FILE} not found.\n"
            "Run  python generate_vnet_dataset.py  first."
        )
    with open(DATASET_FILE) as f:
        dataset: Dict[str, Dict] = json.load(f)

    all_lp_names = list(dataset.keys())
    training_lps = {n: d for n, d in dataset.items() if n != LP_NAME}

    if not training_lps:
        raise RuntimeError(
            f"No training LPs left after excluding '{LP_NAME}'. "
            f"Available LPs: {all_lp_names}"
        )

    if LP_NAME not in dataset:
        print(f"  WARNING: '{LP_NAME}' not found in dataset — "
              f"evaluation data missing, but training will proceed.")

    print(f"  Available LPs  : {all_lp_names}")
    print(f"  Training on    : {list(training_lps.keys())}")
    print(f"  Held-out (eval): {LP_NAME}")

    # ── Load embeddings ───────────────────────────────────────────────────────
    print("\n[2/4] Loading embeddings...")
    df = pd.read_csv(EMBEDDINGS, index_col=0).astype('float32')
    print(f"     {df.shape[0]} entities × {df.shape[1]} dims")

    # ── Build training tensors (one LP at a time, then cat) ───────────────────
    print("\n[3/4] Building training dataset (leave-one-LP-out)...")
    X_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for lp_name, lp_data in training_lps.items():
        nodes    = lp_data['nodes']
        pos_iris = lp_data['positive_examples']
        neg_iris = lp_data['negative_examples']

        # Embedding context for this LP's examples
        emb_pos = get_mean_emb(pos_iris, df)   # (1, 1, dim)
        emb_neg = get_mean_emb(neg_iris, df)   # (1, 1, dim)

        # Bottom-up DP on the recorded search tree
        best_reachable = compute_best_reachable_from_nodes(nodes)

        # Build (X, y) from stored instance_iris — no KB queries needed
        X_ep, y_ep = build_dataset_from_nodes(
            nodes, best_reachable, emb_pos, emb_neg, df
        )
        X_all.append(X_ep)
        y_all.append(y_ep)

        root_val = best_reachable.get(nodes[0]['concept_str'], 0.0)
        print(f"  {lp_name:25s}: {len(y_ep):4d} samples  "
              f"best_reachable={max(best_reachable.values()):.3f}  "
              f"root_value={root_val:.3f}")

    X = torch.cat(X_all, dim=0)   # (N_total, 3, dim)
    y = torch.cat(y_all, dim=0)   # (N_total,)
    print(f"\n  Aggregated: X={tuple(X.shape)},  y={tuple(y.shape)}  "
          f"y_mean={float(y.mean()):.3f}  y_max={float(y.max()):.3f}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[4/4] Training...")
    net = train_v_net(X, y, epochs=EPOCHS, device=DEVICE)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        'model_state_dict':    net.state_dict(),
        'embedding_dim':       df.shape[1],
        'lp_name':             LP_NAME,
        'training_lps':        list(training_lps.keys()),
        'dataset_file':        DATASET_FILE,
        'best_f1_in_training': float(y.max()),
    }, SAVE_PATH)
    print(f"\nSaved → {SAVE_PATH}")
    print(f"  Trained on : {list(training_lps.keys())}")
    print(f"  Held out   : {LP_NAME}")
    print("=" * 60)
    print("Done. Load this model in VOCELL with:")
    print(f"  learner = VOCELL(..., path_embeddings='{EMBEDDINGS}', "
          f"v_net_path='{SAVE_PATH}')")
    print("=" * 60)


if __name__ == '__main__':
    main()
