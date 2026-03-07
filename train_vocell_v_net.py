"""
train_vocell_v_net.py  —  Offline V-Net training for VOCELL
============================================================

Algorithm (mirrors the description exactly):

  1. Start from ⊤ for a given LP = (E+, E-).
  2. BFS-explore the refinement space up to a fixed depth / budget using
     PruneCEL as the refinement operator.
  3. For every generated concept C:
       • compute I(C) via the KB reasoner
       • evaluate F1(C) w.r.t. E+ and E-
       • compute x_C = [mean_emb(I(C)),  emb_pos,  emb_neg]   shape (3, dim)
  4. After exploration, propagate best-reachable F1 bottom-up through the
     exploration tree  →  y_C = max F1 reachable from C.
  5. Train V(C) ≈ y_C  (MSE loss, mini-batch SGD).
  6. Save weights to  vocell_v_net_<LP>.pt

The saved checkpoint is loaded by VOCELL at inference time (beam ranking).
No change to vocell.py's search logic is needed.
"""

import json
import os
import time
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import pandas as pd
import torch
import torch.nn as nn

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import PruneCELBasedRefinement
from owlapy.class_expression import OWLClassExpression, OWLThing
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer


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


def evaluate_f1(concept: OWLClassExpression,
                pos: Set[OWLNamedIndividual],
                neg: Set[OWLNamedIndividual],
                kb: KnowledgeBase) -> float:
    instances = kb.individuals_set(concept)
    tp = len(instances & pos)
    fp = len(instances & neg)
    fn = len(pos - instances)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1-3: BFS Exploration
# ─────────────────────────────────────────────────────────────────────────────

def explore_lp(
    kb: KnowledgeBase,
    operator: PruneCELBasedRefinement,
    pos: Set[OWLNamedIndividual],
    neg: Set[OWLNamedIndividual],
    df: pd.DataFrame,
    max_concepts: int = 2000,
    max_depth:    int = 6,
) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor]:
    """
    BFS from ⊤.

    Returns
    -------
    records      : concept_str → {'f1', 'emb' (1,1,dim), 'depth'}
    children_map : concept_str → [child_str, ...]   (tree edges)
    emb_pos      : (1, 1, dim)  mean embedding of E+
    emb_neg      : (1, 1, dim)  mean embedding of E-
    """

    def concept_mean_emb(c: OWLClassExpression) -> torch.Tensor:
        iris = [ind.str for ind in kb.individuals_set(c)]

        return get_mean_emb(iris, df)
    
    
    pos_iris = [str(iri) for iri in pos]
    neg_iris = [str(iri) for iri in neg]
    emb_pos  = get_mean_emb(pos_iris, df)
    emb_neg  = get_mean_emb(neg_iris, df)

    records:      Dict = {}
    children_map: Dict = defaultdict(list)
    visited:      Set  = set()

    # ── root ──────────────────────────────────────────────────────────────────
    root_str = str(OWLThing)
    visited.add(root_str)
    records[root_str] = {
        'f1':   evaluate_f1(OWLThing, pos, neg, kb),
        'emb':  concept_mean_emb(OWLThing),
        'depth': 0,
    }

    queue = deque([(OWLThing, 0)])
    total = 0
    t0    = time.time()

    while queue and total < max_concepts:
        concept, depth = queue.popleft()
        cstr = str(concept)

        if depth >= max_depth:
            continue

        try:
            for child in operator.refine(concept):
                child_str = str(child)
                # Always record the edge (needed for DP even if already visited)
                children_map[cstr].append(child_str)

                if child_str in visited:
                    continue
                visited.add(child_str)

                f1  = evaluate_f1(child, pos, neg, kb)
                emb = concept_mean_emb(child)
                records[child_str] = {'f1': f1, 'emb': emb, 'depth': depth + 1}
                queue.append((child, depth + 1))
                total += 1

                if total % 200 == 0:
                    print(f"    {total}/{max_concepts} concepts  "
                          f"(best F1 so far: "
                          f"{max(r['f1'] for r in records.values()):.3f})")
                if total >= max_concepts:
                    break
        except Exception:
            continue

    elapsed = time.time() - t0
    print(f"  Explored {len(records)} concepts in {elapsed:.1f}s")
    return records, dict(children_map), emb_pos, emb_neg


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: Best-reachable F1 — bottom-up DP on the exploration tree
# ─────────────────────────────────────────────────────────────────────────────

def compute_best_reachable(records: Dict, children_map: Dict) -> Dict[str, float]:
    """
    y_C = max F1 reachable from C (i.e. max over C and all its descendants).

    Process nodes deepest-first so that when we reach a parent all its
    children already have their final value.
    """
    best = {cstr: info['f1'] for cstr, info in records.items()}

    # deepest first
    ordered = sorted(records, key=lambda s: records[s]['depth'], reverse=True)
    for cstr in ordered:
        for child_str in children_map.get(cstr, []):
            if child_str in best:
                if best[child_str] > best[cstr]:
                    best[cstr] = best[child_str]

    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4b: Build (X, y) training dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    records:       Dict,
    best_reachable: Dict[str, float],
    emb_pos:       torch.Tensor,
    emb_neg:       torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x_C = [mean_emb(I(C)),  emb_pos,  emb_neg]   shape (3, dim)
    y_C = best-reachable F1 from C

    Returns X (N, 3, dim),  y (N,)
    """
    X_list, y_list = [], []
    for cstr, info in records.items():
        emb = info['emb']                                        # (1, 1, dim)
        x   = torch.cat([emb, emb_pos, emb_neg], dim=1)         # (1, 3, dim)
        X_list.append(x)
        y_list.append(best_reachable[cstr])

    X = torch.cat(X_list, dim=0)                                 # (N, 3, dim)
    y = torch.tensor(y_list, dtype=torch.float32)                # (N,)
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
) -> ConceptVNet:
    """
    Train ConceptVNet to predict y from X.
    Prints loss every epoch — you should see it decrease.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    LP_NAME      = 'Cousin'
    EMBEDDINGS   = 'Experiments/embeddings/Keci_entity_embeddings.csv'
    SAVE_PATH    = f'vocell_v_net_{LP_NAME}.pt'
    MAX_CONCEPTS = 1000  # exploration budget
    MAX_DEPTH    = 15      # maximum BFS depth
    EPOCHS       = 200    # training epochs
    DEVICE       = 'cpu'

    # ── Load KB & LP ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"V-Net training  —  LP: {LP_NAME}")
    print("=" * 60)

    print("\n[1/5] Loading KB and LP...")
    kb = KnowledgeBase(path='KGs/Family/family.owl')

    with open('LPs/Family/lps.json') as f:
        lps = json.load(f)
    p   = lps['problems'][LP_NAME]
    pos = frozenset({OWLNamedIndividual(IRI.create(i)) for i in p['positive_examples']})
    neg = frozenset({OWLNamedIndividual(IRI.create(i)) for i in p['negative_examples']})
    print(f"     {LP_NAME}: pos={len(pos)}, neg={len(neg)}")

    # ── Load embeddings ───────────────────────────────────────────────────────
    print("\n[2/5] Loading embeddings...")
    df = pd.read_csv(EMBEDDINGS, index_col=0).astype('float32')
    print(f"     {df.shape[0]} entities × {df.shape[1]} dims")

    # ── Build operator ────────────────────────────────────────────────────────
    print("\n[3/5] Initialising refinement operator...")
    operator = PruneCELBasedRefinement(
        knowledge_base=kb,
        sparql_endpoint='http://localhost:3030/family/sparql',
    )
    operator.precision_threshold = 1.0
    operator.recall_threshold    = 0.6
    operator.set_input_examples(pos, neg)

    # ── BFS Exploration ───────────────────────────────────────────────────────
    print(f"\n[4/5] Exploring from ⊤  "
          f"(budget={MAX_CONCEPTS} concepts, max_depth={MAX_DEPTH})...")
    records, children_map, emb_pos, emb_neg = explore_lp(
        kb, operator, set(pos), set(neg), df,
        max_concepts=MAX_CONCEPTS,
        max_depth=MAX_DEPTH,
    )

    # ── Best-reachable F1 ─────────────────────────────────────────────────────
    best_reachable = compute_best_reachable(records, children_map)
    best_overall   = max(best_reachable.values())
    mean_target    = sum(best_reachable.values()) / len(best_reachable)
    print(f"  Target y  —  min={min(best_reachable.values()):.3f}  "
          f"mean={mean_target:.3f}  max={best_overall:.3f}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    X, y = build_dataset(records, best_reachable, emb_pos, emb_neg)
    print(f"  Dataset: X={tuple(X.shape)},  y={tuple(y.shape)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Training...")
    net = train_v_net(X, y, epochs=EPOCHS, device=DEVICE)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        'model_state_dict': net.state_dict(),
        'embedding_dim':    df.shape[1],
        'lp_name':          LP_NAME,
        'max_concepts':     MAX_CONCEPTS,
        'max_depth':        MAX_DEPTH,
        'best_f1_found':    best_overall,
    }, SAVE_PATH)
    print(f"\nSaved → {SAVE_PATH}  (best F1 seen during exploration: {best_overall:.4f})")
    print("=" * 60)
    print("Done. Load this model in VOCELL with:")
    print(f"  learner = VOCELL(..., path_embeddings='{EMBEDDINGS}', "
          f"v_net_path='{SAVE_PATH}')")
    print("=" * 60)


if __name__ == '__main__':
    main()
