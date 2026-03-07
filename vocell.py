"""
VOCELL: Oracle Based V-learning for Class Expression Learning

Combines best-first search with V-Learning intelligent termination.

Search (PruneCEL-S — Zhang et al., K-CAP 2025):
  - Priority queue ordered by refinementScore = F1 - 0.01 * length
  - Skip rule: D ∈ ρ(C) added to queue iff score(D) > score(C) OR D adds a role
  - SPARQL-based scoring with DL reasoner fallback
  - Complex negation via NNF push-down

RL Termination (V-Learning):
  - V-network predicts expected improvement from continued search
  - Learns from search trajectories to stop early when no improvement expected
  - Gets smarter across runs on the same LP

When use_termination=False, produces identical results to test_refinement_fix.py.
"""

import hashlib
import heapq
import os
import time
from typing import FrozenSet, Tuple, Dict, List, Set, Optional

from owlapy import owl_expression_to_dl

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import PruneCELBasedRefinement
from owlapy.class_expression import (
    OWLObjectSomeValuesFrom, OWLThing, OWLClass, OWLObjectUnionOf,
    OWLObjectComplementOf, OWLObjectIntersectionOf,
    OWLObjectAllValuesFrom, OWLNothing, OWLClassExpression,
)
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
import json
import time
from typing import Set, Optional, List

from ontolearn.knowledge_base import KnowledgeBase
# from refinement_operators import PruneCELBasedRefinement, ModifiedCELOERefinement, LengthBasedRefinement
from owlapy.class_expression import OWLClassExpression, OWLThing
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.utils import simplify_class_expression, get_expression_length
from rl_termination_module import IntelligentTerminationAgent
import torch
import torch.nn as nn
import random
from collections import deque


class BeamVNet(nn.Module):
    """V-network for VOCELL beam scoring.

    Input : (N, 3, embedding_dim) — channels [concept_mean, pos_mean, neg_mean]
    Output: (N,) predicted quality scores in [0, 1]

    *** Architecture must match ConceptVNet in train_vocell_v_net.py ***
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
        self.loss = nn.MSELoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, 3, embedding_dim) → (N,)"""
        return self.net(X).flatten()


class _ReplayBuffer:
    """Replay buffer storing (concept_emb, reward) 2-tuples."""

    def __init__(self, maxlen: int = 50000):
        self._storage: deque = deque(maxlen=maxlen)

    def append(self, item):
        self._storage.append(item)

    def retrieve(self, batch_size: int = 256):
        data = list(self._storage)
        if not data:
            return [], []
        k     = min(batch_size, len(data))
        batch = random.sample(data, k)
        return [x[0] for x in batch], [x[1] for x in batch]

    def __len__(self):
        return len(self._storage)


class VOCELL:
    """
    Simple beam search learner that relies on the refinement operator
    to collect high-precision fragments automatically.
    """
    
    def __init__(self, kb: KnowledgeBase, operator: PruneCELBasedRefinement,
                 beam_width: int = 10,
                 max_depth: int = 5,
                 precision_threshold: float = 1.0,
                 recall_threshold: float = 0.5,
                 time_limit: float = 300.0,
                 use_v_learning: bool = False,
                 v_epsilon: float = 0.3,
                 v_memory_path: str = 'vocell_termination_memory.pkl',
                 df_embeddings=None,
                 path_embeddings: str = None,
                 v_net_path: str = None,
                 device: str = 'cpu',
                 num_epochs_per_replay: int = 3,
                 verbose: bool = False):
        """
        Initialize the learner.
        
        Args:
            kb: Knowledge base
            operator: Refinement operator (collects fragments automatically)
            beam_width: Number of top concepts to refine each iteration (default: 10)
            max_depth: Maximum refinement depth (default: 5)
            time_limit: Time limit in seconds (default: 300)
        """
        self.kb = kb
        self.operator = operator
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold
        self.verbose = verbose
        self.renderer = DLSyntaxObjectRenderer()
        
        # Statistics
        self.total_refinements_explored = 0

        # V-learning termination agent
        self.use_v_learning = use_v_learning
        if use_v_learning:
            self.termination_agent = IntelligentTerminationAgent(
                learning_rate=0.001,
                gamma=0.999,
                epsilon=v_epsilon,
                min_quality_threshold=0.5,
                min_concepts_explored=10,
                max_concepts_explored=50000,
                memory_path=v_memory_path
            )
            print(f"V-Learning enabled (\u03b5={v_epsilon:.2f}, memory={v_memory_path})")
        else:
            self.termination_agent = None

        # ── Embedding-based V-net for beam ranking (mirrors DrillV_Complex) ──
        self.device = device
        self.num_epochs_per_replay = num_epochs_per_replay
        self.df_embeddings = df_embeddings
        self.embedding_dim = 1
        self.v_net: Optional[BeamVNet] = None
        self.v_optimizer = None
        self.v_net_trained = False
        self.v_experiences: Optional[_ReplayBuffer] = None
        self.emb_pos = None
        self.emb_neg = None
        self._episode_pairs   = []
        self._episode_rewards = []

        if path_embeddings and os.path.isfile(path_embeddings):
            import pandas as pd
            self.df_embeddings = pd.read_csv(path_embeddings, index_col=0).astype('float32')
            print(f"Loaded embeddings: {self.df_embeddings.shape}")

        if self.df_embeddings is not None:
            _, self.embedding_dim = self.df_embeddings.shape

        self.v_net_path = v_net_path

        if use_v_learning and self.df_embeddings is not None:
            self.v_net = BeamVNet(self.embedding_dim, device)
            self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=0.001)
            self.v_experiences = _ReplayBuffer(maxlen=50000)
            print(f"V-Net (BeamVNet) enabled: embedding_dim={self.embedding_dim}")
            if v_net_path and os.path.isfile(v_net_path):
                self.load(v_net_path)
        elif use_v_learning and self.df_embeddings is None:
            print("use_v_learning=True but no embeddings provided "
                  "\u2014 V-Net disabled, using IntelligentTerminationAgent only")
    
    def get_instances(self, concept: OWLClassExpression) -> Set[OWLNamedIndividual]:
        """Get all instances of a concept."""
        return self.kb.individuals_set(concept)
    
    def evaluate(self, concept: OWLClassExpression, pos: Set, neg: Set):
        """
        Evaluate a concept on given examples.
        
        Returns:
            (f1, precision, recall, covered_positives)
        """
        instances = self.get_instances(concept)
        
        covered_pos = instances & pos
        covered_neg = instances & neg
        
        tp = len(covered_pos)
        fp = len(covered_neg)
        fn = len(pos - covered_pos)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall, covered_pos

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _get_embedding_from_ind_set(self, individuals) -> torch.Tensor:
        """Mean embedding from a collection of OWLNamedIndividual. Returns (1, 1, dim) tensor."""
        if self.df_embeddings is None:
            return torch.zeros(1, 1, self.embedding_dim)
        # try:
        iris = [ind.str for ind in individuals]
        valid = [i for i in iris if i in self.df_embeddings.index]
        if not valid:
            return torch.zeros(1, 1, self.embedding_dim)
        vals = self.df_embeddings.loc[valid].values
        emb = torch.from_numpy(vals.mean(axis=0)).float()
        return emb.view(1, 1, self.embedding_dim)
        # except Exception:
        #     return torch.zeros(1, 1, self.embedding_dim)

    def get_concept_embedding(self, concept: OWLClassExpression) -> Optional[torch.Tensor]:
        """Mean embedding of concept instances. Returns (1, 1, dim) tensor, or None if no embeddings."""
        if self.df_embeddings is None:
            return None
        return self._get_embedding_from_ind_set(self.get_instances(concept))

    # ── V-learning (mirrors DrillV_Complex) ───────────────────────────────────

    def form_experiences(self, concept_embs: List, rewards: List) -> None:
        """Store (concept_emb, best_reachable_reward) in replay buffer.
        y_C = max future reward reachable from C  (same target as offline training)."""
        if self.v_experiences is None:
            return
        for th, emb in enumerate(concept_embs):
            self.v_experiences.append((emb, max(rewards[th:])))

    def learn_from_replay_memory(self) -> None:
        """Train V-net: x = (concept_emb, pos_mean, neg_mean),  y = best_reachable_f1.
        Same input shape as ConceptVNet in train_vocell_v_net.py."""
        if self.v_net is None or self.v_experiences is None:
            return
        emb_batch, y = self.v_experiences.retrieve()
        if not emb_batch:
            return
        emb_batch    = torch.cat(emb_batch, dim=0)          # (N, 1, dim)
        reward_batch = torch.tensor(y, dtype=torch.float32) # (N,)
        N  = len(reward_batch)
        ep = self.emb_pos if self.emb_pos is not None else torch.zeros(1, 1, self.embedding_dim)
        en = self.emb_neg if self.emb_neg is not None else torch.zeros(1, 1, self.embedding_dim)

        X = torch.cat([
            emb_batch,
            ep.repeat(N, 1, 1),
            en.repeat(N, 1, 1),
        ], dim=1)  # (N, 3, dim)

        self.v_net.train()
        total_loss = 0.0
        for _ in range(self.num_epochs_per_replay):
            pred = self.v_net(X)
            loss = self.v_net.loss(pred, reward_batch)
            self.v_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 5.0)
            self.v_optimizer.step()
            total_loss += loss.item()
        self.v_net.eval()
        self.v_net_trained = True
        avg_loss = total_loss / self.num_epochs_per_replay
        print(f"  V-Net: {N} experiences, avg loss={avg_loss:.4f}")


    def load(self, path: str = 'vocell_v_net.pt') -> None:
        """Load V-net weights from disk."""
        if self.v_net is None:
            print("V-net not initialized (no embeddings provided).")
            return
        ckpt = torch.load(path, weights_only=True)
        self.v_net.load_state_dict(ckpt['model_state_dict'])
        self.v_net_trained = ckpt.get('v_net_trained', True)
        self.v_net.eval()
        print(f"V-Net loaded \u2190 {path}  (trained={self.v_net_trained})")

    # ─────────────────────────────────────────────────────────────────────────

    def beam_search(self, pos: Set, neg: Set, start, training: bool = False) -> Optional[OWLClassExpression]:
        """
        Beam search for best concept.

        The refinement operator automatically collects high-precision fragments
        during the search.

        Args:
            training: If True, disable V-net and termination-agent early stopping
                      so that the search explores fully and collects experiences.

        Returns:
            Best F1 concept found
        """

        # Initialize beam with ⊤
        beam = [OWLThing]

        best_concept = None
        best_f1 = 0.0

        visited = set()
        visited.add(str(OWLThing))

        # Reset termination agent for this episode
        if self.termination_agent is not None:
            self.termination_agent.reset_for_new_episode()

        # Pre-compute pos/neg embeddings (reused across all depth levels)
        if self.v_net is not None:
            self.emb_pos = self._get_embedding_from_ind_set(pos)
            self.emb_neg = self._get_embedding_from_ind_set(neg)

        # Episode buffers (consumed by form_experiences in fit())
        self._episode_pairs   = []
        self._episode_rewards = []

        v_stop = False  # flag to exit nested loops when agent says stop

        if self.verbose:
            mode_tag = ""
            if self.v_net is not None:
                mode_tag += " [BeamVNet ON]"
            if self.termination_agent is not None:
                mode_tag += " [TermAgent ON]"
            print(f"\nStarting beam search (beam width={self.beam_width}, max depth={self.max_depth}){mode_tag}")
            print("="*80)

        for depth in range(self.max_depth):

            if v_stop:
                break

            if (time.time() - start) >= self.time_limit:
                elapsed = time.time() - start
                if self.verbose:
                    print(f"\nTime reached in {elapsed:.1f}s")
                    print(f"Total refinements explored: {self.total_refinements_explored}")
                break

            if self.verbose:
                print(f"\nDepth {depth}: Refining top {len(beam)} concepts...")

            # Generate all refinements of current beam
            all_refinements = []
            all_child_embs  = []  # child_emb parallel to all_refinements

            for concept in beam:
                try:

                    # Generate refinements
                    candidates = []
                    candidate_embs = []

                    for child in self.operator.refine(concept):

                        child_str = str(child)
                        if child_str in visited:
                            continue
                        visited.add(child_str)

                        child_emb = None

                        if self.v_net is not None and self.v_net_trained:
                            try:
                                child_emb = self.get_concept_embedding(child)
                            except Exception:
                                child_emb = None

                        candidates.append((child, child_emb))
                        if child_emb is not None:
                            candidate_embs.append(child_emb)

                    # ---------------------------------------------------------
                    # V-NET FILTER (before evaluation)
                    # ---------------------------------------------------------
                    if (
                        self.v_net is not None
                        and self.v_net_trained
                        and len(candidate_embs) == len(candidates)
                        and self.emb_pos is not None
                    ):

                        N = len(candidate_embs)

                        child_batch = torch.cat(candidate_embs, dim=0)

                        X = torch.cat([
                            child_batch,
                            self.emb_pos.repeat(N, 1, 1),
                            self.emb_neg.repeat(N, 1, 1),
                        ], dim=1)

                        self.v_net.eval()
                        with torch.no_grad():
                            v_scores = self.v_net(X).tolist()

                        DEAD_END_THRESHOLD = 0.7  # V-net score below this = dead end
                        F1_SAFETY_FLOOR    = 0.8  # never prune a concept already this good

                        # Quick F1 approximation per candidate (no extra KB call —
                        # get_concept_embedding already called individuals_set internally,
                        # but here we use a rough heuristic: keep if v_score high enough
                        # OR if this candidate is already scoring well currently).
                        # Since we have not evaluated F1 yet at this point, we use only
                        # the V-net score combined with the safety floor on CURRENT best_f1
                        # to avoid ever pruning when we already have a good lead.
                        filtered = [
                            (c, emb)
                            for (c, emb), v in zip(candidates, v_scores)
                            if v >= DEAD_END_THRESHOLD and best_f1 >= F1_SAFETY_FLOOR
                        ]

                        # Only apply if enough survivors remain to fill the beam
                        if len(filtered) >= self.beam_width:
                            if self.verbose:
                                pruned = N - len(filtered)
                                if pruned > 0:
                                    print(f"  V-net skipped {pruned}/{N} refinements before evaluation")
                            candidates = filtered

                    # ---------------------------------------------------------
                    # Evaluate remaining candidates
                    # ---------------------------------------------------------
                    for child, child_emb in candidates:

                        self.total_refinements_explored += 1

                        f1, precision, recall, covered_pos = \
                            self.evaluate(child, pos, neg)

                        # termination agent
                        if self.termination_agent is not None:
                            self.termination_agent.observe_quality(f1)
                            if not training:
                                should_stop, reason, confidence = \
                                    self.termination_agent.should_stop_exploring(verbose=0)
                                if should_stop:
                                    print(f"\n🤖 Termination agent: {reason} "
                                          f"(confidence: {confidence:.2f})")
                                    print(f"   Best F1: {best_f1:.3f} | "
                                          f"Explored: {self.total_refinements_explored}")
                                    v_stop = True
                                    break

                        # recursive fragment condition
                        if (
                            precision >= self.operator.precision_threshold
                            and recall <= self.operator.recall_threshold
                            and not training
                        ):
                            if self.termination_agent is not None:
                                self.termination_agent.learn_from_episode()
                            return child

                        all_refinements.append(
                            (child, f1, precision, recall, covered_pos)
                        )

                        # perfect concept
                        if f1 == 1.0 and not training:
                            if self.termination_agent is not None:
                                self.termination_agent.learn_from_episode()
                            return child

                        # update best
                        if f1 > best_f1:
                            best_concept = child
                            best_f1 = f1

                except Exception:
                    continue

                if v_stop:
                    break  # break beam loop

            if v_stop:
                break  # break depth loop

            if not all_refinements:
                if self.verbose:
                    print(f"  No new refinements generated")
                break

            # Beam selection: always sort by raw F1 — V-net never changes the order
            all_refinements.sort(key=lambda x: x[1], reverse=True)
            # ──────────────────────────────────────────────────────────────────

            # Print top concepts at this depth
            if self.verbose:
                print(f"  Generated {len(all_refinements)} unique refinements")
                print(f"  Top 5:")
                for i, (concept, f1, prec, rec, _) in enumerate(all_refinements[:5], 1):
                    concept_str = self.renderer.render(concept)
                    if len(concept_str) > 70:
                        concept_str = concept_str[:70] + "..."
                    print(f"    {i}. F1={f1:.3f} P={prec:.3f} R={rec:.3f} | {concept_str}")

            # Update beam with top concepts
            beam = [concept for concept, _, _, _, _ in all_refinements[:self.beam_width]]

            if self.verbose:
                print(f"  Best F1 so far: {best_f1:.3f}")

        elapsed = time.time() - start
        if self.verbose:
            print(f"\nSearch completed in {elapsed:.1f}s")
            print(f"Total refinements explored: {self.total_refinements_explored}")

        if self.termination_agent is not None:
            self.termination_agent.learn_from_episode()

        return best_concept, self.total_refinements_explored


    def fit(self, pos, neg):

        # recursive learning

        remaining_pos = set(pos)
        fragments = []

        start = time.time()
        while remaining_pos:

            if (time.time() - start) >= self.time_limit:
                print(f"\n⏱ Time limit reached!")
                break

            result = self.beam_search(
                remaining_pos,
                neg,
                start
            )
            # beam_search returns (concept, total) on normal exit, bare concept on early exits
            fragment = result[0] if isinstance(result, tuple) else result

            # V-net: train on episode experience collected during beam_search
            if self.v_net is not None and self._episode_pairs:
                self.form_experiences(self._episode_pairs, self._episode_rewards)
                self.learn_from_replay_memory()

            if fragment is None:
                break

            f1, precision, recall, covered = self.evaluate(
                fragment,
                remaining_pos,
                neg
            )

            fragments.append(fragment)
            remaining_pos -= covered

        if len(fragments) > 1:
            final_concepts = OWLObjectUnionOf(fragments)
            f1, precision, recall, _ = self.evaluate(final_concepts, pos, neg)
            print(f"\n{'=' * 80}")
            print(f"FINAL RESULT")
            print(f"{'=' * 80}")
            print(f"{self.renderer.render(final_concepts)}")
            print(f"\nF1: {f1:.3f}")
            print(f"{'=' * 80}")
            return OWLObjectUnionOf(fragments)

        else:
            f1, precision, recall, _ = self.evaluate(fragments[0], pos, neg)
            print(f"\n{'=' * 80}")
            print(f"FINAL RESULT")
            print(f"{'=' * 80}")
            print(f"{self.renderer.render(fragments[0])}")
            print(f"\nF1: {f1:.3f}")
            print(f"{'=' * 80}")
            return fragments[0], self.total_refinements_explored


def _run_learner(kb, operator, pos, neg, use_v_learning, v_epsilon=1.,
                 v_memory_path='vocell_v_memory.pkl', beam_width=5,
                 max_depth=15, time_limit=60.0, path_embeddings=None,
                 v_net_path=None, verbose=False):
    """Helper: create a fresh VOCELL instance and run fit.
    Returns (concept_str, f1, concepts_explored)."""
    learner = VOCELL(
        kb=kb,
        operator=operator,
        beam_width=beam_width,
        max_depth=max_depth,
        time_limit=time_limit,
        use_v_learning=use_v_learning,
        v_epsilon=v_epsilon,
        v_memory_path=v_memory_path,
        path_embeddings=path_embeddings,
        v_net_path=v_net_path,
        verbose=verbose,
    )
    result = learner.fit(pos=set(pos), neg=set(neg))
    if isinstance(result, tuple):
        concept, total = result
    else:
        concept = result
        total = learner.total_refinements_explored

    # Compute F1 of the returned concept on the *full* pos/neg
    if concept is not None:
        f1, _, _, _ = learner.evaluate(concept, set(pos), set(neg))
        concept_str = learner.renderer.render(concept)
    else:
        f1 = 0.0
        concept_str = "None"
    return concept_str, f1, total


def main():
    """
    Evaluate VOCELL on a learning problem using a pre-trained V-Net.

    Train the V-Net first with:
        python train_vocell_v_net.py
    """
    import os

    LP_NAME    = 'Aunt'
    EMBEDDINGS = 'Experiments/embeddings/Keci_entity_embeddings.csv'
    V_NET_SAVE = f'vocell_v_net_{LP_NAME}.pt'
    EVAL_TIME  = 30   # seconds per run

    # ── Load KB & LP ─────────────────────────────────────────────────────────
    print("Loading knowledge base...")
    kb = KnowledgeBase(path='KGs/Family/family.owl')

    with open('LPs/Family/lps.json', 'r') as f:
        lps = json.load(f)

    p       = lps['problems'][LP_NAME]
    pos     = frozenset({OWLNamedIndividual(IRI.create(i)) for i in p['positive_examples']})
    neg     = frozenset({OWLNamedIndividual(IRI.create(i)) for i in p['negative_examples']})
    print(f"LP: {LP_NAME}  (pos={len(pos)}, neg={len(neg)})")

    print("\nInitializing refinement operator...")
    operator = PruneCELBasedRefinement(
        knowledge_base=kb,
        sparql_endpoint='http://localhost:3030/family/sparql'
    )
    operator.precision_threshold = 1.
    operator.recall_threshold    = 0.6
    operator.set_input_examples(pos, neg)

    results = []

    # (a) Baseline — no V-learning
    print("\n" + "-" * 60)
    print("(a) BASELINE  — no V-learning")
    print("-" * 60)
    concept_str, f1, total = _run_learner(kb, operator, pos, neg,
                                          use_v_learning=False,
                                          time_limit=EVAL_TIME)
    results.append({"label": "Baseline (no V)", "concept": concept_str, "f1": f1, "concepts": total})

    # (b) Pre-trained V-Net — loaded from train_vocell_v_net.py output
    if not os.path.exists(V_NET_SAVE):
        print(f"\n[!] {V_NET_SAVE} not found — run `python train_vocell_v_net.py` first.")
    else:
        print("\n" + "-" * 60)
        print(f"(b) PRE-TRAINED V-Net  (loaded from {V_NET_SAVE})")
        print("-" * 60)
        concept_str, f1, total = _run_learner(kb, operator, pos, neg,
                                              use_v_learning=True,
                                              time_limit=EVAL_TIME,
                                              path_embeddings=EMBEDDINGS,
                                              v_net_path=V_NET_SAVE,
                                              verbose=True)
        results.append({"label": "Pre-trained V-Net", "concept": concept_str, "f1": f1, "concepts": total})

    # ── Summary ───────────────────────────────────────────────────────────────
    baseline_total = results[0]["concepts"]
    baseline_f1    = results[0]["f1"]
    print("\n" + "=" * 100)
    print(f"RESULTS — {LP_NAME}")
    print("=" * 100)
    print(f"  {'Setup':<22} {'F1':>6}  {'vs Base F1':>10}  {'Concepts':>10}  {'vs Base #':>10}  Concept")
    print("-" * 100)
    for r in results:
        f1_delta = f"{r['f1'] - baseline_f1:+.3f}" if r['f1'] != baseline_f1 else ""
        c_delta  = ""
        if r["concepts"] != baseline_total and baseline_total:
            pct     = (1 - r["concepts"] / baseline_total) * 100
            c_delta = f"{pct:+.1f}%"
        cstr = r["concept"]
        if len(cstr) > 45:
            cstr = cstr[:42] + "..."
        print(f"  {r['label']:<22} {r['f1']:>6.3f}  {f1_delta:>10}  {r['concepts']:>10}  {c_delta:>10}  {cstr}")
    print("=" * 100)
    print("\n\u2713 Done!")


if __name__ == "__main__":
    main()