"""
VOCELL: V-learning for Class Expression Learning

Combines K-best Vocell search with V-Net beam pruning.

Search (Vocell — Zhang et al., K-CAP 2025):
  - K-best priority queue (beam_width pops per round) ordered by refinementScore:
      refinementScore = F1 - length * 0.01  (PruneCEL's LengthBasedRefinementScorer)
  - Skip rule: child enqueued only if child_f1 > local_best OR child adds a new role
      local_best ratchets up per batch, blocking same-F1 duplicates
  - Vocell inline recursion: high-precision partial solutions trigger a limited
      sub-search; best sub-result combined as D ⊔ sub and re-inserted into main heap

V-Net pruning (when a checkpoint is loaded):
  - After operator.refine() evaluates all children (instances cached in _inst_cache)
    and the skip rule is applied, ALL surviving children are scored by V-Net in one
    batch.  get_concept_embedding() reuses _inst_cache → zero extra SPARQL calls.
  - Hard prune: children scoring below the batch mean V-Net score are NOT pushed to
    the heap.  This keeps the heap smaller → fewer future pops → fewer subtrees
    expanded → real reduction in total concepts explored.
  - Safety floor: concepts that beat the current best_f1 are never dropped.

V-Learning (online training):
  - BeamVNet (mean/deepsets/settransformer) loaded from checkpoint at inference time
  - Online training: collects (concept_emb, f1) pairs during search, trains after each LP
"""

import hashlib
import heapq
import os
import time
from typing import FrozenSet, Tuple, Dict, List, Set, Optional

from owlapy import owl_expression_to_dl

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
# PruneCELBasedRefinement is only in the local workspace copy of ontolearn,
# not the installed package.  Prepend the local package path so that
# ontolearn.refinement_operators resolves to our version, then force a reload.
import importlib, sys as _sys, pathlib as _pl
_local_onto = str(_pl.Path(__file__).parent / "ontolearn")
import ontolearn as _onto_pkg
if _local_onto not in _onto_pkg.__path__:
    _onto_pkg.__path__.insert(0, _local_onto)
_sys.modules.pop("ontolearn.refinement_operators", None)
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
import numpy as np
from concept_aggregators import (
    get_instance_emb_matrix as _get_inst_emb_mat,
    ConceptVNet as AggConceptVNet,
)

class BeamVNet(nn.Module):
    """V-network for VOCELL beam scoring.

    Input : (N, 3, embedding_dim) — channels [concept_mean, pos_mean, neg_mean]
    Output: (N,) predicted quality scores in [0, 1]

    *** Architecture must match ConceptVNet in train_vocell_v_net.py ***
    """

    def __init__(self, embedding_dim: int, device: str = 'gpu'):
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
                 beam_width: int = 5,
                 max_concepts: int = 500,
                 v_net_weight: float = 0.5,
                 precision_threshold: float = 1.0,
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
            operator: Refinement operator
            beam_width: Concepts popped per round in K-best expansion (default: 5)
            max_concepts: Total heap-pop budget per LP (default: 500)
            v_net_weight: Blend weight for V-Net priority; 0 = pure refinement score (default: 0.5)
            precision_threshold: Precision threshold for Vocell fragment detection (default: 1.0)
            time_limit: Time budget per LP in seconds (default: 300)
        """
        self.kb = kb
        self.operator = operator
        self.beam_width = beam_width
        self.max_concepts = max_concepts
        self.v_net_weight = v_net_weight
        self.time_limit = time_limit
        self.precision_threshold = precision_threshold
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
        self.agg_v_net: Optional[AggConceptVNet] = None  # deepsets / settransformer checkpoint
        self.v_optimizer = None
        self.v_net_trained = False       # True only when BeamVNet (mean) is ready
        self.agg_v_net_trained = False   # True only when AggConceptVNet is loaded
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
        elif not use_v_learning and v_net_path and os.path.isfile(v_net_path) \
                and self.df_embeddings is not None:
            # Inference-only: load a pre-trained V-net without online RL machinery
            self.v_net = BeamVNet(self.embedding_dim, device)  # needed for mean nets
            self.load(v_net_path)
    
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
            print("Warning: no embeddings DataFrame available — returning zero embedding")
            return torch.zeros(1, 1, self.embedding_dim)
        iris = [ind.str for ind in individuals]
        if not iris:
            return torch.zeros(1, 1, self.embedding_dim)
        valid = [i for i in iris if i in self.df_embeddings.index]
        if not valid:
            return torch.zeros(1, 1, self.embedding_dim)
        vals = self.df_embeddings.loc[valid].values
        emb = torch.from_numpy(vals.mean(axis=0)).float()
        return emb.view(1, 1, self.embedding_dim)

    def get_concept_embedding(self, concept: OWLClassExpression) -> Optional[torch.Tensor]:
        """Mean embedding of concept instances. Returns (1, 1, dim) tensor, or None if no embeddings.

        Checks the refinement operator's _inst_cache first to avoid a redundant
        kb.individuals_set() call (refine() already populated it for every candidate).
        """
        if self.df_embeddings is None:
            return None
        # Reuse instances already fetched by the refinement operator (no extra KB call)
        op_cache = getattr(self.operator, '_inst_cache', None)
        if op_cache is not None:
            key = str(concept)
            instances = op_cache.get(key)
            if instances is not None:
                return self._get_embedding_from_ind_set(instances)
        # Fallback: concept not yet in cache (e.g. OWLThing at root)
        return self._get_embedding_from_ind_set(self.get_instances(concept))

    def _get_concept_emb_matrix(self, concept: OWLClassExpression) -> torch.Tensor:
        """Raw (K, dim) embedding matrix of concept instances, used by DeepSets / SetTransformer V-nets.
        Returns (0, dim) when no embeddings are available or the concept has no known instances."""
        if self.df_embeddings is None:
            return torch.zeros(0, self.embedding_dim)
        iris = [ind.str for ind in self.get_instances(concept)]
        return _get_inst_emb_mat(iris, self.df_embeddings)

    # ── V-learning ───────────────────────────────────

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
        ep = self.emb_pos #if self.emb_pos is not None else torch.zeros(1, 1, self.embedding_dim)
        en = self.emb_neg #if self.emb_neg is not None else torch.zeros(1, 1, self.embedding_dim)

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
        """Load V-net weights from disk.

        'mean' checkpoints (or old checkpoints without an 'agg_type' key) are loaded
        into the existing BeamVNet (self.v_net) — unchanged behaviour.

        'deepsets' / 'settransformer' checkpoints instantiate a ConceptVNet from
        concept_aggregators and store it in self.agg_v_net; self.v_net is left
        untouched so that online V-learning replay continues to work normally.
        """
        ckpt     = torch.load(path, weights_only=True)
        agg_type = ckpt.get('agg_type', 'mean')

        if agg_type == 'mean':
            # ── original mean path ────────────────────────────────────────────
            if self.v_net is None:
                print("V-net not initialized (no embeddings provided).")
                return
            self.v_net.load_state_dict(ckpt['model_state_dict'])
            self.v_net_trained = ckpt.get('v_net_trained', True)
            self.v_net.eval()
            print(f"V-Net (mean) loaded ← {path}  (trained={self.v_net_trained})")
        else:
            # ── deepsets / settransformer path ─────────────────────────────
            emb_dim = ckpt.get('embedding_dim', self.embedding_dim)
            self.agg_v_net = AggConceptVNet(emb_dim, agg_type, self.device)
            self.agg_v_net.load_state_dict(ckpt['model_state_dict'])
            self.agg_v_net.eval()
            self.agg_v_net_trained = True   # only this flag — mean filter stays off
            print(f"V-Net ({agg_type}) loaded ← {path}")

    # ─────────────────────────────────────────────────────────────────────────

    def best_first_search(self, pos: Set, neg: Set, start, max_concepts: int = 500,
                          _subproblem_keys=None, _allow_recursion: bool = True,
                          training: bool = False) -> OWLClassExpression:
        """
        K-best search with Vocell inline recursion and V-Net batch hard-pruning.

        Search priority = refinementScore = F1 - length * length_penalty
          (pure refinement score — no V-Net blending in the priority)

        Skip rule: child enqueued only if child_f1 > local_best OR adds a new role.
          local_best ratchets up per batch, blocking same-F1 duplicates.

        V-Net hard pruning (when v_net_weight > 0 and a V-Net is loaded):
          After operator.refine() evaluates all children and the skip rule is applied,
          ALL surviving children are batch-scored by the V-Net in a single forward pass.
          get_concept_embedding() is FREE here: instances are already in _inst_cache.
          Children below the batch mean V-Net score are NOT pushed to the heap.
          Safety floor: any child that beats the current best_f1 is always kept.

        Vocell recursion (when _allow_recursion=True):
          After each K-best batch, high-precision fragment candidates trigger a limited
          sub-search on (E+ \ E+', E-).  Best sub-result combined as D ⊔ sub_result
          and pushed back into the main heap.
        """
        if _subproblem_keys is None:
            _subproblem_keys = set()

        min_recursion_pos = max(2, int(len(pos) * 0.1 + 0.999))
        _RECURSIVE_BUDGET = max(10, self.beam_width * 2)

        # Pre-compute pos/neg embeddings for V-Net priority blending
        if self.v_net is not None:
            self.emb_pos = self._get_embedding_from_ind_set(pos)
            self.emb_neg = self._get_embedding_from_ind_set(neg)
        if self.agg_v_net is not None and self.df_embeddings is not None:
            _get_inst_emb_mat([i.str for i in pos], self.df_embeddings).to(self.device)
            _get_inst_emb_mat([i.str for i in neg], self.df_embeddings).to(self.device)

        # Reset termination agent for this episode
        if self.termination_agent is not None:
            self.termination_agent.reset_for_new_episode()

        # Episode buffers for online V-Net training
        self._episode_pairs   = []
        self._episode_rewards = []

        # Seed heap with ⊤
        top_f1, _, _, top_covered = self.evaluate(OWLThing, pos, neg)
        top_tp    = len(top_covered)
        top_score = self.operator.refinement_score(OWLThing, top_f1, top_tp)

        # Min-heap: (neg_priority, neg_f1, seq, concept, f1, tp)
        # neg_priority = -priority  → highest priority pops first
        # seq          = insertion counter → unique tiebreaker
        _seq = 0
        heap = [(-top_score, -top_f1, _seq, OWLThing, top_f1, top_tp)]
        heapq.heapify(heap)

        seen         = {str(OWLThing)}
        best_concept = OWLThing
        best_f1      = top_f1
        pops         = 0

        if self.verbose:
            label    = "" if _allow_recursion else " [sub-problem]"
            v_active = ((self.v_net is not None and self.v_net_trained)
                        or (self.agg_v_net is not None and self.agg_v_net_trained))
            mode_tag = " [V-Net ON]" if (v_active and not training and self.v_net_weight > 0) else ""
            if self.termination_agent is not None and not training:
                mode_tag += " [TermAgent ON]"
            print(f"\nK-best search{label} (budget={max_concepts}, beam={self.beam_width}, "
                  f"|pos|={len(pos)}, penalty={self.operator.length_penalty}){mode_tag}")
            print("=" * 80)

        while heap and pops < max_concepts:
            if (time.time() - start) >= self.time_limit:
                break

            # ── K-best batch: pop beam_width concepts per round ───────────────
            batch = []
            while heap and len(batch) < self.beam_width:
                batch.append(heapq.heappop(heap))

            batch_fragments = []   # Vocell candidates collected this batch
            done = False

            for neg_priority, _, _, parent, parent_f1, parent_tp in batch:
                pops += 1
                if pops > max_concepts or (time.time() - start) >= self.time_limit:
                    done = True
                    break

                parent_n_q = self.operator.count_quantifiers(parent)

                if self.verbose:
                    print(f"  [{pops:4d}] score={-neg_priority:.4f}  F1={parent_f1:.3f}  "
                          f"{self.renderer.render(parent)[:70]}")

                # local_best ratchets up within this expansion:
                # once Female (F1=0.804) is accepted, ¬Male / ¬Father (also 0.804) are dropped.
                local_best = best_f1
                survivors  = []   # (child, child_f1, child_score, tp) — post-skip-rule

                try:
                    for child, child_f1, _, precision, recall, tp in self.operator.refine(parent):
                        child_str = str(child)
                        if child_str in seen:
                            continue
                        seen.add(child_str)
                        self.total_refinements_explored += 1

                        # ── Vocell: collect fragment candidates ──────────────
                        # Checked BEFORE skip rule so high-precision / low-recall
                        # concepts aren't missed just because they don't beat local_best.
                        if (_allow_recursion and not training
                                and precision >= self.operator.precision_threshold
                                and tp >= min_recursion_pos
                                and tp < len(pos) - 1):
                            inst_cache = getattr(self.operator, '_inst_cache', {})
                            batch_fragments.append((child, child_f1, tp, inst_cache.get(child_str)))

                        # ── Skip rule ────────────────────────────────────────
                        child_n_q  = self.operator.count_quantifiers(child)
                        added_role = child_n_q > parent_n_q
                        if child_f1 <= local_best and not added_role:
                            continue
                        local_best = max(local_best, child_f1)

                        # # Termination-agent observation
                        # if self.termination_agent is not None:
                        #     self.termination_agent.observe_quality(child_f1)
                        #     if not training:
                        #         should_stop, reason, confidence = \
                        #             self.termination_agent.should_stop_exploring(verbose=0)
                        #         if should_stop:
                        #             print(f"\nTermination agent: {reason} "
                        #                   f"(confidence: {confidence:.2f})")
                        #             print(f"  Best F1: {best_f1:.3f} | "
                        #                   f"Explored: {self.total_refinements_explored}")
                        #             done = True
                        #             break

                        child_score = self.operator.refinement_score(child, child_f1, tp)

                        if child_f1 == 1.0 and not training:
                            print(f"\n✓ PERFECT F1 at pop {pops}!")
                            print(f"  {self.renderer.render(child)}")
                            if self.termination_agent is not None:
                                self.termination_agent.learn_from_episode()
                            return child

                        if child_f1 > best_f1:
                            best_concept = child
                            best_f1      = child_f1

                        # # Episode collection for online V-Net training
                        # if training and self.v_net is not None:
                        #     try:
                        #         child_emb = self.get_concept_embedding(child)
                        #         if child_emb is not None:
                        #             self._episode_pairs.append(child_emb)
                        #             self._episode_rewards.append(child_f1)
                        #     except Exception:
                        #         pass

                        survivors.append((child, child_f1, child_score, tp))

                except Exception as e:
                    print(f"  Exception refining {self.renderer.render(parent)}: {e}")

                # ── V-Net batch hard-pruning ──────────────────────────────────
                # Score all post-skip-rule survivors at once, then drop the bottom
                # half by V-Net score.  get_concept_embedding() is FREE here because
                # instances are already in _inst_cache from operator.refine() —
                # zero extra SPARQL calls.  Concepts not pushed to the heap are
                # never popped, never expanded, and their subtrees never evaluated
                # → real reduction in concepts explored in future rounds.
                if (not training
                        and self.v_net_weight > 0
                        and self.v_net is not None
                        and self.v_net_trained
                        and self.emb_pos is not None
                        and len(survivors) > 1):
                    embs, valid_idxs = [], []
                    for i, (child, _, _, _) in enumerate(survivors):
                        emb = self.get_concept_embedding(child)
                        if emb is not None:
                            embs.append(emb)
                            valid_idxs.append(i)

                    if embs:
                        N           = len(embs)
                        child_batch = torch.cat(embs, dim=0)           # (N, 1, dim)
                        X = torch.cat([
                            child_batch,
                            self.emb_pos.repeat(N, 1, 1),
                            self.emb_neg.repeat(N, 1, 1),
                        ], dim=1)                                       # (N, 3, dim)
                        self.v_net.eval()
                        with torch.no_grad():
                            v_scores = self.v_net(X).tolist()
                            # print(v_scores)
                            # exit(0)

                        # Keep concepts scoring at or above the mean V-Net score.
                        # Safety floor: never drop a concept that beat best_f1
                        # (those are always worth expanding).
                        threshold = max(float(np.mean(v_scores)), best_f1)
                        keep_set  = {valid_idxs[i]
                                     for i, v in enumerate(v_scores)
                                     if v >= threshold}
                        for i, (_, child_f1, _, _) in enumerate(survivors):
                            if child_f1 >= best_f1:          # safety floor
                                keep_set.add(i)

                        n_before  = len(survivors)
                        survivors = [s for i, s in enumerate(survivors) if i in keep_set]
                        if self.verbose:
                            print(f"    V-Net: kept {len(survivors)}/{n_before} "
                                  f"(pruned {n_before - len(survivors)})")
                # ─────────────────────────────────────────────────────────────

                for child, child_f1, child_score, tp in survivors:
                    _seq += 1
                    heapq.heappush(heap, (-child_score, -child_f1, _seq, child, child_f1, tp))

                if done:
                    break

            if done:
                break

            # ── Vocell: inline recursion after each batch ─────────────────
            if _allow_recursion and not training and batch_fragments and best_f1 < 1.0:
                for fragment, frag_f1, frag_tp, frag_instances in batch_fragments:
                    if (time.time() - start) >= self.time_limit:
                        break

                    if frag_instances is None:
                        frag_instances = self.get_instances(fragment)

                    remaining_pos = pos - frag_instances
                    if len(remaining_pos) < 2:
                        continue

                    sub_key = frozenset(i.str for i in remaining_pos)
                    if sub_key in _subproblem_keys:
                        continue
                    _subproblem_keys.add(sub_key)

                    frag_label = self.renderer.render(fragment)[:50]
                    print(f"\n  ↳ [Vocell] '{frag_label}' covers {frag_tp}/{len(pos)} pos"
                          f" → sub-problem: {len(remaining_pos)} remaining")

                    saved_cache = dict(getattr(self.operator, '_inst_cache', {}))
                    self.operator.set_input_examples(frozenset(remaining_pos), frozenset(neg))

                    sub_result = self.best_first_search(
                        remaining_pos, neg, start,
                        max_concepts=_RECURSIVE_BUDGET,
                        _subproblem_keys=_subproblem_keys,
                        _allow_recursion=False,
                        training=False,
                    )

                    sub_cache = dict(getattr(self.operator, '_inst_cache', {}))
                    self.operator.set_input_examples(frozenset(pos), frozenset(neg))
                    if hasattr(self.operator, '_inst_cache'):
                        self.operator._inst_cache.update(saved_cache)
                        self.operator._inst_cache.update(sub_cache)

                    if sub_result is None:
                        continue

                    combined     = OWLObjectUnionOf([fragment, sub_result])
                    combined_str = str(combined)
                    if combined_str in seen:
                        continue
                    seen.add(combined_str)

                    c_f1, _, _, c_covered = self.evaluate(combined, pos, neg)
                    c_tp    = len(c_covered)
                    c_score = self.operator.refinement_score(combined, c_f1, c_tp)
                    _seq += 1
                    heapq.heappush(heap, (-c_score, -c_f1, _seq, combined, c_f1, c_tp))
                    print(f"    ↳ Combined: F1={c_f1:.3f}  {self.renderer.render(combined)[:60]}")

                    if c_f1 == 1.0:
                        print(f"\n✓ PERFECT F1 via recursion!")
                        if self.termination_agent is not None:
                            self.termination_agent.learn_from_episode()
                        return combined

                    if c_f1 > best_f1:
                        best_concept = combined
                        best_f1      = c_f1
            # ─────────────────────────────────────────────────────────────────

        elapsed = time.time() - start
        if self.verbose:
            print(f"\nDone in {elapsed:.1f}s  pops={pops}  "
                  f"explored={self.total_refinements_explored}  best_f1={best_f1:.3f}")

        if self.termination_agent is not None:
            self.termination_agent.learn_from_episode()

        return best_concept


    def learn_recursive(self, pos, neg, training: bool = False, allow_recursion: bool = True):
        """
        Single-pass Vocell learning with optional online V-Net training.
        Union construction is handled internally via inline recursive sub-problems.
        """
        start = time.time()
        self.operator.set_input_examples(frozenset(pos), frozenset(neg))

        concept = self.best_first_search(
            set(pos), set(neg), start,
            max_concepts=self.max_concepts,
            training=training,
            _allow_recursion=allow_recursion,
        )

        # # Online V-Net training on episode experience collected during search
        # if self.v_net is not None and self._episode_pairs:
        #     self.form_experiences(self._episode_pairs, self._episode_rewards)
        #     self.learn_from_replay_memory()

        f1 = 0.0
        if concept is not None:
            f1, _, _, _ = self.evaluate(concept, set(pos), set(neg))

        print(f"\n{'=' * 80}")
        print(f"FINAL RESULT")
        print(f"{'=' * 80}")
        print(f"{self.renderer.render(concept) if concept is not None else 'None'}")
        print(f"\nF1: {f1:.3f}")
        print(f"{'=' * 80}")
        return concept, self.total_refinements_explored

    # keep fit() as an alias so external scripts don't break
    def fit(self, pos, neg, allow_recursion: bool = True):
        return self.learn_recursive(pos, neg, allow_recursion=allow_recursion)


def _run_learner(kb, operator, pos, neg, use_v_learning, v_epsilon=1.,
                 v_memory_path='vocell_v_memory.pkl', beam_width=5,
                 max_concepts=500, time_limit=60.0, path_embeddings=None,
                 v_net_path=None, v_net_weight=0.5, verbose=False,
                 allow_recursion=True):
    """Helper: create a fresh VOCELL instance and run learn_recursive.
    Returns (concept_str, f1, concepts_explored)."""
    learner = VOCELL(
        kb=kb,
        operator=operator,
        beam_width=beam_width,
        max_concepts=max_concepts,
        v_net_weight=v_net_weight,
        time_limit=time_limit,
        use_v_learning=use_v_learning,
        v_epsilon=v_epsilon,
        v_memory_path=v_memory_path,
        path_embeddings=path_embeddings,
        v_net_path=v_net_path,
        verbose=verbose,
    )
    result = learner.learn_recursive(pos=set(pos), neg=set(neg), allow_recursion=allow_recursion)
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
    Evaluate VOCELL on one or more learning problems and compare V-Net variants.

    Train checkpoints first with:
        python train_vocell_v_net.py --strategy bootstrap --output_dir Carcinogenesis_mean ...
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate VOCELL and compare V-Net filtering variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Problem ───────────────────────────────────────────────────────────────
    parser.add_argument('--lps_file', default='LPs/Family/lps_difficult.json',
                        help='Path to the LP JSON file.')
    parser.add_argument('--num_lps', type=int, default=0,
                        help='How many LPs to evaluate (0 = all).')
    parser.add_argument('--kb',       default='KGs/Family/family.owl',
                        help='Path to the OWL knowledge base.')
    parser.add_argument('--sparql',   default='http://localhost:3030/family/sparql',
                        dest='sparql_endpoint',
                        help='SPARQL endpoint URL.')
    # ── Search ────────────────────────────────────────────────────────────────
    parser.add_argument('--time_limit', type=float, default=600.0,
                        help='Time budget per run in seconds.')
    parser.add_argument('--beam_width', type=int,   default=5,
                        help='Beam width.')
    parser.add_argument('--max_concepts', type=int, default=500,
                        help='Maximum heap pops per LP (search budget).')
    parser.add_argument('--precision_threshold', type=float, default=1.0,
                        help='Operator precision threshold.')
    parser.add_argument('--recall_threshold',    type=float, default=0.6,
                        help='Operator recall threshold.')
    # ── Embeddings & checkpoints ──────────────────────────────────────────────
    parser.add_argument('--embeddings', default='Experiments/embeddings/Keci_entity_embeddings.csv',
                        help='Path to entity embeddings CSV.')
    parser.add_argument('--agg_types', nargs='+', default=['mean'],
                        choices=['mean', 'deepsets', 'settransformer'],
                        help='Aggregation strategies to compare (space-separated).')
    parser.add_argument('--training_strategies', nargs='+', default=['loocv'],
                        choices=['loocv', 'bootstrap'],
                        help='Training strategies whose checkpoints to load.')
    parser.add_argument('--checkpoint_dir', default=None,
                        help=('Override checkpoint directory for all variants. '
                              'Default: Family_{agg_type} per aggregator '
                              '(matches train_vocell_v_net.py output_dir).'))
    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip the no-V-net baseline run.')
    parser.add_argument('--allow_recursion', action='store_true',
                        help='Enable PruneCEL-R inline recursion for both baseline and V-Net runs.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-depth beam-search details.')
    args = parser.parse_args()

    # ── Load KB & LP file ─────────────────────────────────────────────────────
    print("Loading knowledge base...")
    kb = KnowledgeBase(path=args.kb)

    with open(args.lps_file, 'r') as f:
        lps = json.load(f)

    # Support both {"problems": {...}} and flat {"LP_name": {...}} formats.
    lps_problems  = lps.get('problems', lps) if isinstance(lps, dict) else {}
    all_problems  = list(lps_problems.items())
    total_lps     = len(all_problems)
    n_to_run      = total_lps if args.num_lps == 0 else min(args.num_lps, total_lps)

    print(f"LP file      : {args.lps_file}")
    print(f"Total LPs    : {total_lps}")
    print(f"LPs to run   : {n_to_run}")

    problems_to_run = all_problems[:n_to_run]
    # Build two O(1) indexes to remap LP example IRIs onto the KB-canonical
    # namespace (e.g. /animals/trex01 → /animals#trex01).
    _kb_ind_by_iri:   dict = {str(ind.iri): ind for ind in kb.individuals()}
    _kb_ind_by_local: dict = {ind.iri.get_remainder(): ind for ind in kb.individuals()}

    def _resolve_ind(iri_str: str) -> OWLNamedIndividual:
        if iri_str in _kb_ind_by_iri:
            return _kb_ind_by_iri[iri_str]
        local = IRI.create(iri_str).get_remainder()
        return _kb_ind_by_local.get(local, OWLNamedIndividual(IRI.create(iri_str)))
    # ── Build variant list (same for every LP) ────────────────────────────────
    variant_specs = []
    if not args.no_baseline:
        variant_specs.append(("Baseline (no V-net)", None, None))

    for agg in args.agg_types:
        ckpt_dir = args.checkpoint_dir or f'Family_{agg}'
        for strat in args.training_strategies:
            if strat == 'loocv':
                # LOOCV checkpoint is LP-specific; path resolved per LP below
                variant_specs.append((f'V-Net [{agg} / loocv]', agg, 'loocv'))
            else:
                ckpt  = os.path.join(ckpt_dir, f'vocell_v_net_bootstrap_{agg}.pt')
                variant_specs.append((f'V-Net [{agg} / bootstrap]', agg, ckpt))

    # ── Accumulate results across all LPs ────────────────────────────────────
    all_results = []   # list of {lp_name, label, f1, concepts, concept}

    for lp_idx, (lp_name, p) in enumerate(problems_to_run, 1):
        pos = frozenset(_resolve_ind(i) for i in p['positive_examples'])
        neg = frozenset(_resolve_ind(i) for i in p['negative_examples'])

        # Shorten long LP names for display
        lp_display = lp_name if len(lp_name) <= 50 else lp_name[:47] + "..."

        print(f"\n{'=' * 100}")
        print(f"LP {lp_idx}/{n_to_run}: {lp_display}")
        print(f"  pos={len(pos)}  neg={len(neg)}")
        print(f"{'=' * 100}")

        def make_operator(pos=pos, neg=neg):
            op = PruneCELBasedRefinement(
                knowledge_base=kb,
                sparql_endpoint=args.sparql_endpoint,
            )
            op.precision_threshold = args.precision_threshold
            op.recall_threshold    = args.recall_threshold
            op.set_input_examples(pos, neg)
            return op

        lp_results = []

        for label, agg, strat_or_ckpt in variant_specs:
            use_vl = agg is not None  # False only for baseline

            # Resolve checkpoint path
            v_net_path = None
            if use_vl:
                ckpt_dir = args.checkpoint_dir or f'Family_{agg}'
                if strat_or_ckpt == 'loocv':
                    # LOOCV checkpoint is named after the LP — not meaningful for
                    # non-Family datasets, but we try anyway.
                    safe_name = lp_name.replace(' ', '_').replace('/', '_')
                    v_net_path = os.path.join(ckpt_dir, f'vocell_v_net_{safe_name}_{agg}.pt')
                else:
                    v_net_path = strat_or_ckpt   # already a full path

            print(f"\n  [{label}]")
            if use_vl and not os.path.exists(v_net_path):
                print(f"    [!] Checkpoint not found: {v_net_path} — skipping")
                continue

            _t0 = time.time()
            concept_str, f1, total = _run_learner(
                kb, make_operator(), pos, neg,
                use_v_learning=use_vl,
                time_limit=args.time_limit,
                beam_width=args.beam_width,
                max_concepts=args.max_concepts,
                path_embeddings=args.embeddings if use_vl else None,
                v_net_path=v_net_path,
                verbose=args.verbose,
                allow_recursion=args.allow_recursion,
            )
            _runtime = time.time() - _t0
            lp_results.append({
                "lp_name":  lp_name,
                "label":    label,
                "f1":       f1,
                "concepts": total,
                "concept":  concept_str,
                "runtime":  _runtime,
            })

        # ── Per-LP summary table ───────────────────────────────────────────
        if lp_results:
            base_f1    = lp_results[0]["f1"]
            base_total = lp_results[0]["concepts"]
            print(f"\n  {'Setup':<30} {'F1':>6}  {'Δ F1':>8}  {'Concepts':>10}  {'Time (s)':>9}  Concept")
            print(f"  {'-' * 90}")
            for r in lp_results:
                delta = f"{r['f1'] - base_f1:+.3f}" if r['f1'] != base_f1 else "      "
                cstr  = r["concept"] if len(r["concept"]) <= 40 else r["concept"][:37] + "..."
                print(f"  {r['label']:<30} {r['f1']:>6.3f}  {delta:>8}  {r['concepts']:>10}  {r['runtime']:>8.1f}s  {cstr}")

        all_results.extend(lp_results)

    # ── Grand summary (average F1 per variant across all LPs) ────────────────
    if all_results and n_to_run > 1:
        from collections import defaultdict
        import numpy as _np
        agg_f1 = defaultdict(list)
        agg_ct = defaultdict(list)
        agg_rt = defaultdict(list)
        for r in all_results:
            agg_f1[r["label"]].append(r["f1"])
            agg_ct[r["label"]].append(r["concepts"])
            agg_rt[r["label"]].append(r["runtime"])

        print(f"\n{'=' * 110}")
        print(f"GRAND SUMMARY  ({n_to_run} LPs)")
        print(f"{'=' * 110}")
        print(f"  {'Setup':<30} {'Avg F1':>8}  {'Std F1':>8}  {'Avg Concepts':>14}  {'Avg Time (s)':>13}  {'Total Time':>11}")
        print(f"  {'-' * 90}")
        for label, f1s in agg_f1.items():
            cts = agg_ct[label]
            rts = agg_rt[label]
            print(f"  {label:<30} {_np.mean(f1s):>8.3f}  {_np.std(f1s):>8.3f}  {_np.mean(cts):>14.0f}  {_np.mean(rts):>12.1f}s  {sum(rts):>10.1f}s")
        print("=" * 110)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()

# Family (all LPs):
#   python vocell.py --lps_file LPs/Family/lps_difficult.json --kb KGs/Family/family.owl --sparql http://localhost:3030/family/sparql --embeddings Experiments/embeddings/Keci_entity_embeddings.csv --agg_types mean --training_strategies loocv --num_lps 0 --verbose
#
# Carcinogenesis (first 3 LPs, bootstrap, no baseline):
#   python vocell.py --lps_file LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl \
#     --sparql http://localhost:3030/carcinogenesis/sparql \
#     --embeddings ../Ontolearn_ISWC/datasets/carcinogenesis/embeddings/DeCaL_entity_embeddings.csv \
#     --checkpoint_dir Carcinogenesis_mean --agg_types mean --training_strategies bootstrap \
#     --num_lps 3 --beam_width 15 --no_baseline