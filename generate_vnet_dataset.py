"""
generate_vnet_dataset.py
========================
Runs the Vocell K-best search (with inline recursion) on every LP and records
every concept popped from the heap as a training data point.

Unlike the old depth-by-depth beam search, this strategy:
  - Finds the F1=1.0 solution via union (A ⊔ B) through Vocell recursion
  - Records the actual winning path so the V-Net learns to distinguish
    on-path nodes from dead-ends
  - Labels are propagated by train_vocell_v_net.py: each node's label =
    best F1 reachable in its subtree (not just its own F1)

What is saved per LP
--------------------
  lp_name           : str    — name of the LP
  positive_examples : list   — IRI strings of E+
  negative_examples : list   — IRI strings of E-
  best_concept_str  : str    — DL string of the best concept found
  best_f1           : float  — F1 of that concept w.r.t. full E+/E-
  elapsed_seconds   : float  — wall time for this LP
  nodes             : list   — recorded search-tree nodes, each with:
      concept_str   : str    — DL string of the concept
      f1            : float  — F1 w.r.t. full E+/E-
      depth         : int    — quantifier depth of the concept
      parent_str    : str|null — DL string of the parent concept
      instance_iris : list   — IRI strings of every individual in I(C)

Run as:
  python generate_vnet_dataset.py
"""

import heapq
import json
import time
from typing import Dict, List, Optional, Set, Tuple

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import PruneCELBasedRefinement
from owlapy.class_expression import OWLClassExpression, OWLObjectUnionOf, OWLThing
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer

from test_covering_algorithm import BeamSearchLearner


# ─────────────────────────────────────────────────────────────────────────────
#  RecordingBeamSearchLearner
# ─────────────────────────────────────────────────────────────────────────────

class RecordingBeamSearchLearner(BeamSearchLearner):
    """
    Runs the Vocell K-best search with inline recursion and records every
    concept popped from the heap as a training data point.

    The key difference from depth-by-depth beam search:
      - Uses a priority heap sorted by refinement_score (F1 - depth*0.01)
      - When a fragment covers a subset of pos with precision=1.0, recursively
        solves the remaining pos and records those sub-problem nodes too
      - This means union concepts (A ⊔ B) appear in the recorded data,
        including the F1=1.0 solution for LPs like Cousin

    After calling record_vocell_search(), results are in self.recorded_nodes.
    """

    def __init__(self, *args, top_k_record: int = 25,
                 precision_threshold: float = 1.0,
                 min_recursion_pos: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k_record        = top_k_record
        self.precision_threshold = precision_threshold
        self.min_recursion_pos   = min_recursion_pos
        self.recorded_nodes: List[Dict] = []

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_instances(self, concept: OWLClassExpression) -> frozenset:
        return frozenset(self.kb.individuals_set(concept))

    def _f1(self, instances, pos, neg) -> float:
        tp = len(instances & pos)
        fp = len(instances & neg)
        fn = len(pos - instances)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _make_node(
        self,
        concept:    OWLClassExpression,
        f1:         float,
        iris:       List[str],
        depth:      int,
        parent_str: Optional[str],
    ) -> Dict:
        return {
            "concept_str":   self.renderer.render(concept),
            "f1":            round(f1, 6),
            "depth":         depth,
            "parent_str":    parent_str,
            "instance_iris": iris,
        }

    # ── main method ───────────────────────────────────────────────────────────

    def record_vocell_search(
        self,
        pos:        Set[OWLNamedIndividual],
        neg:        Set[OWLNamedIndividual],
        start_time: float,
        parent_str: Optional[str] = None,
        _subproblem_keys: Optional[Set] = None,
        _allow_recursion: bool = True,
    ) -> Optional[OWLClassExpression]:
        """
        K-best search with Vocell recursion.  Records every popped concept.
        """
        if _subproblem_keys is None:
            _subproblem_keys = set()

        pos_f = frozenset(pos)
        neg_f = frozenset(neg)
        self.operator.set_input_examples(pos_f, neg_f)

        renderer   = self.renderer
        _seq       = 0
        heap: list = []
        seen:  Set[str] = set()
        best_f1      = 0.0
        best_concept = None
        recorded     = 0

        # ── seed heap with ⊤ ─────────────────────────────────────────────────
        root_inst = self._get_instances(OWLThing)
        root_f1   = self._f1(root_inst, pos_f, neg_f)
        root_str  = renderer.render(OWLThing)
        if parent_str is None:   # only record root for the top-level call
            self.recorded_nodes.append(self._make_node(
                OWLThing, root_f1,
                sorted(i.str for i in root_inst), 0, None
            ))
            recorded += 1
        heapq.heappush(heap, (-root_f1, 0, _seq, OWLThing, root_f1, root_str))
        seen.add(root_str)

        while heap and recorded < self.top_k_record:
            if time.time() - start_time >= self.time_limit:
                break

            _, _, _, parent, parent_f1, par_str = heapq.heappop(heap)
            parent_n_q = self.operator.count_quantifiers(parent)
            local_best = best_f1
            batch_fragments = []

            try:
                for child, child_f1, _, precision, recall, tp in self.operator.refine(parent):
                    child_str = renderer.render(child)
                    if child_str in seen:
                        continue
                    seen.add(child_str)

                    # collect fragment candidates for recursion
                    if (_allow_recursion
                            and precision >= self.precision_threshold
                            and tp >= self.min_recursion_pos
                            and tp < len(pos_f) - 1):
                        inst_cache = getattr(self.operator, '_inst_cache', {})
                        batch_fragments.append((
                            child, child_f1, tp,
                            inst_cache.get(child_str)
                        ))

                    child_n_q  = self.operator.count_quantifiers(child)
                    added_role = child_n_q > parent_n_q
                    if child_f1 <= local_best and not added_role:
                        continue
                    local_best = max(local_best, child_f1)

                    score = child_f1 - child_n_q * 0.01
                    if child_f1 > best_f1:
                        best_f1      = child_f1
                        best_concept = child

                    _seq += 1
                    heapq.heappush(heap, (-score, -child_f1, _seq, child, child_f1, child_str))

            except Exception:
                pass

            # ── Vocell recursion ─────────────────────────────────────────────
            for fragment, frag_f1, frag_tp, frag_instances in batch_fragments:
                if time.time() - start_time >= self.time_limit:
                    break
                if recorded >= self.top_k_record:
                    break

                if frag_instances is None:
                    frag_instances = self._get_instances(fragment)

                remaining_pos = pos_f - frag_instances
                if len(remaining_pos) < 2:
                    continue

                sub_key = frozenset(i.str for i in remaining_pos)
                if sub_key in _subproblem_keys:
                    continue
                _subproblem_keys.add(sub_key)

                frag_str = renderer.render(fragment)
                print(f"    ↳ recursion: '{frag_str[:50]}' covers {frag_tp}/{len(pos_f)} pos")

                saved_cache = dict(getattr(self.operator, '_inst_cache', {}))
                sub_result = self.record_vocell_search(
                    set(remaining_pos), set(neg_f),
                    start_time        = start_time,
                    parent_str        = frag_str,
                    _subproblem_keys  = _subproblem_keys,
                    _allow_recursion  = False,
                )
                self.operator.set_input_examples(pos_f, neg_f)
                if hasattr(self.operator, '_inst_cache'):
                    self.operator._inst_cache.update(saved_cache)

                if sub_result is None:
                    continue

                combined     = OWLObjectUnionOf([fragment, sub_result])
                combined_str = renderer.render(combined)
                combined_inst = self._get_instances(fragment) | self._get_instances(sub_result)
                combined_f1   = self._f1(combined_inst, pos_f, neg_f)

                # Record the union concept as a node (parent = fragment)
                self.recorded_nodes.append(self._make_node(
                    combined, combined_f1,
                    sorted(i.str for i in combined_inst),
                    self.operator.count_quantifiers(combined),
                    frag_str,
                ))
                recorded += 1

                if combined_f1 > best_f1:
                    best_f1      = combined_f1
                    best_concept = combined

                if combined_f1 == 1.0:
                    print(f"    ✓ F1=1.0 union: {combined_str[:60]}")
                    return combined

            # ── record this popped concept ────────────────────────────────────
            inst_cache = getattr(self.operator, '_inst_cache', {})
            child_inst = inst_cache.get(renderer.render(parent))
            if child_inst is None:
                try:
                    child_inst = self._get_instances(parent)
                except Exception:
                    child_inst = frozenset()
            self.recorded_nodes.append(self._make_node(
                parent, parent_f1,
                sorted(i.str for i in child_inst),
                self.operator.count_quantifiers(parent),
                par_str if par_str != root_str else None,
            ))
            recorded += 1

        return best_concept

    # keep old name as alias so solve_lp doesn't need changing
    def record_beam_search(
        self,
        pos: Set[OWLNamedIndividual],
        neg: Set[OWLNamedIndividual],
    ) -> Optional[OWLClassExpression]:
        self.recorded_nodes = []
        return self.record_vocell_search(
            pos, neg, start_time=time.time()
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-LP solver
# ─────────────────────────────────────────────────────────────────────────────

def _build_ind_index(kb: KnowledgeBase):
    """Return (by_iri, by_local) dicts for IRI-namespace remapping.

    Some LP files use a different separator or namespace prefix from the KB
    (e.g. ``.../animals/trex01`` vs ``.../animals#trex01``).  We build two
    O(1) lookup tables so we can resolve individual IRIs regardless of which
    format the LP file uses.
    """
    by_iri:   Dict[str, OWLNamedIndividual] = {}
    by_local: Dict[str, OWLNamedIndividual] = {}
    for ind in kb.individuals():
        by_iri[str(ind.iri)] = ind
        by_local[ind.iri.get_remainder()] = ind
    return by_iri, by_local


def _resolve_ind(
    iri_str:  str,
    by_iri:   Dict[str, OWLNamedIndividual],
    by_local: Dict[str, OWLNamedIndividual],
) -> OWLNamedIndividual:
    """Return the KB-canonical individual, remapping namespace if needed."""
    if iri_str in by_iri:
        return by_iri[iri_str]
    local = IRI.create(iri_str).get_remainder()
    return by_local.get(local, OWLNamedIndividual(IRI.create(iri_str)))


def solve_lp(
    lp_name:         str,
    pos_iris:        List[str],
    neg_iris:        List[str],
    kb:              KnowledgeBase,
    sparql_endpoint: str,
    beam_width:      int   = 5,
    max_depth:       int   = 15,
    time_limit:      float = 120.0,
    top_k_record:    int   = 5,
) -> Dict:
    """
    Run the recording beam search on one LP and return the result dict
    ready to be inserted into the dataset JSON.
    """
    by_iri, by_local = _build_ind_index(kb)
    pos = frozenset(_resolve_ind(i, by_iri, by_local) for i in pos_iris)
    neg = frozenset(_resolve_ind(i, by_iri, by_local) for i in neg_iris)

    operator = PruneCELBasedRefinement(
        knowledge_base=kb,
        sparql_endpoint=sparql_endpoint,
    )
    operator.precision_threshold = 1.0
    operator.recall_threshold    = 0.6
    # During dataset generation we want ALL refinements recorded (the V-net
    # learns from both good and bad candidates).  The inference-time score
    # filter (tp<=1 guard + parent-score comparison) would silently discard
    # every candidate in sparse KBs where each class has only 1 individual.
    operator.skip_score_filter   = True
    operator.set_input_examples(pos, neg)

    learner = RecordingBeamSearchLearner(
        kb=kb,
        operator=operator,
        beam_width=beam_width,
        max_depth=max_depth,
        time_limit=time_limit,
        top_k_record=top_k_record,
    )

    t0   = time.time()
    best = learner.record_beam_search(set(pos), set(neg))
    elapsed = time.time() - t0

    renderer = DLSyntaxObjectRenderer()
    if best is not None:
        best_str  = renderer.render(best)
        # best_iris = learner._eval_with_instances(best, set(pos), set(neg))
        # recompute F1 from the full pos/neg for the summary
        best_instances = set(learner.kb.individuals_set(best))
        _tp = len(best_instances & set(pos))
        _fp = len(best_instances & set(neg))
        _fn = len(set(pos) - best_instances)
        _p  = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _r  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        best_f1 = 2 * _p * _r / (_p + _r) if (_p + _r) > 0 else 0.0
    else:
        # No refinement improved over root ⊤ — fall back to reporting ⊤ with its
        # actual F1 (the first recorded node is always the root node).
        best_str = renderer.render(OWLThing)
        root_node = learner.recorded_nodes[0] if learner.recorded_nodes else None
        best_f1   = root_node['f1'] if root_node else 0.0

    print(f"  → {len(learner.recorded_nodes)} nodes recorded, "
          f"best_f1={best_f1:.4f}, elapsed={elapsed:.1f}s")

    # Store KB-canonical IRI strings (after namespace remapping) so downstream
    # consumers (train_vocell_v_net.py) can look them up in the embeddings index
    # without needing to redo the namespace remapping themselves.
    canonical_pos = sorted(ind.iri.as_str() for ind in pos)
    canonical_neg = sorted(ind.iri.as_str() for ind in neg)

    return {
        "lp_name":           lp_name,
        "positive_examples": canonical_pos,
        "negative_examples": canonical_neg,
        "best_concept_str":  best_str,
        "best_f1":           round(best_f1, 6),
        "elapsed_seconds":   round(elapsed, 2),
        "nodes":             learner.recorded_nodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate V-Net training dataset by running beam search on a set of LPs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--lp_file', default='LPs/Family/lps.json',
        help='Path to the LP JSON file (must have a top-level "problems" key).',
    )
    parser.add_argument(
        '--kb', default='KGs/Family/family.owl',
        dest='kb_path',
        help='Path to the OWL knowledge base.',
    )
    parser.add_argument(
        '--sparql', default='http://localhost:3030/family/sparql',
        dest='sparql',
        help='SPARQL endpoint URL.',
    )
    parser.add_argument(
        '--output', default='vnet_search_data_lps.json',
        dest='output_file',
        help='Output JSON file path.',
    )
    parser.add_argument(
        '--beam_width', type=int, default=5,
        help='Beam width for search.',
    )
    parser.add_argument(
        '--top_k_record', type=int, default=25,
        help='How many refinements to record per depth step.',
    )
    parser.add_argument(
        '--max_depth', type=int, default=15,
        help='Maximum search depth.',
    )
    parser.add_argument(
        '--time_limit', type=float, default=120.0,
        help='Time budget in seconds per LP.',
    )
    parser.add_argument(
        '--lps', nargs='+', default=None,
        metavar='LP_NAME',
        help='Only process these LP names (default: all LPs in the file).',
    )
    parser.add_argument(
        '--num_lps', type=int, default=0,
        help='Number of LPs to process (0 = all in file).',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Re-run LPs that are already present in the output file.',
    )
    args = parser.parse_args()

    print("=" * 60)
    print("V-Net Dataset Generator")
    print(f"  LP file      : {args.lp_file}")
    print(f"  KB           : {args.kb_path}")
    print(f"  SPARQL       : {args.sparql}")
    print(f"  Output       : {args.output_file}")
    print(f"  beam_width   : {args.beam_width}")
    print(f"  top_k_record : {args.top_k_record}")
    print(f"  max_depth    : {args.max_depth}")
    print(f"  time_limit   : {args.time_limit}s per LP")
    if args.lps:
        print(f"  filter LPs   : {args.lps}")
    if args.overwrite:
        print("  overwrite    : ON")
    print("=" * 60)

    # ── Load KB (shared across all LPs) ───────────────────────────────────────
    print("\nLoading KB...")
    kb = KnowledgeBase(path=args.kb_path)

    # ── Load LP definitions ───────────────────────────────────────────────────
    with open(args.lp_file) as f:
        lps_data = json.load(f)


    lps_data = lps_data.get('problems', lps_data) if isinstance(lps_data, dict) else {}
    num_samples = min(args.num_lps, len(lps_data)) if hasattr(args, 'num_lps') else len(lps_data)
    lps_data = random.sample(lps_data, num_samples)
    # lps_data = list(lps_data.items())

    # Apply optional LP filter
    if args.lps:
        unknown = set(args.lps) - set(lps_data)
        if unknown:
            raise ValueError(f"Unknown LP name(s): {unknown}. "
                             f"Available: {list(lps_data.keys())}")
        lps_data = {k: lps_data[k] for k in args.lps}

    lp_names = list(lps_data.keys())
    print(f"Found {len(lp_names)} LP(s) to process: {lp_names}")

    # ── Resume from partial results if they exist ─────────────────────────────
    try:
        with open(args.output_file) as f:
            dataset = json.load(f)
        already_done = [n for n in lp_names if n in dataset]
        if already_done and not args.overwrite:
            print(f"\nResuming — {len(already_done)} LP(s) already done: "
                  f"{already_done}")
    except FileNotFoundError:
        dataset = {}

    # ── Solve each LP ─────────────────────────────────────────────────────────
    for lp_name, lp in lps_data.items():
        if lp_name in dataset and not args.overwrite:
            print(f"\n[skip] {lp_name} — already in dataset")
            continue

        print(f"\n{'─' * 60}")
        print(f"LP: {lp_name}  "
              f"(pos={len(lp['positive_examples'])}, "
              f"neg={len(lp['negative_examples'])})")
        print(f"{'─' * 60}")

        result = solve_lp(
            lp_name         = lp_name,
            pos_iris        = lp['positive_examples'],
            neg_iris        = lp['negative_examples'],
            kb              = kb,
            sparql_endpoint = args.sparql,
            beam_width      = args.beam_width,
            max_depth       = args.max_depth,
            time_limit      = args.time_limit,
            top_k_record    = args.top_k_record,
        )
        dataset[lp_name] = result

        # Save after every LP — never lose progress if the run is interrupted
        with open(args.output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"  ✓  Saved → {args.output_file} "
              f"({len(dataset)}/{len(lp_names)} done)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Dataset complete!  {len(dataset)} LPs  →  {args.output_file}")
    print("\nSummary:")
    total_nodes = 0
    for name in lp_names:
        if name not in dataset:
            continue
        d = dataset[name]
        n = len(d['nodes'])
        total_nodes += n
        print(f"  {name:25s}: {n:4d} nodes,  "
              f"best_f1={d['best_f1']:.4f}  ({d['elapsed_seconds']:.1f}s)")
    print(f"\n  Total nodes: {total_nodes}")
    print("=" * 60)


if __name__ == '__main__':
    main()

# Carcinogenesis: python generate_vnet_dataset.py --lp_file LPs/Carcinogenesis/lps.json --output vnet_search_data_carcinogenesis.json --beam_width 10 --time_limit 180  --sparql http://localhost:3030/carcinogenesis/sparql --kb KGs/Carcinogenesis/carcinogenesis.owl --num_lps 3
# Mutagenesis: python generate_vnet_dataset.py --lp_file LPs/Mutagenesis/lps.json --output vnet_search_data_mutagenesis.json --beam_width 10 --time_limit 180  --sparql http://localhost:3030/mutagenesis/sparql --kb KGs/Mutagenesis/mutagenesis.owl
# Animals: python generate_vnet_dataset.py --lp_file ../Ontolearn_ISWC/datasets/animals/training_data/training_data_prep.json --kb  ../Ontolearn_ISWC/datasets/animals/kb/ontology.owl --output vnet_search_data_animals.json --beam_width 5 --time_limit 60 --sparql http://localhost:3030/animals/sparql