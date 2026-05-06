"""
generate_vnet_dataset.py
========================
Runs the BeamSearchLearner (from test_covering_algorithm.py) on every LP in
LPs/Family/lps_difficult.json, records the beam-search tree at each depth
level, and writes the results to vnet_search_data_difficult.json.

This is the expensive one-time pre-computation step (needs SPARQL + reasoner).
The output JSON is then consumed by train_vocell_v_net.py for leave-one-LP-out
V-Net training — no further KB queries are needed at training time.

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
      depth         : int    — search depth (root ⊤ = 0)
      parent_str    : str|null — DL string of the parent concept
      instance_iris : list   — IRI strings of every individual in I(C)

Search tree structure
---------------------
  depth 0  : root ⊤
  depth 1  : top-K refinements of ⊤ (by F1)
  depth 2  : top-K refinements of EACH concept that was in the beam at depth 1
  ...
  The beam advances with the global top beam_width concepts at each depth,
  but we only record the top top_k_record (≤ beam_width) per depth step.

Run as:
  python generate_vnet_dataset.py
"""

import json
import time
from typing import Dict, List, Optional, Set, Tuple

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import PruneCELBasedRefinement
from owlapy.class_expression import OWLClassExpression, OWLThing
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer

from test_covering_algorithm import BeamSearchLearner


# ─────────────────────────────────────────────────────────────────────────────
#  RecordingBeamSearchLearner
# ─────────────────────────────────────────────────────────────────────────────

class RecordingBeamSearchLearner(BeamSearchLearner):
    """
    Subclass of BeamSearchLearner that records every node added to the beam
    during search.

    After calling record_beam_search(), results are in self.recorded_nodes —
    a list of dicts, one per recorded concept, with keys:
        concept_str, f1, depth, parent_str, instance_iris.

    Recording follows the user's description exactly:
        depth 0  — root ⊤
        depth d  — top top_k_record refinements of the current beam
                   (parent_str points to the beam concept that generated them)
    """

    def __init__(self, *args, top_k_record: int = 5, **kwargs):
        """
        Parameters
        ----------
        top_k_record : int
            How many refinements (ranked by F1) to record at each depth step.
            Must be ≤ beam_width.  These become the training data points.
        """
        super().__init__(*args, **kwargs)
        self.top_k_record   = top_k_record
        self.recorded_nodes: List[Dict] = []

    # ── internal helpers ──────────────────────────────────────────────────────

    def _eval_with_instances(
        self,
        concept: OWLClassExpression,
        pos: Set[OWLNamedIndividual],
        neg: Set[OWLNamedIndividual],
    ) -> List[str]:
        """
        Returns sorted instance IRIs for a concept.
        F1 is NOT computed here — it is already returned by refine().
        This is called only for the root node (⊤) and for the final best
        concept in solve_lp, where refine() is not used.
        """
        instances = self.kb.individuals_set(concept)
        iris = sorted(ind.str for ind in instances)   # sorted → deterministic
        return iris

    def _make_node(
        self,
        concept: OWLClassExpression,
        f1: float,
        iris: List[str],
        depth: int,
        parent_str: Optional[str]
    ) -> Dict:
        return {
            "concept_str":   self.renderer.render(concept),
            "f1":            round(f1, 6),
            "depth":         depth,
            "parent_str":    parent_str,
            "instance_iris": iris,   # stored so train step needs no KB queries
        }

    # ── main method ───────────────────────────────────────────────────────────

    def record_beam_search(
        self,
        pos: Set[OWLNamedIndividual],
        neg: Set[OWLNamedIndividual],
    ) -> Optional[OWLClassExpression]:
        """
        Beam search with recording.

        At every depth d the method:
          1. Generates all refinements of every concept currently in the beam.
          2. Evaluates each candidate — one KB call per candidate.
          3. Records the top top_k_record candidates as training data points,
             storing their concept_str, f1, depth, parent_str, instance_iris.
          4. Advances the beam with the global top beam_width candidates.

        Returns the best OWLClassExpression found (highest F1).
        """
        self.recorded_nodes = []

        # ── root ⊤ ────────────────────────────────────────────────────────────
        # refine() is not called on ⊤ itself, so we compute F1 manually.
        root_instances = self.kb.individuals_set(OWLThing)
        _tp = len(root_instances & pos)
        _fp = len(root_instances & neg)
        _fn = len(pos - root_instances)
        _p  = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _r  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        root_f1  = 2 * _p * _r / (_p + _r) if (_p + _r) > 0 else 0.0
        root_iris = sorted(ind.str for ind in root_instances)
        self.recorded_nodes.append(
            self._make_node(OWLThing, root_f1, root_iris, 0, None)
        )

        beam: List[Tuple[OWLClassExpression, str]] = [
            (OWLThing, self.renderer.render(OWLThing))
        ]
        best_concept: Optional[OWLClassExpression] = None
        best_f1   = root_f1
        visited: Set[str] = {self.renderer.render(OWLThing)}

        start = time.time()

        for depth in range(self.max_depth):
            if time.time() - start >= self.time_limit:
                print(f"  ⏱  Time limit reached at depth {depth}")
                break

            # ── generate & evaluate all refinements of current beam ────────
            all_candidates: List[Tuple[OWLClassExpression, str, float, str, List[str]]] = []

            for concept, parent_str in beam:
                try:
                    for child, f1, _, precision, recall, tp in self.operator.refine(concept):
                        child_str = self.renderer.render(child)
                        if child_str in visited:
                            continue
                        visited.add(child_str)

                        child_iris = self._eval_with_instances(child, pos, neg)
                        all_candidates.append(
                            (child, child_str, f1, parent_str, child_iris)
                        )

                        if f1 > best_f1:
                            best_f1      = f1
                            best_concept = child

                        if f1 == 1.0:
                            # Record and stop immediately
                            self.recorded_nodes.append(
                                self._make_node(child, f1, child_iris, depth + 1, parent_str)
                            )
                            print(f"  ✓  F1=1.0 at depth {depth + 1}: "
                                  f"{child_str[:60]}")
                            return child

                except Exception:
                    continue

            if not all_candidates:
                print(f"  No new refinements at depth {depth + 1}")
                break

            # ── sort, record top_k_record, advance beam ────────────────────
            all_candidates.sort(key=lambda x: x[2], reverse=True)

            best_candidates = all_candidates[:self.top_k_record]
            bad_candidates  = []  # all_candidates[-self.top_k_record:]

            # list (not set) — tuples contain OWLClassExpression objects that
            # may not be hashable
            to_record = best_candidates + bad_candidates
            for child, child_str, f1, parent_str, iris in to_record:
                self.recorded_nodes.append(
                    self._make_node(child, f1, iris, depth + 1, parent_str)
                )

            beam = [
                (child, child_str)
                for child, child_str, _, _, _ in all_candidates[:self.beam_width]
            ]

            top_f1s = [f"{c[2]:.3f}" for c in to_record]
            print(f"  depth {depth + 1:2d}: {len(all_candidates):4d} candidates"
                  f" → recorded top-{len(to_record)}: [{', '.join(top_f1s)}]"
                  f"  beam={len(beam)}  overall_best={best_f1:.4f}")

        return best_concept


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
        best_iris = learner._eval_with_instances(best, set(pos), set(neg))
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

# Carcinogenesis: python generate_vnet_dataset.py --lp_file LPs/Carcinogenesis/lps.json --output vnet_search_data_carcinogenesis.json --beam_width 10 --time_limit 180  --sparql http://localhost:3030/carcinogenesis/sparql --kb KGs/Carcinogenesis/carcinogenesis.owl
# Mutagenesis: python generate_vnet_dataset.py --lp_file LPs/Mutagenesis/lps.json --output vnet_search_data_mutagenesis.json --beam_width 10 --time_limit 180  --sparql http://localhost:3030/mutagenesis/sparql --kb KGs/Mutagenesis/mutagenesis.owl
# Animals: python generate_vnet_dataset.py --lp_file ../Ontolearn_ISWC/datasets/animals/training_data/training_data_prep.json --kb  ../Ontolearn_ISWC/datasets/animals/kb/ontology.owl --output vnet_search_data_animals.json --beam_width 5 --time_limit 60 --sparql http://localhost:3030/animals/sparql