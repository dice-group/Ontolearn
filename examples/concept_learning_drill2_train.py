"""
====================================================================
Concept Expression Learning — Multi-Learner Comparison

Supported learners (pass any subset to --learners):
  drill         Drill (DQN-based)
  drillv        DrillV (V-learning variant, see --drill_variant)
  ocel          OCEL
  celoe         CELOE
  tdl           TDL (tree-based)
  evolearner    EvoLearner (genetic)
  alcsat        ALCSAT (SAT-based)
  spell         SPELL (SAT-based)
  nero          NERO (neural set-transformer)
  prunecel      PruneCEL-S (Java, needs --prunecel_jar + --prunecel_sparql_url)
  vocell        VOCELL (beam search + optional V-net; needs --vocell_* args)

Examples
--------
# Only EvoLearner, VOCELL, ALCSAT, SPELL, NERO and PruneCEL
python examples/concept_learning_drill2_train.py \\
    --learners evolearner vocell alcsat spell nero prunecel \\
    --path_knowledge_base KGs/Family/family.owl \\
    --path_learning_problem LPs/Family/lps.json \\
    --prunecel_jar PruneCEL/target/prune-cel-0.0.1-SNAPSHOT.jar \\
    --prunecel_sparql_url http://localhost:3030/family/sparql \\
    --vocell_sparql http://localhost:3030/family/sparql \\
    --vocell_agg_type mean --vocell_strategy loocv

# All learners
python examples/concept_learning_drill2_train.py --learners all

Learn Embeddings (needed for Drill / DrillV / VOCELL):
  dicee --path_single_kg KGs/Family/family-benchmark_rich_background.owl \\
        --path_to_store_single_run embeddings --backend rdflib \\
        --save_embeddings_as_csv --model Keci --num_epoch 10
====================================================================
"""
import json
import sys
import os
import time
from argparse import ArgumentParser
from datetime import datetime
import random
from pathlib import Path

# Insert workspace root BEFORE any ontolearn imports so the local workspace
# copy is always used instead of the site-packages version (which may be older
# and lack classes like DrillVHeuristic that were added locally).
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import numpy as np
import pandas as pd
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learners import Drill, OCEL, CELOE, TDL
from ontolearn.learners import EvoLearner, ALCSAT, SPELL, NERO
from ontolearn.metrics import F1
from ontolearn.heuristics import CeloeBasedReward

from drillv_variants import DrillV_Minimal, DrillV_Standard, DrillV_Enhanced, DrillV_Complex
from prunecel_wrapper import PruneCELWrapper, check_prunecel_available

# VOCELL lives one level up from examples/.
# vocell.py itself now loads PruneCELBasedRefinement from the local workspace
# copy, so a plain import is enough here.
sys.path.insert(0, '..')
try:
    from vocell import _run_learner as _vocell_run, PruneCELBasedRefinement
    _VOCELL_AVAILABLE = True
except Exception as e:
    print(f"  ⚠ VOCELL import failed: {e}")
    _VOCELL_AVAILABLE = False
    PruneCELBasedRefinement = None

# ── All known learner names ───────────────────────────────────────────────────
ALL_LEARNERS = [
    'drill', 'drillv', 'ocel', 'celoe', 'tdl',
    'evolearner', 'alcsat', 'spell', 'nero',
    'prunecel', 'vocell',
]


# ── Experiment tracker ────────────────────────────────────────────────────────

class ExperimentTracker:
    """Collect per-run results and export to CSV."""

    def __init__(self):
        self.results = []

    def add(self, method, problem, run, train_time, inference_time,
            f1, concepts_tested, prediction):
        self.results.append({
            'method': method,
            'problem': problem,
            'run': run,
            'train_time': train_time,
            'inference_time': inference_time,
            'total_time': train_time + inference_time,
            'f1': f1,
            'concepts_tested': concepts_tested,
            'prediction': prediction,
        })

    def save(self, filename):
        df = pd.DataFrame(self.results)
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\n{'=' * 80}")
        print(f"✓ Results saved to: {path}")
        print(f"{'=' * 80}")
        return df



# ── Timing helpers ────────────────────────────────────────────────────────────

def _safe_kb_individuals(kb, pred):
    """Return the set of KB individuals that are instances of *pred*.

    Handles the case where the learner returns a class expression whose named
    class IRIs use a different namespace from the one loaded in the KB (e.g.
    ALCSAT on nctrer returns ``http://dl-learner.org/ont/Molecule#Foo`` while
    the KB stores the same class as ``http://dl-learner.org/ont/Foo``).
    We attempt the lookup as-is first, and on failure we rebuild a local-name
    → canonical-class index and substitute namespaces before retrying.
    """
    from owlapy.class_expression import OWLClass
    try:
        return kb.individuals(pred)
    except (AttributeError, TypeError):
        pass
    # Build class IRI remapping: local name → canonical OWLClass in KB
    cls_by_local: dict = {}
    for c in kb.ontology.classes_in_signature():
        cls_by_local[c.iri.get_remainder()] = c

    def _remap_ce(ce):
        """Recursively substitute mismatched named-class IRIs."""
        if isinstance(ce, OWLClass):
            local = ce.iri.get_remainder()
            return cls_by_local.get(local, ce)
        # For complex expressions, replace operands/filler/property recursively
        import owlapy.class_expression as _ce
        if hasattr(ce, 'operands'):          # OWLNaryBooleanClassExpression
            new_ops = [_remap_ce(o) for o in ce.operands()]
            return type(ce)(new_ops)
        if hasattr(ce, 'get_filler'):        # OWLQuantifiedObjectRestriction
            new_filler = _remap_ce(ce.get_filler())
            return type(ce)(ce.get_property(), new_filler)
        if hasattr(ce, 'get_operand'):       # OWLObjectComplementOf
            return type(ce)(_remap_ce(ce.get_operand()))
        return ce

    try:
        return kb.individuals(_remap_ce(pred))
    except Exception:
        return frozenset()


def _time_train(learner, train_args, directory):
    t0 = time.time()
    learner.train(**train_args)
    t = time.time() - t0
    learner.save(directory=directory)
    return t


def _time_predict(learner, lp, kb):
    dl_render = DLSyntaxObjectRenderer()
    t0 = time.time()
    if hasattr(learner, 'best_hypothesis'):
        pred = learner.fit(lp).best_hypothesis()
    else:
        pred = learner.fit(lp).best_hypotheses()
    elapsed = time.time() - t0

    # ── Derive a meaningful concept-exploration count per learner type ────────
    if isinstance(learner, PruneCELWrapper):
        # PruneCEL tracks this directly
        elapsed = learner.last_runtime
        concepts_tested = learner.number_of_tested_concepts

    elif hasattr(learner, 'search_tree') and hasattr(learner, 'target_class_expressions'):
        # NERO: neural model scores `target_class_expressions` candidates;
        # symbolic search then explores `search_tree.gate` of those further.
        neural_candidates = len(learner.target_class_expressions) \
            if learner.target_class_expressions else 0
        symbolic_explored = len(learner.search_tree) \
            if learner.search_tree else 0
        # Report total unique concepts considered (neural + symbolic refinements)
        concepts_tested = neural_candidates + symbolic_explored

    elif hasattr(learner, 'max_concept_size') and not hasattr(learner, 'search_tree'):
        # ALCSAT: SAT-based, no concept-by-concept search.
        # Report None so the caller can display N/A.
        concepts_tested = None

    elif hasattr(learner, 'max_query_size'):
        # SPELL: query-based SAT solver, same situation.
        concepts_tested = None

    else:
        # Search-based learners (Drill, DrillV, OCEL, CELOE, TDL, EvoLearner)
        concepts_tested = getattr(learner, '_number_of_tested_concepts', 0)

    f1 = compute_f1_score(
        individuals=frozenset(_safe_kb_individuals(kb, pred)),
        pos=lp.pos, neg=lp.neg,
    )
    return {
        'prediction': dl_render.render(pred),
        'f1': f1,
        'prediction_time': elapsed,
        'concepts_tested': concepts_tested,  # int or None (N/A for SAT-based)
    }


def _print_result(name, r):
    ct = r['concepts_tested']
    ct_str = 'N/A (SAT-based)' if ct is None else str(ct)
    print(f"  {name}:")
    print(f"    Prediction:      {r['prediction']}")
    print(f"    F1:              {r['f1']:.3f}")
    print(f"    Concepts tested: {ct_str}")
    print(f"    Time:            {r['prediction_time']:.2f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def start(args):
    # ── Resolve learner set ──────────────────────────────────────────────────
    if args.learners == ['all']:
        active = set(ALL_LEARNERS)
    else:
        active = set(l.lower() for l in args.learners)
        unknown = active - set(ALL_LEARNERS)
        if unknown:
            raise ValueError(f"Unknown learner(s): {unknown}. Choose from {ALL_LEARNERS}")

    print(f"\n{'=' * 80}")
    print(f"Active learners: {sorted(active)}")
    print(f"{'=' * 80}\n")

    # ── Knowledge base ───────────────────────────────────────────────────────
    kb = KnowledgeBase(path=args.path_knowledge_base)
    # Build two indexes to handle IRI namespace mismatches between LP files
    # and the KB (e.g. http://dl-learner.org/ont/ vs http://dl-learner.org/nctrer/).
    _kb_ind_by_iri: dict = {}    # full IRI string  → OWLNamedIndividual
    _kb_ind_by_local: dict = {}  # local name (remainder) → OWLNamedIndividual
    for _ind in kb.individuals():
        _kb_ind_by_iri[str(_ind.iri)] = _ind
        _kb_ind_by_local[_ind.iri.get_remainder()] = _ind

    def _resolve_ind(iri_str: str) -> OWLNamedIndividual:
        """Return the KB-canonical individual, remapping namespace if needed."""
        if iri_str in _kb_ind_by_iri:               # O(1) — IRI already matches KB
            return _kb_ind_by_iri[iri_str]
        local = IRI.create(iri_str).get_remainder()
        return _kb_ind_by_local.get(               # O(1) — match by local name
            local, OWLNamedIndividual(IRI.create(iri_str))  # fallback: keep original
        )

    train_args = dict(
        num_of_target_concepts=args.num_of_target_concepts,
        num_learning_problems=args.num_of_training_learning_problems,
    )
    tracker = ExperimentTracker() if args.save_results else None

    # ── Initialize learners ──────────────────────────────────────────────────
    learners = {}  # name → instance

    # --- Drill ---------------------------------------------------------------
    if 'drill' in active:
        learners['drill'] = Drill(
            knowledge_base=kb,
            path_embeddings=args.path_embeddings,
            refinement_operator=LengthBasedRefinement(knowledge_base=kb),
            quality_func=F1(),
            reward_func=CeloeBasedReward(),
            epsilon_decay=args.epsilon_decay,
            learning_rate=args.learning_rate,
            verbose=0,
            num_of_sequential_actions=args.num_of_sequential_actions,
            num_episode=args.num_episode,
            iter_bound=args.iter_bound,
            max_runtime=args.max_runtime,
        )
        print("  ✓ Drill initialized")

    # --- DrillV --------------------------------------------------------------
    if 'drillv' in active:
        variant_map = {
            'minimal': DrillV_Minimal,
            'standard': DrillV_Standard,
            'enhanced': DrillV_Enhanced,
            'complex': DrillV_Complex,
        }
        DrillVClass = variant_map.get(args.drill_variant, DrillV_Complex)
        learners['drillv'] = DrillVClass(
            knowledge_base=kb,
            path_embeddings=args.path_embeddings,
            refinement_operator=LengthBasedRefinement(knowledge_base=kb),
            quality_func=F1(),
            reward_func=CeloeBasedReward(),
            epsilon_decay=args.epsilon_decay,
            learning_rate=args.learning_rate,
            verbose=0,
            num_of_sequential_actions=args.num_of_sequential_actions,
            num_episode=args.num_episode,
            iter_bound=args.iter_bound,
            max_runtime=args.max_runtime,
        )
        print(f"  ✓ DrillV ({args.drill_variant}) initialized")

    # --- OCEL ----------------------------------------------------------------
    if 'ocel' in active:
        learners['ocel'] = OCEL(knowledge_base=kb, quality_func=F1(),
                                max_runtime=args.max_runtime)
        print("✓ OCEL initialized")

    # --- CELOE ---------------------------------------------------------------
    if 'celoe' in active:
        learners['celoe'] = CELOE(knowledge_base=kb, quality_func=F1(),
                                  max_runtime=args.max_runtime)
        print("✓ CELOE initialized")

    # --- TDL -----------------------------------------------------------------
    if 'tdl' in active:
        learners['tdl'] = TDL(
            knowledge_base=kb,
            use_nominals=False,
            kwargs_classifier={'random_state': args.random_seed},
            max_runtime=args.max_runtime,
        )
        print("✓ TDL initialized")

    # --- EvoLearner ----------------------------------------------------------
    if 'evolearner' in active:
        learners['evolearner'] = EvoLearner(
            knowledge_base=kb,
            use_card_restrictions=False,
            use_data_properties=False,
            max_runtime=args.max_runtime,
        )
        print("  ✓ EvoLearner initialized")

    # --- ALCSAT --------------------------------------------------------------
    if 'alcsat' in active:
        learners['alcsat'] = ALCSAT(
            knowledge_base=kb,
            max_runtime=args.max_runtime,
            max_concept_size=30,
        )
        print("  ✓ ALCSAT initialized")

    # --- SPELL ---------------------------------------------------------------
    if 'spell' in active:
        learners['spell'] = SPELL(
            knowledge_base=kb,
            max_runtime=args.max_runtime,
            max_query_size=10,
            search_mode='full_approx',
        )
        print("  ✓ SPELL initialized")

    # --- NERO ----------------------------------------------------------------
    if 'nero' in active:
        learners['nero'] = NERO(
            knowledge_base=kb,
            num_embedding_dim=128,
            neural_architecture='DeepSet',
            learning_rate=0.001,
            num_epochs=50,
            batch_size=32,
        )
        print("  ✓ NERO initialized")

    # --- PruneCEL ------------------------------------------------------------
    if 'prunecel' in active:
        if not args.prunecel_jar or not args.prunecel_sparql_url:
            print("  ⚠ PruneCEL skipped — provide --prunecel_jar and --prunecel_sparql_url")
        elif check_prunecel_available(args.prunecel_jar):
            try:
                learners['prunecel'] = PruneCELWrapper(
                    jar_path=args.prunecel_jar,
                    sparql_url=args.prunecel_sparql_url,
                    knowledge_base=kb,
                    max_runtime=args.max_runtime,
                    recursive=args.prunecel_recursive,
                    skip_none=args.prunecel_skip_none,
                )
                print("  ✓ PruneCEL initialized")
            except Exception as e:
                print(f"  ⚠ PruneCEL init failed: {e}")
        else:
            print("  ⚠ PruneCEL JAR not found — run ./setup_prunecel.sh")

    # --- VOCELL (constructed fresh per-problem at inference time) ------------
    run_vocell = 'vocell' in active
    if run_vocell and not _VOCELL_AVAILABLE:
        print("  ⚠ VOCELL not importable (check vocell.py is one level up)")
        run_vocell = False
    elif run_vocell:
        print("  ✓ VOCELL will be constructed per-problem at inference time")

    # # ── Train Drill / DrillV -------------------------------------------------
    # if args.path_pretrained_dir:
    #     if 'drill' in learners:
    #         learners['drill'].load(directory='pretrained_drill')
    #         print("Loaded pretrained Drill")
    #     if 'drillv' in learners:
    #         learners['drillv'].load(directory='pretrained_drillv')
    #         print("Loaded pretrained DrillV")
    # else:
    #     if 'drill' in learners:
    #         print("Training Drill…")
    #         _time_train(learners['drill'], train_args, 'pretrained_drill')
    #     if 'drillv' in learners:
    #         print("Training DrillV…")
    #         _time_train(learners['drillv'], train_args, 'pretrained_drillv')

    # ── Load LP file ---------------------------------------------------------
    with open(args.path_learning_problem, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Support both {"problems": {...}} and flat {"LP_name": {...}, ...} formats.
    problems = data.get('problems', data) if isinstance(data, dict) else {}

    problem_items = random.sample(list(problems.items()), len(problems))

    if args.num_problems > 0:
        problem_items = problem_items[:args.num_problems]
    print(f"\nEvaluating on {len(problem_items)} problem(s), {args.num_runs} run(s).\n")

    # ── Per-learner accumulators ─────────────────────────────────────────────
    stats: dict = {
        name: {'times': [], 'f1s': [], 'concepts': []}
        for name in list(learners.keys()) + (['vocell'] if run_vocell else [])
    }

    # ── Evaluation loop ──────────────────────────────────────────────────────
    for run_idx in range(args.num_runs):
        for str_target_concept, examples in problem_items:
            pos_iris = examples['positive_examples']
            neg_iris = examples['negative_examples']
            typed_pos = set(_resolve_ind(i) for i in pos_iris)
            typed_neg = set(_resolve_ind(i) for i in neg_iris)
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            print(f"\n{'=' * 80}")
            print(f"  LP: {str_target_concept}   run {run_idx + 1}/{args.num_runs}")
            print(f"  pos={len(typed_pos)}  neg={len(typed_neg)}")
            print(f"{'=' * 80}")

            # ── Standard learners ─────────────────────────────────────────
            for name, learner in learners.items():
                # NERO needs a namespace hint
                if name == 'nero':
                    try:
                        a_prop = next(iter(kb.ontology.object_properties_in_signature()))
                        learner.ns = a_prop.iri.get_namespace()
                    except StopIteration:
                        pass

                try:
                    r = _time_predict(learner, lp, kb)
                except Exception as e:
                    print(f"  [!] {name.upper()} failed: {e}")
                    continue

                stats[name]['times'].append(r['prediction_time'])
                stats[name]['f1s'].append(r['f1'])
                stats[name]['concepts'].append(r['concepts_tested'])
                _print_result(name.upper(), r)

                if tracker:
                    tracker.add(
                        method=name,
                        problem=str_target_concept,
                        run=run_idx + 1,
                        train_time=0,
                        inference_time=r['prediction_time'],
                        f1=r['f1'],
                        concepts_tested=r['concepts_tested'],
                        prediction=r['prediction'],
                    )

            # ── VOCELL ────────────────────────────────────────────────────
            if run_vocell:
                try:
                    vocell_op = PruneCELBasedRefinement(
                        knowledge_base=kb,
                        sparql_endpoint=args.vocell_sparql,
                    )
                    vocell_op.precision_threshold = args.vocell_precision_threshold
                    vocell_op.recall_threshold    = args.vocell_recall_threshold
                    vocell_op.set_input_examples(frozenset(typed_pos), frozenset(typed_neg))

                    v_net_path = None
                    if args.vocell_agg_type and args.vocell_ckpt_dir:
                        lp_key = str_target_concept.replace(' ', '_')
                        if args.vocell_strategy == 'loocv':
                            v_net_path = str(
                                Path(args.vocell_ckpt_dir) /
                                f'vocell_v_net_{lp_key}_{args.vocell_agg_type}.pt'
                            )
                        else:
                            v_net_path = str(
                                Path(args.vocell_ckpt_dir) /
                                f'vocell_v_net_bootstrap_{args.vocell_agg_type}.pt'
                            )
                        if not Path(v_net_path).exists():
                            print(f"  [!] VOCELL checkpoint not found: {v_net_path} "
                                  f"— running without V-net")
                            v_net_path = None

                    use_vl = v_net_path is not None
                    t0 = time.time()
                    concept_str, f1, concepts_tested = _vocell_run(
                        kb=kb,
                        operator=vocell_op,
                        pos=frozenset(typed_pos),
                        neg=frozenset(typed_neg),
                        use_v_learning=use_vl,
                        time_limit=args.max_runtime,
                        beam_width=args.vocell_beam_width,
                        max_depth=args.vocell_max_depth,
                        path_embeddings=args.vocell_embeddings if use_vl else None,
                        v_net_path=v_net_path,
                        verbose=args.verbose,
                    )
                    elapsed = time.time() - t0

                    r_v = {'prediction': concept_str, 'f1': f1,
                           'prediction_time': elapsed, 'concepts_tested': concepts_tested}
                    stats['vocell']['times'].append(elapsed)
                    stats['vocell']['f1s'].append(f1)
                    stats['vocell']['concepts'].append(concepts_tested)
                    _print_result('VOCELL', r_v)

                    if tracker:
                        tracker.add(
                            method='vocell',
                            problem=str_target_concept,
                            run=run_idx + 1,
                            train_time=0,
                            inference_time=elapsed,
                            f1=f1,
                            concepts_tested=concepts_tested,
                            prediction=concept_str,
                        )
                except Exception as e:
                    print(f"  [!] VOCELL failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Learner':<18} {'Avg F1':>8}  {'Avg Time (s)':>14}  {'Avg Concepts':>14}")
    print(f"  {'-' * 18} {'-' * 8}  {'-' * 14}  {'-' * 14}")
    for name, s in stats.items():
        if not s['f1s']:
            continue
        # concepts list may contain None (SAT-based learners) — filter before averaging
        valid_concepts = [c for c in s['concepts'] if c is not None]
        if valid_concepts:
            avg_concepts_str = f"{np.mean(valid_concepts):>14.0f}"
        else:
            avg_concepts_str = f"{'N/A (SAT)':>14}"
        print(f"  {name.upper():<18} {np.mean(s['f1s']):>8.3f}  "
              f"{np.mean(s['times']):>14.3f}  {avg_concepts_str}")
    print(f"{'=' * 80}")

    if tracker and args.save_results:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        names_tag = '_'.join(sorted(
            list(learners.keys()) + (['vocell'] if run_vocell else [])
        ))
        csv_file = f"results/comparison_{names_tag}_{ts}.csv"
        tracker.save(csv_file)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Multi-learner concept expression learning comparison.',
        epilog=f'Available learners: {ALL_LEARNERS}',
    )

    # ── Learner selection ─────────────────────────────────────────────────────
    parser.add_argument(
        '--learners', nargs='+',
        default=['drill', 'drillv', 'ocel', 'celoe'],
        choices=['drill', 'drillv', 'ocel', 'celoe', 'tdl',
    'evolearner', 'alcsat', 'spell', 'nero',
    'prunecel', 'vocell',] + ['all'],
        metavar='LEARNER',
        help=(f'Learners to run (space-separated). Use "all" for every learner. '
              f'Available: {ALL_LEARNERS}'),
    )

    # ── KB & problems ─────────────────────────────────────────────────────────
    parser.add_argument('--path_knowledge_base', type=str,
                        default='KGs/Family/family.owl')
    parser.add_argument('--path_learning_problem', type=str,
                        default='LPs/Family/lps_difficult.json')
    parser.add_argument('--num_problems', type=int, default=0,
                        help='Number of LPs to evaluate (0 = all).')
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--max_runtime', type=int, default=30,
                        help='Per-learner time budget in seconds.')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--save_results', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=False)

    # ── Drill / DrillV ────────────────────────────────────────────────────────
    parser.add_argument('--path_embeddings', type=str,
                        default='Experiments/embeddings/Keci_entity_embeddings.csv')
    parser.add_argument('--path_pretrained_dir', type=str, default=None)
    parser.add_argument('--num_of_target_concepts', type=int, default=1)
    parser.add_argument('--num_of_training_learning_problems', type=int, default=20)
    parser.add_argument('--drill_variant', type=str, default='complex',
                        choices=['default', 'minimal', 'standard', 'enhanced', 'complex'])
    parser.add_argument('--num_episode', type=int, default=1)
    parser.add_argument('--epsilon_decay', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--iter_bound', type=int, default=10_000)
    parser.add_argument('--num_of_sequential_actions', type=int, default=1)

    # ── PruneCEL ──────────────────────────────────────────────────────────────
    parser.add_argument('--prunecel_jar', type=str,
                        default='PruneCEL/target/prune-cel-0.0.1-SNAPSHOT.jar')
    parser.add_argument('--prunecel_sparql_url', type=str,
                        default='http://localhost:3030/family/sparql')
    parser.add_argument('--prunecel_recursive', action='store_true', default=False)
    parser.add_argument('--prunecel_skip_none', action='store_true', default=True)

    # ── VOCELL ────────────────────────────────────────────────────────────────
    parser.add_argument('--vocell_sparql', type=str,
                        default='http://localhost:3030/family/sparql',
                        help="SPARQL endpoint for VOCELL's PruneCEL refinement operator.")
    parser.add_argument('--vocell_embeddings', type=str,
                        default='Experiments/embeddings/Keci_entity_embeddings.csv',
                        help='Entity embeddings CSV used by the VOCELL V-net.')
    parser.add_argument('--vocell_beam_width', type=int, default=5)
    parser.add_argument('--vocell_max_depth', type=int, default=15)
    parser.add_argument('--vocell_precision_threshold', type=float, default=1.0)
    parser.add_argument('--vocell_recall_threshold', type=float, default=0.4)
    parser.add_argument('--vocell_agg_type', type=str, default='mean',
                        choices=['mean', 'deepsets', 'settransformer'],
                        help='V-net aggregation type. Omit to run VOCELL without a V-net.')
    parser.add_argument('--vocell_strategy', type=str, default='loocv',
                        choices=['loocv', 'bootstrap'],
                        help='Training strategy used to produce the checkpoint.')
    parser.add_argument('--vocell_ckpt_dir', type=str, default=None,
                        help='Directory containing VOCELL checkpoints '
                             '(default: Family_{vocell_agg_type}).')

    args = parser.parse_args()

    # Default checkpoint dir for VOCELL
    if args.vocell_agg_type and args.vocell_ckpt_dir is None:
        args.vocell_ckpt_dir = f'Family_{args.vocell_agg_type}'

    start(args)

#  python examples/concept_learning_drill2_train.py --learners vocell evolearner alcsat spell nero prunecel --vocell_agg_type mean --vocell_strategy loocv --max_runtime 60 --num_problems 0
#  python examples/concept_learning_drill2_train.py --path_learning_problem LPs/Mutagenesis/lps.json --vocell_ckpt_dir Mutagegenesis_mean --vocell_embeddings ../Ontolearn_ISWC/datasets/mutagenesis/embeddings/DeCaL_entity_embeddings.csv --vocell_strategy bootstrap --num_problems 0 --prunecel_sparql http://localhost:3030/mutagenesis/sparql --vocell_sparql http://localhost:3030/mutagenesis/sparql --path_knowledge_base KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --learners vocell evolearner alcsat spell nero prunecel
#  animals: python examples/concept_learning_drill2_train.py --path_learning_problem ../Ontolearn_ISWC/datasets/animals/training_data/training_data_prep.json --num_problems 2 --path_knowledge_base ../Ontolearn_ISWC/datasets/animals/kb/ontology.owl --max_runtime 60 --learners ocel celoe evolearner prunecel alcsat spell nero vocell --vocell_ckpt_dir animals_mean   --vocell_embeddings ../Ontolearn_ISWC/datasets/animals/embeddings/DeCaL_entity_embeddings.csv --vocell_strategy bootstrap --prunecel_sparql http://localhost:3030/animals/sparql --vocell_sparql http://localhost:3030/animals/sparql
# nctrer: python examples/concept_learning_drill2_train.py --path_learning_problem ../Ontolearn_ISWC/datasets/nctrer/training_data/training_data_prep.json --num_problems 9 --path_knowledge_base ../Ontolearn_ISWC/datasets/nctrer/kb/ontology.owl --max_runtime 30 --learners drill drill2 --vocell_ckpt_dir nctrer_mean   --vocell_embeddings ../Ontolearn_ISWC/datasets/ncter/embeddings/DeCaL_entity_embeddings.csv --vocell_strategy bootstrap --prunecel_sparql http://localhost:3030/nctrer/sparql --vocell_sparql http://localhost:3030/nctrer/sparql --path_embeddings ../Ontolearn_ISWC/datasets/ncter/embeddings/DeCaL_entity_embeddings.csv
# python report_results.py --csv_file results/comparison_nctrer_alcsat_celoe_evolearner_nero_ocel_prunecel_spell_vocell_20260406_013116.csv --vocell_f1_threshold 0.9