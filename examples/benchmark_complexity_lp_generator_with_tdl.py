#!/usr/bin/env python3
"""Benchmark: how does TDL's F1 degrade as target-concept complexity grows?

This drives ``ontolearn.lp_generator.GoalDirectedLPGenerator``: instead of refining
``owl:Thing`` and keeping whatever non-redundant concepts fall out (what
``ontolearn.lp_generator.LPGen``/``KB2Data`` do), it samples target concepts top-down from an
explicit DL-construct grammar (:class:`~ontolearn.lp_generator.ComplexityProfile`) -- e.g. "only
atomic classes and negation", then "add intersection", then "add nested existential
restrictions", then "add cardinality restrictions", then "everything" -- producing a graded
curriculum of learning problems with known ground-truth targets and controlled length/depth.

For every generated learning problem we fit ``TDL`` (a decision-tree-based concept learner) and
score its learned hypothesis's extension against the target concept's own (E+, E-), so the F1
reported is "did TDL recover a concept with (close to) the same instances as the hidden target",
not TDL's usual train-set fit.

Usage
-----
    python examples/benchmark_complexity_lp_generator_with_tdl.py \\
        --kb KGs/Family/family-benchmark_rich_background.owl --num-lps 6 \\
        --benchmark-output family_complexity_lps.json --results-output family_complexity_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Dict, FrozenSet, List

from owlapy import owl_expression_to_dl
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL
from ontolearn.lp_generator import (
    ComplexityProfile,
    DLConstruct,
    GoalDirectedLPGenerator,
    save_benchmark,
)
from ontolearn.utils.static_funcs import concept_len

Individuals = FrozenSet[OWLNamedIndividual]


def f1(pos: Individuals, neg: Individuals, covered: Individuals) -> float:
    tp = len(pos & covered)
    if tp == 0:
        return 0.0
    fp = len(neg & covered)
    fn = len(pos - covered)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def default_curriculum() -> List[ComplexityProfile]:
    """A curriculum of six DL-construct budgets, roughly in increasing order of difficulty."""
    return [
        ComplexityProfile(
            name="1-atomic",
            allowed_constructs=frozenset({DLConstruct.NEGATION}),
            min_length=1, max_length=2, min_depth=0, max_depth=0,
            max_baseline_f1=1.01,  # atoms/negated-atoms ARE the baseline probe pool
        ),
        ComplexityProfile(
            name="2-conjunctive",
            allowed_constructs=frozenset({DLConstruct.INTERSECTION, DLConstruct.NEGATION}),
            min_length=3, max_length=6, min_depth=0, max_depth=0,
            max_baseline_f1=0.9,
        ),
        ComplexityProfile(
            name="3-existential",
            allowed_constructs=frozenset(
                {DLConstruct.EXISTENTIAL, DLConstruct.INTERSECTION, DLConstruct.NEGATION}),
            min_length=4, max_length=8, min_depth=1, max_depth=2,
            max_baseline_f1=0.9,
        ),
        ComplexityProfile(
            name="4-nested-existential",
            allowed_constructs=frozenset(
                {DLConstruct.EXISTENTIAL, DLConstruct.INTERSECTION, DLConstruct.NEGATION}),
            min_length=6, max_length=12, min_depth=2, max_depth=3,
            max_baseline_f1=0.85,
        ),
        ComplexityProfile(
            name="5-cardinality",
            allowed_constructs=frozenset(
                {DLConstruct.MIN_CARDINALITY, DLConstruct.MAX_CARDINALITY, DLConstruct.INTERSECTION}),
            min_length=4, max_length=10, min_depth=1, max_depth=2, card_limit=4,
            max_baseline_f1=0.85,
        ),
        ComplexityProfile(
            name="6-full-alcq",
            allowed_constructs=frozenset({
                DLConstruct.INTERSECTION, DLConstruct.UNION, DLConstruct.EXISTENTIAL,
                DLConstruct.UNIVERSAL, DLConstruct.MIN_CARDINALITY, DLConstruct.MAX_CARDINALITY,
                DLConstruct.NEGATION,
            }),
            min_length=8, max_length=16, min_depth=2, max_depth=4,
            max_baseline_f1=0.8,
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb", default="KGs/Family/family-benchmark_rich_background.owl")
    p.add_argument("--num-lps", type=int, default=6, help="learning problems generated PER profile")
    p.add_argument("--max-attempts", type=int, default=3000, help="sampling attempts allotted per profile")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--benchmark-output", default=None, help="optional path to dump the generated LPs as JSON")
    p.add_argument("--results-output", default=None, help="optional path to dump the per-LP TDL results as JSON")
    args = p.parse_args()

    kb = KnowledgeBase(path=args.kb)
    print(f"[i] knowledge base: {args.kb}  ({len(list(kb.individuals()))} individuals)")

    generator = GoalDirectedLPGenerator(kb, seed=args.seed)
    curriculum = default_curriculum()

    benchmark = generator.generate_benchmark(curriculum, num_lps_per_profile=args.num_lps,
                                              max_attempts_per_profile=args.max_attempts)
    for profile in curriculum:
        print(f"[i] profile '{profile.name}': generated {len(benchmark[profile.name])}/{args.num_lps} LPs")

    if args.benchmark_output:
        save_benchmark(benchmark, args.benchmark_output)
        print(f"[i] saved generated learning problems to {args.benchmark_output}")

    results: List[Dict] = []
    for profile in curriculum:
        for lp in benchmark[profile.name]:
            learning_problem = lp.to_lp()
            model = TDL(knowledge_base=kb, report_classification=False, verbose=0)
            t0 = time.time()
            model.fit(learning_problem=learning_problem)
            runtime = time.time() - t0
            learned = model.best_hypotheses()
            covered = kb.individuals_set(learned)
            score = f1(lp.pos, lp.neg, covered)

            results.append(dict(
                profile=profile.name,
                target_dl=lp.dl,
                target_length=lp.length,
                target_depth=lp.depth,
                baseline_f1=lp.baseline_f1,
                num_pos=len(lp.pos),
                num_neg=len(lp.neg),
                learned_dl=owl_expression_to_dl(learned),
                learned_length=concept_len(learned),
                tdl_f1=round(score, 4),
                runtime_sec=round(runtime, 3),
            ))
            print(f"  [{profile.name}] target={lp.dl}  (len={lp.length}, depth={lp.depth})  "
                  f"-> TDL F1={score:.3f}  learned={owl_expression_to_dl(learned)}")

    print("\n=== summary (mean TDL F1 by complexity profile) ===")
    print(f"{'profile':<24}{'#lps':>6}{'mean target len':>18}{'mean baseline F1':>18}{'mean TDL F1':>14}")
    for profile in curriculum:
        rows = [r for r in results if r["profile"] == profile.name]
        if not rows:
            print(f"{profile.name:<24}{0:>6}{'-':>18}{'-':>18}{'-':>14}")
            continue
        mean_len = statistics.mean(r["target_length"] for r in rows)
        mean_base = statistics.mean(r["baseline_f1"] for r in rows)
        mean_f1 = statistics.mean(r["tdl_f1"] for r in rows)
        print(f"{profile.name:<24}{len(rows):>6}{mean_len:>18.2f}{mean_base:>18.3f}{mean_f1:>14.3f}")

    if args.results_output:
        with open(args.results_output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\n[i] saved per-LP TDL results to {args.results_output}")


if __name__ == "__main__":
    main()
