#!/usr/bin/env python3
"""
Generate hard, numeric-biased Learning Problems with Ontolearn .

Why not `ontolearn.lp_generator.LPGen` directly?
  * LPGen/KB2Data writes  [[concept, {"positive examples": [...], ...}], ...]  (a LIST, keys with
    spaces, individuals truncated to their short name (without full IRI)).
  * `beyond_alc=True`  enables numeric datatypes in ExpressRefinement; it does not bias
    targets towards them, does not enforce a minimum example count, and does not filter for hardness.

We add:

  1. NUMERIC BIAS   - every retained target concept contains >= `--min-numeric` numeric
                      data-property restrictions (OWLDataSomeValuesFrom over an OWLDatatypeRestriction).
                      Targets are built by conjoining refinement-operator output with numeric atoms
                      and numeric *intervals* (min_inclusive AND max_inclusive on the same property).
  2. HARDNESS       - an LP is kept only if no "cheap" hypothesis (any atomic class, its negation,
                      any single numeric atom, any bare exists r.Top) reaches F1 >= --max-baseline-f1
                      on it. This is exactly the region where greedy top-down learners (CELOE/OCEL)
                      start, so a low baseline F1 means the learner cannot shortcut the answer.
                      Negatives are additionally sampled as near-misses: individuals that satisfy
                      the target with one conjunct ablated (e.g. right class, wrong numeric range).

Usage
-----
    python generate_numeric_lps.py --kb KGs/Mutagenesis/mutagenesis.owl \
        --output lps_numeric.json --num-lps 50 --min-examples 10

"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.value_splitter import BinningValueSplitter
from ontolearn.utils.static_funcs import concept_len

from owlapy import owl_expression_to_dl
from owlapy.class_expression import (
    OWLClass,
    OWLClassExpression,
    OWLDataSomeValuesFrom,
    OWLDatatypeRestriction,
    OWLObjectComplementOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectUnionOf,
    OWLThing,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLDataProperty
from owlapy.providers import (
    owl_datatype_max_inclusive_restriction,
    owl_datatype_min_inclusive_restriction,
)

Individuals = FrozenSet[OWLNamedIndividual]


# --------------------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------------------
def numeric_atoms(kb: KnowledgeBase, max_nr_splits: int) -> List[OWLDataSomeValuesFrom]:
    """Build ∃ dp.[>= v] and ∃ dp.[<= v] atoms for every numeric data property.
    """
    splitter = BinningValueSplitter(max_nr_splits=max_nr_splits)
    dps: List[OWLDataProperty] = list(kb.get_numeric_data_properties())
    if not dps:
        raise RuntimeError(
            "This knowledge base exposes no numeric data properties, so a numerically biased "
            "learning problem cannot be built."
        )
    splits: Dict[OWLDataProperty, List] = splitter.compute_splits_properties(kb.reasoner, dps)

    atoms: List[OWLDataSomeValuesFrom] = []
    for dp, values in splits.items():
        for v in values:
            atoms.append(OWLDataSomeValuesFrom(property=dp, filler=owl_datatype_min_inclusive_restriction(v)))
            atoms.append(OWLDataSomeValuesFrom(property=dp, filler=owl_datatype_max_inclusive_restriction(v)))
    return atoms


def numeric_intervals(kb: KnowledgeBase, max_nr_splits: int, max_intervals: int,
                      rng: random.Random) -> List[OWLObjectIntersectionOf]:
    splitter = BinningValueSplitter(max_nr_splits=max_nr_splits)
    dps = list(kb.get_numeric_data_properties())
    splits = splitter.compute_splits_properties(kb.reasoner, dps)

    intervals: List[OWLObjectIntersectionOf] = []
    for dp, values in splits.items():
        if len(values) < 2:
            continue
        for _ in range(max_intervals):
            lo, hi = sorted(rng.sample(range(len(values)), 2))
            if lo == hi:
                continue
            intervals.append(
                OWLObjectIntersectionOf([
                    OWLDataSomeValuesFrom(property=dp, filler=owl_datatype_min_inclusive_restriction(values[lo])),
                    OWLDataSomeValuesFrom(property=dp, filler=owl_datatype_max_inclusive_restriction(values[hi])),
                ])
            )
    return intervals


def count_numeric(ce: OWLClassExpression) -> int:
    """Number of numeric data restrictions occurring anywhere in `ce`."""
    if isinstance(ce, OWLDataSomeValuesFrom):
        return 1 if isinstance(ce.get_filler(), OWLDatatypeRestriction) else 0
    if isinstance(ce, (OWLObjectIntersectionOf, OWLObjectUnionOf)):
        return sum(count_numeric(op) for op in ce.operands())
    if isinstance(ce, OWLObjectComplementOf):
        return count_numeric(ce.get_operand())
    if isinstance(ce, (OWLObjectSomeValuesFrom,)):
        return count_numeric(ce.get_filler())
    return 0


def conjuncts(ce: OWLClassExpression) -> List[OWLClassExpression]:
    """Flatten a top-level intersection into its conjuncts."""
    if isinstance(ce, OWLObjectIntersectionOf):
        out: List[OWLClassExpression] = []
        for op in ce.operands():
            out.extend(conjuncts(op))
        return out
    return [ce]


# --------------------------------------------------------------------------------------
# hardness
# --------------------------------------------------------------------------------------
def f1(pos: Individuals, neg: Individuals, covered: Individuals) -> float:
    tp = len(pos & covered)
    fp = len(neg & covered)
    fn = len(pos - covered)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


class BaselineProbe:
    """The hypotheses a greedy top-down learner reaches almost immediately.

    If any of them already separates pos/neg, the LP is easy and we throw it away.
    """

    def __init__(self, kb: KnowledgeBase, num_atoms: List[OWLDataSomeValuesFrom], probe_cap: int,
                 rng: random.Random):
        cheap: List[OWLClassExpression] = []
        for cls in kb.ontology.classes_in_signature():
            cheap.append(cls)
            cheap.append(OWLObjectComplementOf(cls))
        for op in kb.get_object_properties():
            cheap.append(OWLObjectSomeValuesFrom(property=op, filler=OWLThing))
        cheap.extend(num_atoms)

        if len(cheap) > probe_cap:
            cheap = rng.sample(cheap, k=probe_cap)

        self.extensions: List[Tuple[OWLClassExpression, Individuals]] = []
        for ce in cheap:
            try:
                self.extensions.append((ce, kb.individuals_set(ce)))
            except Exception:  # a malformed probe should never kill the run
                continue

    def best(self, pos: Individuals, neg: Individuals) -> Tuple[float, Optional[OWLClassExpression], Individuals]:
        best_score, best_ce, best_ext = 0.0, None, frozenset()
        for ce, ext in self.extensions:
            score = f1(pos, neg, ext)
            if score > best_score:
                best_score, best_ce, best_ext = score, ce, ext
        return best_score, best_ce, best_ext


def near_miss_negatives(kb: KnowledgeBase, target: OWLClassExpression, pos: Individuals) -> Individuals:
    """Individuals satisfying the target with exactly one conjunct dropped.

    Example: For C = Compound ⊓ ∃logp.[>= 2.1] these are compounds whose logp is just below the threshold.
    """
    parts = conjuncts(target)
    if len(parts) < 2:
        return frozenset()
    near: Set[OWLNamedIndividual] = set()
    for i in range(len(parts)):
        ablated = [p for j, p in enumerate(parts) if j != i]
        ce = ablated[0] if len(ablated) == 1 else OWLObjectIntersectionOf(ablated)
        try:
            near |= set(kb.individuals_set(ce))
        except Exception:
            continue
    return frozenset(near) - pos


# --------------------------------------------------------------------------------------
# target concept construction
# --------------------------------------------------------------------------------------
def build_targets(kb: KnowledgeBase, args, rng: random.Random) -> List[OWLClassExpression]:
    """Refinement-operator output, forcibly conjoined with numeric atoms/intervals."""
    rho = ExpressRefinement(
        knowledge_base=kb,
        max_child_length=args.max_child_length,
        expressivity=args.expressivity,
        downsample=True,
        sample_fillers_count=args.sample_fillers_count,
        use_inverse=True,
        use_card_restrictions=True,
        use_numeric_datatypes=True,   # ALCHIQ(D)
        use_time_datatypes=False,
        use_boolean_datatype=False,
        value_splitter=BinningValueSplitter(max_nr_splits=args.num_splits),
        random_seed=args.seed,
    )

    atoms = numeric_atoms(kb, args.num_splits)
    intervals = numeric_intervals(kb, args.num_splits, args.max_intervals, rng)
    numeric_pool: List[OWLClassExpression] = atoms + intervals
    print(f"[i] numeric pool: {len(atoms)} threshold atoms + {len(intervals)} interval length")

    # structural components from the refinement operator
    roots = list(rho.refine(OWLThing, max_length=args.max_child_length))
    print(f"[i] |refinements of Thing| = {len(roots)}")
    structural_components: Set[OWLClassExpression] = set(roots)
    for root in rng.sample(roots, k=min(args.num_sub_roots, len(roots))):
        try:
            structural_components.update(rho.refine(root, max_length=args.max_child_length))
        except Exception:
            continue
    structural_components_l = [b for b in structural_components if kb.individuals_count(b) >= args.min_examples]
    print(f"[i] structural components concepts with >= {args.min_examples} instances: {len(structural_components_l)}")

    # force the numeric bias
    targets: List[OWLClassExpression] = []
    for _ in range(args.candidate_pool):
        k = rng.randint(args.min_numeric, args.max_numeric)
        picks = rng.sample(numeric_pool, k=min(k, len(numeric_pool)))
        parts: List[OWLClassExpression] = []
        # 25% purely numeric targets 
        if structural_components_l and rng.random() > 0.25:
            parts.append(rng.choice(structural_components_l))
        parts.extend(picks)
        if len(parts) < 2:
            continue
        ce: OWLClassExpression = OWLObjectIntersectionOf(parts)
        # occasionally deepen the target with one more refinement step
        if rng.random() < args.deepen_prob:
            try:
                refs = list(rho.refine(ce, max_length=args.max_child_length + 10))
                refs = [r for r in refs if count_numeric(r) >= args.min_numeric]
                if refs:
                    ce = rng.choice(refs)
            except Exception:
                pass
        targets.append(ce)
    return targets


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb", required=True, help="path to the .owl knowledge base")
    p.add_argument("--output", default="lps_numeric.json")
    p.add_argument("--metadata", default=None,
                   help="optional sidecar with length/hardness stats")
    p.add_argument("--num-lps", type=int, default=50, help="how many LPs to keep at maximum")
    p.add_argument("--min-examples", type=int, default=10,
                   help="minimum number of positive AND negative examples per LP")
    p.add_argument("--max-examples", type=int, default=100,
                   help="cap on examples per side (sampled)")
    # bias
    p.add_argument("--min-numeric", type=int, default=1)
    p.add_argument("--max-numeric", type=int, default=4)
    p.add_argument("--num-splits", type=int, default=12,
                   help="BinningValueSplitter bins; higher = harder")
    p.add_argument("--max-intervals", type=int, default=6)
    # hardness
    p.add_argument("--max-baseline-f1", type=float, default=0.85)
    p.add_argument("--hard-negative-ratio", type=float, default=0.7,
                   help="fraction of negatives drawn from the near-miss pool")
    p.add_argument("--probe-cap", type=int, default=400, help="max cheap hypotheses used for the baseline probe")
    # search size
    p.add_argument("--max-child-length", type=int, default=25)
    p.add_argument("--expressivity", type=float, default=0.4)
    p.add_argument("--sample-fillers-count", type=int, default=10)
    p.add_argument("--num-sub-roots", type=int, default=50)
    p.add_argument("--candidate-pool", type=int, default=600)
    p.add_argument("--deepen-prob", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)

    kb = KnowledgeBase(path=args.kb)
    all_inds: Individuals = frozenset(kb.individuals())
    print(f"[i] individuals: {len(all_inds)}")
    if len(all_inds) < 2 * args.min_examples:
        raise SystemExit("KB too small for the requested --min-examples.")

    targets = build_targets(kb, args, rng)
    print(f"[i] candidate targets: {len(targets)}")

    probe = BaselineProbe(kb, numeric_atoms(kb, args.num_splits), args.probe_cap, rng)
    print(f"[i] baseline probe hypotheses: {len(probe.extensions)}")

    scored = []
    # non-redundancy
    seen_ext: Set[Individuals] = set()   
    seen_dl: Set[str] = set()

    for ce in targets:
        try:
            pos = kb.individuals_set(ce)
        except Exception:
            continue
        neg_all = all_inds - pos
        if len(pos) < args.min_examples or len(neg_all) < args.min_examples:
            continue
        if pos in seen_ext:
            continue

        nnf = ce.get_nnf()
        dl = owl_expression_to_dl(nnf)
        if dl in seen_dl:
            continue

        n_num = count_numeric(nnf)
        if n_num < args.min_numeric:
            continue

        base_f1, _base_ce, base_ext = probe.best(pos, neg_all)
        if base_f1 >= args.max_baseline_f1:
            continue   

        near = near_miss_negatives(kb, ce, pos) & neg_all
        length = concept_len(nnf)

        # This does not really make sense as a hardness score
        hardness = (
            0.30 * min(length / 15.0, 1.0)
            + 0.25 * min(n_num / 4.0, 1.0)
            + 0.35 * (1.0 - base_f1)
            + 0.10 * min(len(near) / max(len(pos), 1), 1.0)
        )

        seen_ext.add(pos)
        seen_dl.add(dl)
        scored.append(dict(dl=dl, pos=pos, neg=neg_all, near=near, length=length,
                           numeric=n_num, baseline_f1=base_f1, hardness=hardness,
                           baseline_covered=base_ext))

    scored.sort(key=lambda d: d["hardness"], reverse=True)
    scored = scored[: args.num_lps]
    print(f"[i] retained {len(scored)} hard numeric LPs")

    problems: Dict[str, Dict[str, List[str]]] = {}
    meta: Dict[str, Dict] = {}

    for item in scored:
        pos, neg, near = item["pos"], item["neg"], item["near"]
        n_pos = min(len(pos), args.max_examples)
        missed = list(pos - item["baseline_covered"])
        rest_pos = list(pos - set(missed))
        rng.shuffle(missed)
        rng.shuffle(rest_pos)
        pos_sample = (missed + rest_pos)[:n_pos]

        n_neg = min(len(neg), args.max_examples)
        n_hard = int(round(args.hard_negative_ratio * n_neg))
        near_l, far_l = list(near), list(neg - near)
        rng.shuffle(near_l)
        rng.shuffle(far_l)
        neg_sample = near_l[:n_hard]
        neg_sample += far_l[: n_neg - len(neg_sample)]
        if len(neg_sample) < n_neg:
            neg_sample += near_l[n_hard: n_hard + (n_neg - len(neg_sample))]

        if len(pos_sample) < args.min_examples or len(neg_sample) < args.min_examples:
            continue

        problems[item["dl"]] = {
            "positive_examples": sorted(i.str for i in pos_sample),
            "negative_examples": sorted(i.str for i in neg_sample),
        }
        meta[item["dl"]] = {
            "length": item["length"],
            "num_numeric_restrictions": item["numeric"],
            "best_baseline_f1": round(item["baseline_f1"], 4),
            "hardness": round(item["hardness"], 4),
            "num_hard_negatives": len(near_l[:n_hard]),
            "total_positives_in_kb": len(pos),
        }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"problems": problems}, fh, indent=2, ensure_ascii=False)
    print(f"[i] saved {len(problems)} LPs to {args.output}")

    if args.metadata:
        with open(args.metadata, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        print(f"[i] saved stats to {args.metadata}")


if __name__ == "__main__":
    main()