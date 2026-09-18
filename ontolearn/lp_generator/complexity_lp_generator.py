# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""Goal-directed, complexity-controlled learning-problem generation.

``ontolearn.lp_generator.LPGen``/``KB2Data`` synthesize learning problems bottom-up: they
refine ``owl:Thing`` with ``ExpressRefinement`` and keep whatever non-redundant concepts fall
out, sorted by length only as a dedup tie-break. There is no way to ask for "a learning problem
whose target uses exactly a nested existential restriction and an object intersection, depth <= 3".

This module instead samples a target ``OWLClassExpression`` **top-down from an explicit DL-construct
grammar** (:class:`ComplexityProfile`): the caller names which constructors are allowed
(intersection, union, existential/universal restriction, cardinality restrictions, has-value,
negation, inverse roles) and bounds on length/depth, and the generator recursively builds concepts
that respect that budget. Each candidate is turned into a learning problem via reasoner-based
retrieval (``kb.individuals_set``) and kept only if it clears minimum example-count and hardness
bars (no "cheap" atomic/negated-atomic/bare-existential hypothesis already separates pos/neg).

A :class:`ComplexityProfile` curriculum (e.g. atomic -> conjunctive -> nested existential ->
cardinality -> full ALCQ) can then be run through :meth:`GoalDirectedLPGenerator.generate_benchmark`
to build a graded benchmark suite, e.g. to measure how a learner's F1 degrades as target-concept
complexity grows -- see ``examples/benchmark_complexity_lp_generator_with_tdl.py``.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from owlapy import owl_expression_to_dl
from owlapy.class_expression import (
    OWLClass,
    OWLClassExpression,
    OWLObjectAllValuesFrom,
    OWLObjectComplementOf,
    OWLObjectExactCardinality,
    OWLObjectIntersectionOf,
    OWLObjectMaxCardinality,
    OWLObjectMinCardinality,
    OWLObjectSomeValuesFrom,
    OWLObjectUnionOf,
    OWLThing,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLObjectProperty, OWLObjectPropertyExpression

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils.static_funcs import concept_len

Individuals = FrozenSet[OWLNamedIndividual]

__all__ = [
    "DLConstruct",
    "ComplexityProfile",
    "GeneratedLearningProblem",
    "GoalDirectedLPGenerator",
    "save_benchmark",
]


class DLConstruct(Enum):
    """DL constructors a :class:`ComplexityProfile` can allow/forbid."""
    ATOMIC = "atomic"
    NEGATION = "negation"
    INTERSECTION = "intersection"
    UNION = "union"
    EXISTENTIAL = "existential"
    UNIVERSAL = "universal"
    MIN_CARDINALITY = "min_cardinality"
    MAX_CARDINALITY = "max_cardinality"
    EXACT_CARDINALITY = "exact_cardinality"
    HAS_VALUE = "has_value"


_RESTRICTION_CONSTRUCTS = frozenset({
    DLConstruct.EXISTENTIAL, DLConstruct.UNIVERSAL,
    DLConstruct.MIN_CARDINALITY, DLConstruct.MAX_CARDINALITY, DLConstruct.EXACT_CARDINALITY,
})


@dataclass(frozen=True)
class ComplexityProfile:
    """A named point (or curriculum step) in the space of "how complex should the target be".

    :param name: identifier used to tag generated problems and group benchmark results.
    :param allowed_constructs: DL constructors :meth:`GoalDirectedLPGenerator.generate` may use
        when sampling a target concept.
    :param min_length/max_length: bounds on ``concept_len`` of the (NNF) target.
    :param min_depth/max_depth: bounds on restriction-nesting depth (see
        :meth:`GoalDirectedLPGenerator.concept_depth`); a bare named class has depth 0.
    :param max_arity: maximum number of operands sampled under one intersection/union.
    :param card_limit: cardinality restrictions sample ``n`` uniformly from ``[1, card_limit]``.
    :param use_inverse_roles: if True, restrictions may use ``r-`` in place of ``r``.
    :param min_pos/min_neg: an LP is discarded unless the target has at least this many
        positive and negative instances in the knowledge base.
    :param max_examples_per_side: positives/negatives are subsampled down to this cap.
    :param max_baseline_f1: an LP is discarded if the best of a pool of "cheap" one-step
        hypotheses (an atomic class, its negation, or a bare ``exists r.Top``) already reaches
        this F1 -- i.e. it is trivially solvable without the constructs being tested.
    """
    name: str
    allowed_constructs: FrozenSet[DLConstruct]
    min_length: int = 1
    max_length: int = 10
    min_depth: int = 0
    max_depth: int = 3
    max_arity: int = 3
    card_limit: int = 5
    use_inverse_roles: bool = False
    min_pos: int = 3
    min_neg: int = 3
    max_examples_per_side: int = 200
    max_baseline_f1: float = 0.9


@dataclass
class GeneratedLearningProblem:
    """A goal-directed learning problem: a sampled target concept plus its (E+, E-)."""
    profile_name: str
    target_concept: OWLClassExpression
    dl: str
    pos: Individuals
    neg: Individuals
    length: int
    depth: int
    construct_histogram: Dict[str, int]
    baseline_f1: float

    def to_lp(self) -> PosNegLPStandard:
        """Build the ``PosNegLPStandard`` consumed by Ontolearn learners (e.g. TDL)."""
        return PosNegLPStandard(pos=set(self.pos), neg=set(self.neg))


def _f1(pos: Individuals, neg: Individuals, covered: Individuals) -> float:
    tp = len(pos & covered)
    if tp == 0:
        return 0.0
    fp = len(neg & covered)
    fn = len(pos - covered)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


class GoalDirectedLPGenerator:
    """Samples target concepts from an explicit DL-construct grammar and turns them into LPs.

    Usage::

        kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        gen = GoalDirectedLPGenerator(kb, seed=1)
        profile = ComplexityProfile(
            name="nested-existential",
            allowed_constructs=frozenset({DLConstruct.EXISTENTIAL, DLConstruct.INTERSECTION}),
            max_length=8, max_depth=3,
        )
        lps = gen.generate(profile, num_lps=10)
        model = TDL(kb).fit(lps[0].to_lp())
    """

    def __init__(self, knowledge_base: KnowledgeBase, seed: Optional[int] = None, probe_cap: int = 300):
        self.kb = knowledge_base
        self.rng = random.Random(seed)
        self.all_individuals: Individuals = frozenset(self.kb.individuals())
        self.atomic_classes: List[OWLClass] = [
            c for c in self.kb.ontology.classes_in_signature() if c != OWLThing
        ]
        self.object_properties: List[OWLObjectProperty] = list(self.kb.get_object_properties())
        if not self.atomic_classes:
            raise ValueError("Knowledge base exposes no named classes to build targets from.")
        self._probe: List[Tuple[OWLClassExpression, Individuals]] = self._build_baseline_probe(probe_cap)

    # ------------------------------------------------------------------ hardness / baseline
    def _build_baseline_probe(self, probe_cap: int) -> List[Tuple[OWLClassExpression, Individuals]]:
        cheap: List[OWLClassExpression] = []
        for c in self.atomic_classes:
            cheap.append(c)
            cheap.append(self.kb.generator.negation(c))
        for p in self.object_properties:
            cheap.append(self.kb.generator.existential_restriction(OWLThing, p))
        if len(cheap) > probe_cap:
            cheap = self.rng.sample(cheap, k=probe_cap)
        extensions: List[Tuple[OWLClassExpression, Individuals]] = []
        for ce in cheap:
            try:
                extensions.append((ce, self.kb.individuals_set(ce)))
            except Exception:
                continue
        return extensions

    def baseline_f1(self, pos: Individuals, neg: Individuals) -> float:
        """Best F1 any single "cheap" one-step hypothesis reaches on (pos, neg)."""
        best = 0.0
        for _, ext in self._probe:
            score = _f1(pos, neg, ext)
            if score > best:
                best = score
        return best

    # ------------------------------------------------------------------ concept-shape utilities
    @staticmethod
    def concept_depth(ce: OWLClassExpression) -> int:
        """Restriction-nesting depth: a named class or ``has value`` atom has depth 0."""
        if isinstance(ce, (OWLObjectIntersectionOf, OWLObjectUnionOf)):
            return max((GoalDirectedLPGenerator.concept_depth(o) for o in ce.operands()), default=0)
        if isinstance(ce, OWLObjectComplementOf):
            return GoalDirectedLPGenerator.concept_depth(ce.get_operand())
        if isinstance(ce, (OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom,
                            OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality)):
            return 1 + GoalDirectedLPGenerator.concept_depth(ce.get_filler())
        return 0

    def _sample_property(self, profile: ComplexityProfile) -> OWLObjectPropertyExpression:
        prop = self.rng.choice(self.object_properties)
        if profile.use_inverse_roles and self.rng.random() < 0.3:
            return prop.get_inverse_property()
        return prop

    def _sample_atom(self, profile: ComplexityProfile, hist: Counter) -> OWLClassExpression:
        c = self.rng.choice(self.atomic_classes)
        if DLConstruct.NEGATION in profile.allowed_constructs and self.rng.random() < 0.3:
            hist[DLConstruct.NEGATION] += 1
            return self.kb.generator.negation(c)
        hist[DLConstruct.ATOMIC] += 1
        return c

    def _sample_rec(self, profile: ComplexityProfile, depth_budget: int, length_budget: int,
                     hist: Counter) -> OWLClassExpression:
        # Intersection/union/has-value don't add restriction-nesting depth (see `concept_depth`),
        # so they stay available even once `depth_budget` is exhausted; only ∃/∀/cardinality -
        # which each cost one level of nesting for their filler - are gated on `depth_budget > 0`.
        depth_free = [c for c in profile.allowed_constructs
                      if c in (DLConstruct.INTERSECTION, DLConstruct.UNION, DLConstruct.HAS_VALUE)]
        options = list(depth_free)
        if depth_budget > 0:
            options += [c for c in profile.allowed_constructs if c in _RESTRICTION_CONSTRUCTS]
        if length_budget <= 1 or not options:
            return self._sample_atom(profile, hist)

        construct = self.rng.choice(options)

        if construct in (DLConstruct.INTERSECTION, DLConstruct.UNION):
            arity = self.rng.randint(2, max(2, profile.max_arity))
            parts: List[OWLClassExpression] = []
            remaining_len = length_budget - 1
            for i in range(arity):
                if remaining_len <= 1:
                    break
                share = max(1, remaining_len // (arity - i))
                sub = self._sample_rec(profile, depth_budget, share, hist)
                parts.append(sub)
                remaining_len -= concept_len(sub)
            if len(parts) < 2:
                return parts[0] if parts else self._sample_atom(profile, hist)
            hist[construct] += 1
            return self.kb.generator.intersection(parts) if construct == DLConstruct.INTERSECTION \
                else self.kb.generator.union(parts)

        if construct in (DLConstruct.EXISTENTIAL, DLConstruct.UNIVERSAL):
            prop = self._sample_property(profile)
            filler = self._sample_rec(profile, depth_budget - 1, length_budget - 2, hist)
            hist[construct] += 1
            return self.kb.generator.existential_restriction(filler, prop) if construct == DLConstruct.EXISTENTIAL \
                else self.kb.generator.universal_restriction(filler, prop)

        if construct in (DLConstruct.MIN_CARDINALITY, DLConstruct.MAX_CARDINALITY, DLConstruct.EXACT_CARDINALITY):
            prop = self._sample_property(profile)
            card = self.rng.randint(1, max(1, profile.card_limit))
            if depth_budget > 1 and length_budget > 3:
                filler = self._sample_rec(profile, depth_budget - 1, length_budget - 3, hist)
            else:
                filler = self.kb.generator.thing
            hist[construct] += 1
            if construct == DLConstruct.MIN_CARDINALITY:
                return self.kb.generator.min_cardinality_restriction(filler, prop, card)
            if construct == DLConstruct.MAX_CARDINALITY:
                return self.kb.generator.max_cardinality_restriction(filler, prop, card)
            return self.kb.generator.exact_cardinality_restriction(filler, prop, card)

        if construct == DLConstruct.HAS_VALUE:
            prop = self._sample_property(profile)
            ind = self.rng.choice(tuple(self.all_individuals))
            hist[construct] += 1
            return self.kb.generator.has_value_restriction(ind, prop)

        return self._sample_atom(profile, hist)  # pragma: no cover - defensive fallback

    def _sample_examples(self, individuals: Individuals, cap: int) -> Individuals:
        pool = list(individuals)
        if len(pool) > cap:
            pool = self.rng.sample(pool, cap)
        return frozenset(pool)

    # ------------------------------------------------------------------ public API
    def sample_target(self, profile: ComplexityProfile) -> Tuple[OWLClassExpression, Counter]:
        """Sample one target concept (in NNF) respecting `profile`, plus its construct histogram."""
        hist: Counter = Counter()
        ce = self._sample_rec(profile, profile.max_depth, profile.max_length, hist)
        return ce.get_nnf(), hist

    def generate(self, profile: ComplexityProfile, num_lps: int, max_attempts: int = 3000
                 ) -> List[GeneratedLearningProblem]:
        """Sample up to `num_lps` distinct, non-trivial learning problems matching `profile`."""
        results: List[GeneratedLearningProblem] = []
        seen_ext: set = set()
        seen_dl: set = set()
        attempts = 0
        while len(results) < num_lps and attempts < max_attempts:
            attempts += 1
            ce, hist = self.sample_target(profile)

            length = concept_len(ce)
            if not (profile.min_length <= length <= profile.max_length):
                continue
            depth = self.concept_depth(ce)
            if not (profile.min_depth <= depth <= profile.max_depth):
                continue

            dl = owl_expression_to_dl(ce)
            if dl in seen_dl:
                continue

            try:
                pos = self.kb.individuals_set(ce)
            except Exception:
                continue
            if pos in seen_ext:
                continue
            neg = self.all_individuals - pos
            if len(pos) < profile.min_pos or len(neg) < profile.min_neg:
                continue

            base_f1 = self.baseline_f1(pos, neg)
            if base_f1 >= profile.max_baseline_f1:
                continue

            seen_ext.add(pos)
            seen_dl.add(dl)
            results.append(GeneratedLearningProblem(
                profile_name=profile.name,
                target_concept=ce,
                dl=dl,
                pos=self._sample_examples(pos, profile.max_examples_per_side),
                neg=self._sample_examples(neg, profile.max_examples_per_side),
                length=length,
                depth=depth,
                construct_histogram={k.value: v for k, v in hist.items()},
                baseline_f1=round(base_f1, 4),
            ))
        return results

    def generate_benchmark(self, profiles: Sequence[ComplexityProfile], num_lps_per_profile: int = 5,
                            max_attempts_per_profile: int = 3000
                            ) -> Dict[str, List[GeneratedLearningProblem]]:
        """Run :meth:`generate` for each profile in a curriculum; returns `{profile.name: [...]}`."""
        return {
            profile.name: self.generate(profile, num_lps_per_profile, max_attempts_per_profile)
            for profile in profiles
        }


def save_benchmark(benchmark: Dict[str, List[GeneratedLearningProblem]], path: str) -> None:
    """Serialize a benchmark (as returned by :meth:`GoalDirectedLPGenerator.generate_benchmark`)
    to JSON: `{dl_string: {"profile", "positive_examples", "negative_examples", "length", "depth",
    "construct_histogram", "baseline_f1"}}`, individuals kept as full IRIs.
    """
    out: Dict[str, dict] = {}
    for profile_name, lps in benchmark.items():
        for lp in lps:
            out[lp.dl] = {
                "profile": profile_name,
                "positive_examples": sorted(i.str for i in lp.pos),
                "negative_examples": sorted(i.str for i in lp.neg),
                "length": lp.length,
                "depth": lp.depth,
                "construct_histogram": lp.construct_histogram,
                "baseline_f1": lp.baseline_f1,
            }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
