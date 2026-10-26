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
retrieval (``kb.individuals_set``).

Candidates are additionally ranked by an explicit **difficulty score** (:class:`DifficultyWeights`)
combining four signals: normalized target length/depth, how far the best "cheap" one-step
hypothesis (an atomic class, its negation, or a bare ``exists r.Top``) falls short of solving the
LP outright, and the density of *near-miss negatives* -- individuals that satisfy the target with
exactly one top-level conjunct dropped (e.g. the right class but the wrong relation), which is
exactly where a greedy top-down learner is most likely to overgeneralize. :meth:`generate`
oversamples candidates and keeps the hardest ``num_lps`` of them, and biases the negative sample
towards near-misses via ``hard_negative_ratio``, so the returned problems are pointed at the
learner's actual failure modes rather than generic random negatives.

The grammar and probes are built purely from the knowledge base's own signature
(``kb.ontology.classes_in_signature()``, ``kb.get_object_properties()``), so the same
:class:`ComplexityProfile` curriculum runs unmodified across knowledge graphs with very different
shapes -- e.g. Family, Mutagenesis, Carcinogenesis, Biopax, Lymphography. Relation-based constructs
(∃/∀/cardinality/has-value) are silently dropped from the grammar on a KB that exposes no object
properties (e.g. Lymphography) rather than raising, so a curriculum degrades gracefully instead of
crashing on such KBs.

A :class:`ComplexityProfile` curriculum (e.g. atomic -> conjunctive -> nested existential ->
cardinality -> full ALCQ) can then be run through :meth:`GoalDirectedLPGenerator.generate_benchmark`
to build a graded benchmark suite, e.g. to measure how a learner's F1 degrades as target-concept
complexity grows -- see ``examples/benchmark_complexity_lp_generator_with_tdl.py``.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
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
    "DifficultyWeights",
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
class DifficultyWeights:
    """Weights combining four difficulty signals into one score in (roughly) ``[0, 1]``.

    :param length: weight on ``length / profile.max_length`` (longer target -> harder).
    :param depth: weight on ``depth / profile.max_depth`` (deeper nesting -> harder).
    :param baseline_hardness: weight on ``1 - baseline_f1`` -- how far the best cheap one-step
        hypothesis falls short of solving the LP outright.
    :param near_miss: weight on the fraction of positives that have a matching near-miss negative
        (an individual satisfying the target with exactly one top-level conjunct dropped) -- these
        are the negatives a greedy learner is most likely to accidentally cover.
    """
    length: float = 0.25
    depth: float = 0.15
    baseline_hardness: float = 0.40
    near_miss: float = 0.20


@dataclass(frozen=True)
class ComplexityProfile:
    """A named point (or curriculum step) in the space of "how complex should the target be".

    :param name: identifier used to tag generated problems and group benchmark results.
    :param allowed_constructs: DL constructors :meth:`GoalDirectedLPGenerator.generate` may use
        when sampling a target concept. Relation-based constructs (∃/∀/cardinality/has-value) are
        silently unavailable on a knowledge base exposing no object properties.
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
    :param min_difficulty: an LP is discarded if its :class:`DifficultyWeights`-weighted score
        falls below this bar. ``0.0`` (default) disables this filter.
    :param oversample_factor: :meth:`GoalDirectedLPGenerator.generate` collects up to
        ``num_lps * oversample_factor`` passing candidates (subject to ``max_attempts``) before
        ranking by difficulty and keeping the hardest ``num_lps`` -- i.e. it actively searches
        for hard problems instead of returning the first ones found.
    :param hard_negative_ratio: fraction of the sampled negatives drawn from the near-miss pool
        (when one exists) rather than arbitrary non-instances.
    :param difficulty_weights: see :class:`DifficultyWeights`.
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
    min_difficulty: float = 0.0
    oversample_factor: int = 4
    hard_negative_ratio: float = 0.5
    difficulty_weights: DifficultyWeights = field(default_factory=DifficultyWeights)


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
    difficulty: float
    num_near_miss_negatives: int

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


def _conjuncts(ce: OWLClassExpression) -> List[OWLClassExpression]:
    """Flatten a top-level intersection into its conjuncts (``[ce]`` if `ce` isn't one)."""
    if isinstance(ce, OWLObjectIntersectionOf):
        out: List[OWLClassExpression] = []
        for op in ce.operands():
            out.extend(_conjuncts(op))
        return out
    return [ce]


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
        if not cheap:
            return []
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

    def near_miss_negatives(self, target: OWLClassExpression, pos: Individuals) -> Individuals:
        """Individuals satisfying `target` with exactly one top-level conjunct dropped, minus `pos`.

        E.g. for ``C = Compound ⊓ (∃ hasChild.Father)`` these are individuals that are a
        ``Compound`` but whose child (if any) isn't a ``Father``, or that have a ``Father`` child
        but aren't a ``Compound`` -- the negatives one ablated conjunct away from being positive,
        which is exactly what trips up a greedy top-down learner.
        """
        parts = _conjuncts(target)
        if len(parts) < 2:
            return frozenset()
        near: set = set()
        for i in range(len(parts)):
            ablated = [p for j, p in enumerate(parts) if j != i]
            ce = ablated[0] if len(ablated) == 1 else self.kb.generator.intersection(ablated)
            try:
                near |= set(self.kb.individuals_set(ce))
            except Exception:
                continue
        return frozenset(near) - pos

    @staticmethod
    def difficulty_score(profile: ComplexityProfile, length: int, depth: int, base_f1: float,
                          pos: Individuals, near_miss: Individuals) -> float:
        """Combine length/depth/baseline-hardness/near-miss-density into one score in ``[0, 1]``."""
        w = profile.difficulty_weights
        norm_length = min(length / max(profile.max_length, 1), 1.0)
        norm_depth = min(depth / max(profile.max_depth, 1), 1.0)
        near_density = min(len(near_miss) / max(len(pos), 1), 1.0)
        return (w.length * norm_length + w.depth * norm_depth
                + w.baseline_hardness * (1.0 - base_f1) + w.near_miss * near_density)

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
        # Intersection/union don't add restriction-nesting depth (see `concept_depth`), so they
        # stay available even once `depth_budget` is exhausted; only ∃/∀/cardinality/has-value -
        # which each need at least one object property - are gated on `depth_budget > 0` and on
        # the knowledge base actually exposing object properties (e.g. Lymphography has none, so
        # the grammar degrades to intersection/union/negation of atomic classes on it).
        has_properties = bool(self.object_properties)
        depth_free = [c for c in profile.allowed_constructs
                      if c in (DLConstruct.INTERSECTION, DLConstruct.UNION)
                      or (c == DLConstruct.HAS_VALUE and has_properties)]
        options = list(depth_free)
        if depth_budget > 0 and has_properties:
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

    def _sample_negatives(self, neg: Individuals, near_miss: Individuals, cap: int,
                           hard_ratio: float) -> Individuals:
        """Sample negatives biased towards `near_miss`, capped at `cap`."""
        target_n = min(cap, len(neg))
        near_pool = list(near_miss)
        far_pool = list(neg - near_miss)
        self.rng.shuffle(near_pool)
        self.rng.shuffle(far_pool)
        n_hard = min(len(near_pool), int(round(hard_ratio * target_n)))
        chosen = near_pool[:n_hard] + far_pool[: target_n - n_hard]
        if len(chosen) < target_n:
            chosen += near_pool[n_hard: n_hard + (target_n - len(chosen))]
        return frozenset(chosen)

    # ------------------------------------------------------------------ public API
    def sample_target(self, profile: ComplexityProfile) -> Tuple[OWLClassExpression, Counter]:
        """Sample one target concept (in NNF) respecting `profile`, plus its construct histogram."""
        hist: Counter = Counter()
        ce = self._sample_rec(profile, profile.max_depth, profile.max_length, hist)
        return ce.get_nnf(), hist

    def generate(self, profile: ComplexityProfile, num_lps: int, max_attempts: int = 3000
                 ) -> List[GeneratedLearningProblem]:
        """Sample the `num_lps` **hardest** distinct, non-trivial learning problems matching
        `profile`: up to ``num_lps * profile.oversample_factor`` passing candidates are collected
        (subject to `max_attempts` sampling attempts total), ranked by
        :meth:`difficulty_score` and the top `num_lps` are kept.
        """
        pool_target = max(num_lps, num_lps * profile.oversample_factor)
        candidates: List[GeneratedLearningProblem] = []
        seen_ext: set = set()
        seen_dl: set = set()
        attempts = 0
        while len(candidates) < pool_target and attempts < max_attempts:
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

            near_miss = self.near_miss_negatives(ce, pos) & neg
            difficulty = self.difficulty_score(profile, length, depth, base_f1, pos, near_miss)
            if difficulty < profile.min_difficulty:
                continue

            seen_ext.add(pos)
            seen_dl.add(dl)
            candidates.append(GeneratedLearningProblem(
                profile_name=profile.name,
                target_concept=ce,
                dl=dl,
                pos=self._sample_examples(pos, profile.max_examples_per_side),
                neg=self._sample_negatives(neg, near_miss, profile.max_examples_per_side,
                                            profile.hard_negative_ratio),
                length=length,
                depth=depth,
                construct_histogram={k.value: v for k, v in hist.items()},
                baseline_f1=round(base_f1, 4),
                difficulty=round(difficulty, 4),
                num_near_miss_negatives=len(near_miss),
            ))
        candidates.sort(key=lambda lp: lp.difficulty, reverse=True)
        return candidates[:num_lps]

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
    "construct_histogram", "baseline_f1", "difficulty", "num_near_miss_negatives"}}`, individuals
    kept as full IRIs.
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
                "difficulty": lp.difficulty,
                "num_near_miss_negatives": lp.num_near_miss_negatives,
            }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
