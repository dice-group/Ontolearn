import json
import random

import numpy as np
import pytest
from deap import gp
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.ea_semantic import lexicase_select
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard


class _Ind(list):
    def __init__(self, errors, length):
        super().__init__(range(length))
        self.errors = np.array(errors, dtype=bool)


def test_lexicase_keeps_specialists_and_prefers_short_trees():
    random.seed(0)
    a = _Ind([0, 1, 1, 1], 5)  # only right on example 0
    b = _Ind([1, 0, 1, 1], 5)  # only right on example 1
    dominated = _Ind([1, 1, 1, 1], 1)  # wrong everywhere (short, but must never win)
    picks = lexicase_select([a, b, dominated], 200)
    assert all(p is not dominated for p in picks)
    assert {id(p) for p in picks} == {id(a), id(b)}  # both specialists get selected

    long_, short = _Ind([0, 0], 9), _Ind([0, 0], 3)  # identical behaviour: the shorter one wins
    assert all(p is short for p in lexicase_select([long_, short], 20))


def test_lexicase_ignores_examples_nobody_solves():
    a, b = _Ind([1, 0], 4), _Ind([1, 1], 4)
    assert all(p is a for p in lexicase_select([a, b], 20))


def test_invalid_selection_is_rejected():
    kb = KnowledgeBase(path='KGs/Family/family-benchmark_rich_background.owl')
    with pytest.raises(ValueError):
        EvoLearner(knowledge_base=kb, selection='roulette')


def test_lexicase_learns_and_keeps_consistent_errors():
    with open('examples/synthetic_problems.json') as f:
        settings = json.load(f)
    ex = settings['problems']['Uncle']
    lp = PosNegLPStandard(pos={OWLNamedIndividual(i) for i in ex['positive_examples']},
                          neg={OWLNamedIndividual(i) for i in ex['negative_examples']})
    kb = KnowledgeBase(path=settings['data_path'][3:])
    random.seed(3)
    model = EvoLearner(knowledge_base=kb, selection='lexicase', population_size=100, num_generations=8,
                       max_runtime=120, terminate_on_goal=False)
    model.fit(lp)
    assert len(model._result_population) == 100

    # The stored error vector must agree with a fresh evaluation of the individual's concept
    n_pos = len(ex['positive_examples'])
    for ind in model._result_population[:20]:
        covered = kb.individuals_set(gp.compile(ind, model.pset))
        expected = [(e in covered) != (i < n_pos) for i, e in enumerate(model._example_index)]
        assert ind.errors.tolist() == expected


def _uncle():
    with open('examples/synthetic_problems.json') as f:
        settings = json.load(f)
    ex = settings['problems']['Uncle']
    lp = PosNegLPStandard(pos={OWLNamedIndividual(i) for i in ex['positive_examples']},
                          neg={OWLNamedIndividual(i) for i in ex['negative_examples']})
    return KnowledgeBase(path=settings['data_path'][3:]), lp


def test_default_selection_is_tournament_and_still_sets_errors():
    # Regression: the default must stay the original tournament selection, and the new error vectors must be
    # attached to every individual regardless of the selection method.
    kb, lp = _uncle()
    random.seed(1)
    model = EvoLearner(knowledge_base=kb, population_size=50, num_generations=3, max_runtime=120,
                       terminate_on_goal=False)
    assert model.selection == 'tournament'
    model.fit(lp)
    assert all(ind.errors.dtype == bool and len(ind.errors) == len(model._example_index)
               for ind in model._result_population)


def test_cache_hit_restores_the_same_errors():
    # Regression: an individual whose tree was evaluated before must get its error vector back from the cache.
    kb, lp = _uncle()
    random.seed(2)
    model = EvoLearner(knowledge_base=kb, population_size=20, num_generations=1, max_runtime=120,
                       terminate_on_goal=False)
    model.fit(lp)
    original = model._result_population[0]
    twin = model.toolbox.clone(original)
    del twin.fitness.values
    del twin.errors
    tested_before = model.number_of_tested_concepts
    model._fitness_func(twin)
    assert model.number_of_tested_concepts == tested_before  # served from the cache, no new reasoner call
    assert np.array_equal(twin.errors, original.errors)


def test_lexicase_finds_a_perfect_hypothesis_on_a_simple_problem():
    # Regression on an easy problem (Uncle on Family): lexicase must still be able to solve it completely.
    kb, lp = _uncle()
    random.seed(1)
    model = EvoLearner(knowledge_base=kb, selection='lexicase', population_size=200, num_generations=30,
                       max_runtime=120, terminate_on_goal=True)
    model.fit(lp)
    best = next(model._get_top_hypotheses(model._result_population, 1))
    assert best.quality == 1.0
