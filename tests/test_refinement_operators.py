""" Test for refinement_operators.py"""

from ontolearn.base import KnowledgeBase
from ontolearn.refinement_operators import Refinement


def test_refinement_operator():
    path_of_example_kb = 'data/family-benchmark_rich_background.owl'
    kb = KnowledgeBase(path_of_example_kb)
    rho = Refinement(kb)

    for _ in rho.refine(kb.thing):
        assert _ # refinements can not be None.