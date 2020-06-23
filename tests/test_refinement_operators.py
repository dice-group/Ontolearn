""" Test for refinement_operators.py"""
from ontolearn import KnowledgeBase, CustomRefinementOperator

def test_refinement_operator():
    kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')
    rho = CustomRefinementOperator(kb)
    for refs in enumerate(rho.refine(kb.thing)):
        pass
