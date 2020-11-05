""" Test for refinement_operators.py"""
from ontolearn import KnowledgeBase, CustomRefinementOperator
import json
with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# because '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=settings['data_path'][3:])

def test_refinement_operator():
    rho = CustomRefinementOperator(kb)
    for _ in enumerate(rho.refine(kb.thing)):
        pass
