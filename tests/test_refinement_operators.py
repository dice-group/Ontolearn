""" Test for refinement_operators.py"""
import json

from ontolearn import KnowledgeBase
from ontolearn.owlapy.render import DLSyntaxRenderer
from ontolearn.refinement_operators import CustomRefinementOperator, ModifiedCELOERefinement, LengthBasedRefinement
from ontolearn.search import Node

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# because '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=settings['data_path'][3:])


def test_celoe_refinement_operator():
    r = DLSyntaxRenderer()
    rho = ModifiedCELOERefinement(kb)
    for _ in enumerate(rho.refine(Node(kb.thing, root=True), max_length=10, current_domain=kb.thing)):
        print(r.render(_[1]))
        pass


def test_length_refinement_operator():
    r = DLSyntaxRenderer()
    rho = LengthBasedRefinement(kb)
    for _ in enumerate(rho.refine(Node(kb.thing, root=True), max_length=10, apply_combinations=False)):
        print(r.render(_[1]))
        pass


if __name__ == '__main__':
    test_celoe_refinement_operator()
    test_length_refinement_operator()

