""" Test for refinement_operators.py"""
import json

from ontolearn import KnowledgeBase
from ontolearn.utils import setup_logging
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.refinement_operators import CustomRefinementOperator, ModifiedCELOERefinement, LengthBasedRefinement
from ontolearn.search import Node

setup_logging("logging_test.conf")

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# because '../KGs/Family/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=settings['data_path'][3:])


def test_celoe_refinement_operator():
    r = DLSyntaxObjectRenderer()
    rho = ModifiedCELOERefinement(kb)
    for _ in enumerate(rho.refine(kb.thing, max_length=10, current_domain=kb.thing)):
        print(r.render(_[1]))
        pass


def test_length_refinement_operator():
    r = DLSyntaxObjectRenderer()
    rho = LengthBasedRefinement(kb)
    for _ in enumerate(rho.refine(kb.thing, max_length=5, apply_combinations=False)):
        print(r.render(_[1]))
        pass


if __name__ == '__main__':
    test_celoe_refinement_operator()
    test_length_refinement_operator()

