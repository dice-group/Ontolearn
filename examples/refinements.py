import os

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import CustomRefinementOperator

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

kb = KnowledgeBase(path='../KGs/Family/family-benchmark_rich_background.owl')
rho = CustomRefinementOperator(kb)
# TODO: Line 14 throws a TypeError
for r in rho.refine(kb.thing):
    print(r)
