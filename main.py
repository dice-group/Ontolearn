from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data

kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')

data = Data(knowledge_base=kb)
rho = Refinement(kb)

concepts_to_be_refined = []
refined_concepts = set()

concept_to_refine=kb.T
for i in range(10):
    print(concept_to_refine)
    refinements=rho.refine(concept_to_refine)
    concepts_to_be_refined.extend(refinements)
    concept_to_refine=concepts_to_be_refined.pop()

