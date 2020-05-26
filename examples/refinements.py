from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data

kb = KnowledgeBase(path='../data/family-benchmark_rich_background.owl')

data = Data(knowledge_base=kb)
rho = Refinement(kb)

for ith, refs in enumerate(rho.refine(kb.thing)):
    print(refs)
