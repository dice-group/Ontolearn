from core.base import KnowledgeBase
from core.refinement_operators import Refinement
from core.data_struct import Data

kb = KnowledgeBase(path='../data/family-benchmark_rich_background.owl')

data = Data(knowledge_base=kb)
rho = Refinement(kb)

enrichments = []
for ith, refs in enumerate(rho.refine(kb.thing)):
    if len(refs) == 2 and len(refs.instances) < 150:  # selection criterion
        enrichments.append(refs)

kb.apply_type_enrichment(enrichments)
kb.save('enriched_kb.owl')
