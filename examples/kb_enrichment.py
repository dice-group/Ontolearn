"""
kb_enrichment.py illustrates extending Tbox of given KB and saving KB in
different formats. This is not tested.
"""

from ontolearn import CustomRefinementOperator
from ontolearn import Data
from ontolearn import KnowledgeBase

kb = KnowledgeBase(path='../data/family-benchmark_rich_background.owl')
data = Data(knowledge_base=kb)
rho = CustomRefinementOperator(kb)

enrichments = []
for ith, refs in enumerate(rho.refine(kb.thing)):
    # selection criterion
    if len(refs) == 2 and len(refs.instances) < 150:
        enrichments.append(refs)

kb.apply_type_enrichment_from_iterable(enrichments)
kb.save('enriched_kb.owl', rdf_format="rdfxml")
kb.save('enriched_kb.nt', rdf_format="ntriples")
