""" kb_enrichment.py illustrates extending Tbox of given KB and saving KB in different formats."""
from ontolearn import KnowledgeBase, CustomRefinementOperator,Data


kb = KnowledgeBase(path='../data/family-benchmark_rich_background.owl')
data = Data(knowledge_base=kb)
rho = CustomRefinementOperator(kb)

enrichments = []
for ith, refs in enumerate(rho.refine(kb.thing)):
    if len(refs) == 2 and len(refs.instances) < 150:  # selection criterion
        enrichments.append(refs)

kb.apply_type_enrichment_from_iterable(enrichments)
kb.save('enriched_kb.owl',rdf_format="rdfxml")
kb.save('enriched_kb.nt',rdf_format="ntriples")