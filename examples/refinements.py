from OntoPy import KnowledgeBase, Refinement

kb = KnowledgeBase(path='OntoPy/data/family-benchmark_rich_background.owl')
rho = Refinement(kb)
for refs in enumerate(rho.refine(kb.thing)):
    print(refs)