from ontolearn import KnowledgeBase, CustomRefinementOperator

kb = KnowledgeBase(path='../data/family-benchmark_rich_background.owl')
rho = CustomRefinementOperator(kb)
for refs in enumerate(rho.refine(kb.thing)):
    print(refs)