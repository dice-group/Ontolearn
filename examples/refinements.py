from ontolearn import KnowledgeBase, CustomRefinementOperator
kb = KnowledgeBase(path='../data/family-benchmark_rich_background.owl')
rho = CustomRefinementOperator(kb)
for r in rho.refine(kb.thing):
    print(r)