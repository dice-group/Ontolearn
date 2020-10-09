from ontolearn import CustomRefinementOperator,KnowledgeBase

kb = KnowledgeBase(path='../data/family-benchmark_rich_background.owl')
rho = CustomRefinementOperator(kb)
for r in rho.refine(kb.thing):
    print(r)
