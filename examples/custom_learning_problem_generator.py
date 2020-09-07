from ontolearn import KnowledgeBase, LengthBasedRefinement, LearningProblemGenerator
import json

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
rho = LengthBasedRefinement(kb=kb)
lp_gen = LearningProblemGenerator(knowledge_base=kb,
                                  refinement_operator=rho,
                                  num_problems=200, depth=2, min_length=3)

for path in lp_gen:
    for p in path:
        print(p) # node
        # p.concept =>  concept
        # p.concept.str => # string representation of concept.
    print('###')

kb.save('../data/extended_family-benchmark_rich_background.owl',rdf_format='rdfxml')