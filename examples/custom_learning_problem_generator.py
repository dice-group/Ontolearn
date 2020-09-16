from ontolearn import KnowledgeBase, LengthBasedRefinement, LearningProblemGenerator
from ontolearn.util import serialize_concepts
import json

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
rho = LengthBasedRefinement(kb=kb)
lp_gen = LearningProblemGenerator(knowledge_base=kb,
                                  refinement_operator=rho,
                                  num_problems=10, depth=2, min_length=4)

random_concepts=[]
for path in lp_gen:
    for p in path:
        print(p)  # node
        # p.concept =>  concept
        # p.concept.str => # string representation of concept.
    random_concepts.append(p) # last item

for i in random_concepts:
    serialize_concepts(concepts=[i], serialize_name='dummy_'+i.concept.str, metric='F1', attribute='quality', rdf_format='nt')  #
