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

for concept in lp_gen:
    print(concept)
