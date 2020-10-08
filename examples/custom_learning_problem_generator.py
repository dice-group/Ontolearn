from ontolearn import KnowledgeBase, LearningProblemGenerator
import json

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
lp_gen = LearningProblemGenerator(knowledge_base=kb)

for concept in lp_gen:
    print(concept)
