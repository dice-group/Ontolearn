from ontolearn import *
import json

with open('artifical_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

p = set(settings['problems']['Brother']['positive_examples'])
n = set(settings['problems']['Brother']['negative_examples'])

model = CELOE(knowledge_base=kb,
              refinement_operator=ModifiedCELOERefinement(kb=kb),
              quality_func=F1(),
              heuristic_func=CELOEHeuristic(),
              search_tree=CELOESearchTree(),
              terminate_on_goal=True,
              iter_bound=100,
              verbose=True)

model.predict(pos=p, neg=n)
model.show_best_predictions(top_n=10)
