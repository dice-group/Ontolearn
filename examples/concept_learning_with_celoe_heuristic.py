""" concept_learning.py"""
from ontolearn import KnowledgeBase, SampleConceptLearner
from ontolearn.metrics import F1, PredictiveAccuracy, CELOEHeuristic
import json

with open('artifical_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])

    model = SampleConceptLearner(knowledge_base=kb,
                                 quality_func=F1(),
                                 terminate_on_goal=True,
                                 heuristic_func=CELOEHeuristic(),
                                 iter_bound=100,
                                 verbose=False)

    model.predict(pos=p, neg=n)
    model.show_best_predictions(top_n=10)

######################################################################

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])

    model = SampleConceptLearner(knowledge_base=kb,
                                 quality_func=PredictiveAccuracy(),
                                 terminate_on_goal=False,
                                 heuristic_func=CELOEHeuristic(),
                                 iter_bound=1000,
                                 verbose=False)

    model.predict(pos=p, neg=n)

    print('\tA target concept:',str_target_concept)
    model.show_best_predictions(top_n=10)

