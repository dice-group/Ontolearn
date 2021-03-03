import json
import os

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CustomConceptLearner
from ontolearn.heuristics import DLFOILHeuristic

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
model = CustomConceptLearner(knowledge_base=kb, heuristic_func=DLFOILHeuristic())

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    model.fit(pos=p, neg=n)
    # Get Top n hypotheses
    hypotheses = model.best_hypotheses(n=1)
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(p) + list(n), hypotheses=hypotheses)
