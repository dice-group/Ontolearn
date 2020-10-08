import json

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CustomConceptLearner
from ontolearn.heuristics import DLFOILHeuristic
from ontolearn.metrics import F1
from ontolearn.refinement_operators import CustomRefinementOperator
from ontolearn.search import SearchTree

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    model = CustomConceptLearner(knowledge_base=kb)

    model.fit(pos=p, neg=n)
    hypotheses = model.best_hypotheses(n=2)
    for h in hypotheses:
        print(h)
    predictions = model.predict(individuals=list(n), hypotheses=hypotheses)
