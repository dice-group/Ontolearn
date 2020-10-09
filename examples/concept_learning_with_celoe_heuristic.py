import json

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.metrics import F1
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.search import CELOESearchTree
import random
from sklearn.metrics import confusion_matrix

import numpy as np

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'http://www.benchmark.org/family#Brother',
             'Father', 'http://www.benchmark.org/family#Grandparent'}) # Use URI, or concept with length 1.

    model = CELOE(knowledge_base=kb,
                  ignored_concepts=concepts_to_ignore)

    model.fit(pos=p, neg=n)
    hypotheses = model.best_hypotheses(n=2)
    predictions = model.predict(individuals=list(p), hypotheses=hypotheses)
    print(predictions)
