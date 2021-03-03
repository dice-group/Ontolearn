import json
import os

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import OCEL

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
model = OCEL(knowledge_base=kb, verbose=1)
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'http://www.benchmark.org/family#Brother',
             'Father', 'http://www.benchmark.org/family#Grandparent'})  # Use URI, or concept with length 1.
    model.fit(pos=p, neg=n, ignore=concepts_to_ignore)
    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    hypotheses = model.best_hypotheses(n=1)
    print(hypotheses[0])
