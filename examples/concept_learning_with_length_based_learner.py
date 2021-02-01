import json

#from ontolearn import LengthBaseLearner, KnowledgeBase
from ontolearn import KnowledgeBase
from ontolearn.concept_learner import LengthBaseLearner

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
model = LengthBaseLearner(knowledge_base=kb, verbose=1)

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        concepts_to_ignore.update(
            {'http://www.benchmark.org/family#Brother', 'Father', 'Grandparent'})  # Use URI, or concept with length 1.
    model.fit(pos=p, neg=n, ignore=concepts_to_ignore)
    hypotheses = model.best_hypotheses(n=10)
