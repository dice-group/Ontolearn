import json
from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
model = CELOE(knowledge_base=kb, verbose=1, max_runtime=1)
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
    # Get Top n hypotheses
    hypotheses = model.best_hypotheses(n=3)
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(p) + list(n), hypotheses=hypotheses)
    # print(predictions)