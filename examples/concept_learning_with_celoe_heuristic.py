import json
from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE

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
             'Father', 'http://www.benchmark.org/family#Grandparent'})  # Use URI, or concept with length 1.

    model = CELOE(knowledge_base=kb,
                  ignored_concepts=concepts_to_ignore,max_num_of_concepts_tested=10)
    model.fit(pos=p, neg=n)
    hypotheses = model.best_hypotheses(n=2)
    print(hypotheses[0])
    predictions = model.predict(individuals=list(p), hypotheses=hypotheses)
    print(predictions)
