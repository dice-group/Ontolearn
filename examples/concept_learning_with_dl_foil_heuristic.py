import json
from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CustomConceptLearner

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    model = CustomConceptLearner(knowledge_base=kb, verbose=1)
    model.fit(pos=p, neg=n)
    hypotheses = model.best_hypotheses(n=2)
    print(hypotheses[0])
