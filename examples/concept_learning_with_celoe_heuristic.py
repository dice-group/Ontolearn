import json
from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.owlapy import IRI
from ontolearn.owlapy.model import OWLClass, OWLNamedIndividual

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
model = CELOE(knowledge_base=kb, max_runtime=1)

# noinspection DuplicatedCode
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)
    concepts_to_ignore = set()
    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        NS = 'http://www.benchmark.org/family#'
        concepts_to_ignore.update(
            {OWLClass(IRI(NS, 'Brother')),
             OWLClass(IRI(NS, 'Father')),
             OWLClass(IRI(NS, 'Grandparent')),
             })  # Use OWLClass
    model.fit(pos=set(map(OWLNamedIndividual, map(IRI.create, p))),
              neg=set(map(OWLNamedIndividual, map(IRI.create, n))),
              ignore=concepts_to_ignore)
    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    # Get Top n hypotheses
    hypotheses = model.best_hypotheses(n=3)
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(p) + list(n), hypotheses=hypotheses)
    exit(1)
