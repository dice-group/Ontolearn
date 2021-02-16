import json
import random

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.owlapy import IRI
from ontolearn.owlapy.model import OWLClass, OWLNamedIndividual
from ontolearn.owlapy.render import DLSyntaxRenderer
from ontolearn.search import OENode

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])

random.seed(0)

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
             OWLClass(IRI(NS, 'Sister')),
             OWLClass(IRI(NS, 'Daughter')),
             OWLClass(IRI(NS, 'Mother')),
             OWLClass(IRI(NS, 'Grandmother')),
             OWLClass(IRI(NS, 'Father')),
             OWLClass(IRI(NS, 'Grandparent')),
             })  # Use OWLClass

    typed_pos = list(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = list(map(OWLNamedIndividual, map(IRI.create, n)))
    model = CELOE(knowledge_base=kb, max_runtime=600, verbose=2, max_num_of_concepts_tested=10_000_000_000, iter_bound=10_000_000_000)
    model.fit(pos=set(typed_pos),
              neg=set(typed_neg),
              ignore=concepts_to_ignore)

    # model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    # Get Top n hypotheses
    hypotheses = list(model.best_hypotheses(n=3))
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=typed_pos + typed_neg, hypotheses=hypotheses)
    # print(predictions)
    [print(_) for _ in hypotheses]
    exit(1)
