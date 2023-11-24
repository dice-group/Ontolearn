import json
import os

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging

setup_logging()

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])


# noinspection DuplicatedCode
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)

    # lets inject more background info
    if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
        NS = 'http://www.benchmark.org/family#'
        concepts_to_ignore = {
            OWLClass(IRI(NS, 'Brother')),
            OWLClass(IRI(NS, 'Sister')),
            OWLClass(IRI(NS, 'Daughter')),
            OWLClass(IRI(NS, 'Mother')),
            OWLClass(IRI(NS, 'Grandmother')),
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
            OWLClass(IRI(NS, 'PersonWithASibling')),
            OWLClass(IRI(NS, 'Granddaughter')),
            OWLClass(IRI(NS, 'Son')),
            OWLClass(IRI(NS, 'Child')),
            OWLClass(IRI(NS, 'Grandson')),
            OWLClass(IRI(NS, 'Grandfather')),
            OWLClass(IRI(NS, 'Grandchild')),
            OWLClass(IRI(NS, 'Parent')),
        }
        target_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
    else:
        target_kb = kb

    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

    model = EvoLearner(knowledge_base=target_kb, max_runtime=600)
    model.fit(lp, verbose=False)

    model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    # Get Top n hypotheses
    hypotheses = list(model.best_hypotheses(n=3))
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                hypotheses=hypotheses)
    # print(predictions)
    [print(_) for _ in hypotheses]
