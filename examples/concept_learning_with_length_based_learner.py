import json
import os

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import LengthBaseLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils import setup_logging
from owlapy.model import OWLClass, IRI, OWLNamedIndividual

setup_logging()

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=settings['data_path'])
model = LengthBaseLearner(knowledge_base=kb)

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
            OWLClass(IRI(NS, 'Father')),
            OWLClass(IRI(NS, 'Grandparent')),
        }
        target_kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
    else:
        target_kb = kb
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(knowledge_base=kb, pos=typed_pos, neg=typed_neg)
    model.fit(lp)
    hypotheses = model.best_hypotheses(n=10)
