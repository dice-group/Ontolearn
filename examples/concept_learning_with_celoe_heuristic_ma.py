import json
import os
import random

from ontolearn.concept_learner import CELOE
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.model_adapter import ModelAdapter, Trainer
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass
from ontolearn.utils import setup_logging
from owlapy.owl_ontology import Ontology
from owlapy.owl_reasoner import SyncReasoner, BaseReasoner
from typing import cast
setup_logging()

try:
    os.chdir("examples")
except FileNotFoundError:
    pass

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)

random.seed(0)

# noinspection DuplicatedCode
for str_target_concept, examples in settings['problems'].items():
    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    print('Target concept: ', str_target_concept)

    concepts_to_ignore = None
    # let's inject more background info
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

    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))

    kb = KnowledgeBase(path=settings['data_path'])
    reasoner = SyncReasoner(cast(Ontology, kb.ontology()), BaseReasoner.HERMIT)

    model = ModelAdapter(path=settings['data_path'],
                         ignore=concepts_to_ignore,
                         reasoner=reasoner,
                         learner_type=CELOE,
                         max_runtime=5,
                         max_num_of_concepts_tested=10_000_000_000,
                         iter_bound=10_000_000_000,
                         expansionPenaltyFactor=0.01)

    model = model.fit(pos=typed_pos, neg=typed_neg)

    # model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))
    # Get Top n hypotheses
    hypotheses = list(model.best_hypotheses(n=3))
    # Use hypotheses as binary function to label individuals.
    predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                hypotheses=hypotheses)
    # print(predictions)
    [print(_) for _ in hypotheses]

    # Using Trainer
    model2 = CELOE(knowledge_base=kb, max_runtime=5)
    trainer = Trainer(model, reasoner)
    trainer.fit(pos=typed_pos, neg=typed_neg)
    hypotheses = list(model2.best_hypotheses(n=3))
    [print(_) for _ in hypotheses]


