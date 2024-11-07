import json
import random
import unittest
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.iri import IRI

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.utils import setup_logging

import json
import os
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass

random.seed(1)


class TestEvoLearner(unittest.TestCase):

    def test_regression_family(self):
        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)
        kb = KnowledgeBase(path=settings['data_path'][3:])
        # @TODO: Explicitly define params
        model = EvoLearner(knowledge_base=kb, max_runtime=10)

        regression_test_evolearner = {'Aunt': .90, 'Brother': .89,
                                      'Cousin': 0.90, 'Granddaughter': .89,
                                      'Uncle': 0.79, 'Grandgrandfather': .89}
        for str_target_concept, examples in settings['problems'].items():
            pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
            neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
            lp = PosNegLPStandard(pos=pos, neg=neg)
            returned_model = model.fit(learning_problem=lp)
            assert returned_model == model
            hypotheses = list(returned_model.best_hypotheses(n=3, return_node=True))
            assert hypotheses[0].quality >= regression_test_evolearner[str_target_concept]

    def test_regression_mutagenesis_multiple_fits(self):
        kb = KnowledgeBase(path='KGs/Mutagenesis/mutagenesis.owl')

        namespace_ = 'http://dl-learner.org/mutagenesis#'
        pos_inds = ['d190', 'd191', 'd194', 'd197', 'e1', 'e2', 'e27', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']
        pos = {OWLNamedIndividual(IRI.create(namespace_, ind)) for ind in pos_inds}
        neg_inds = ['d189', 'd192', 'd193', 'd195', 'd196', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16',
                    'e17', 'e18', 'e19', 'e20', 'e21', 'e22', 'e23', 'e24', 'e25', 'e26', 'e3', 'e4', 'e5',
                    'e6', 'e7', 'e8', 'e9']
        neg = {OWLNamedIndividual(IRI.create(namespace_, ind)) for ind in neg_inds}

        lp = PosNegLPStandard(pos=pos, neg=neg)
        model = EvoLearner(knowledge_base=kb, max_runtime=10)
        returned_model = model.fit(learning_problem=lp)
        best_pred = returned_model.best_hypotheses(n=1, return_node=True)
        assert best_pred.quality == 1.00

        returned_model = model.fit(learning_problem=lp)
        best_pred = returned_model.best_hypotheses(n=1, return_node=True)
        assert best_pred.quality == 1.00

    def test_example(self):

        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)

        kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")

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

            model.save_best_hypothesis(n=3, path=f"Predictions_{str_target_concept}")
            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            # Use hypotheses as binary function to label individuals.
            predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                        hypotheses=hypotheses)
            # print(predictions)
            [print(_) for _ in hypotheses]
