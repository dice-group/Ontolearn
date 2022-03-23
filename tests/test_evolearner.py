import json
import random
import unittest
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual, IRI

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.utils import setup_logging

random.seed(1)
setup_logging("ontolearn/logging_test.conf")


class TestEvoLearner(unittest.TestCase):

    def test_regression_family(self):
        with open('examples/synthetic_problems.json') as json_file:
            settings = json.load(json_file)
        kb = KnowledgeBase(path=settings['data_path'][3:])
        model = EvoLearner(knowledge_base=kb, max_runtime=10)

        regression_test_evolearner = {'Aunt': .9, 'Brother': 1.0,
                                      'Cousin': .78, 'Granddaughter': 1.0,
                                      'Uncle': .85, 'Grandgrandfather': .94}
        for str_target_concept, examples in settings['problems'].items():
            pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
            neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
            print('Target concept: ', str_target_concept)

            lp = PosNegLPStandard(pos=pos, neg=neg)
            returned_model = model.fit(learning_problem=lp)
            self.assertEqual(returned_model, model)
            hypotheses = list(returned_model.best_hypotheses(n=3))
            self.assertGreaterEqual(hypotheses[0].quality, regression_test_evolearner[str_target_concept])
            self.assertGreaterEqual(hypotheses[0].quality, hypotheses[1].quality)
            self.assertGreaterEqual(hypotheses[1].quality, hypotheses[2].quality)

    def test_regression_mutagenesis(self):
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
        best_pred = next(returned_model.best_hypotheses(n=1))
        self.assertEqual(best_pred.quality, 1.00)
