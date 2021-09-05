""" Test the default pipeline for structured machine learning"""
import json
import random
from owlapy.model import OWLNamedIndividual, IRI

from ontolearn import KnowledgeBase
from ontolearn.evo_learner import EvoLearner
from ontolearn.utils import setup_logging

random.seed(1)
setup_logging("logging_test.conf")

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
kb = KnowledgeBase(path=settings['data_path'][3:])


class TestEvoLearner:

    def test_regression(self):
        model = EvoLearner(knowledge_base=kb, max_runtime=10)
        regression_test_evolearner = {'Aunt': .9, 'Brother': 1.0,
                                      'Cousin': .78, 'Granddaughter': 1.0,
                                      'Uncle': .85, 'Grandgrandfather': .94}
        for str_target_concept, examples in settings['problems'].items():
            pos = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
            neg = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
            print('Target concept: ', str_target_concept)

            returned_val = model.fit(pos=pos, neg=neg)
            assert returned_val == model
            hypotheses = list(model.best_hypotheses(n=3))
            assert hypotheses[0].quality >= regression_test_evolearner[str_target_concept]
            assert hypotheses[0].quality >= hypotheses[1].quality
            assert hypotheses[1].quality >= hypotheses[2].quality
