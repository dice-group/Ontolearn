""" Test the default pipeline for structured machine learning"""
import json

from pytest import mark

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import OCEL
from ontolearn.utils import setup_logging

setup_logging("logging_test.conf")

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# because '../KGs/Family/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=settings['data_path'][3:])


class TestOcel:
    @mark.xfail(run=False, reason="TODO")
    def test_regression(self):
        regression_test_ocel = {'Aunt': .71429, 'Brother': .96774,
                                'Cousin': .66667, 'Granddaughter': .86047,
                                'Uncle': .67857, 'Grandgrandfather': .94444}
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

            model = OCEL(knowledge_base=kb)

            returned_val = model.fit(pos=p, neg=n, ignore=concepts_to_ignore)
            assert returned_val == model
            hypotheses = list(model.best_hypotheses(n=3))
            assert hypotheses[0].quality >= regression_test_ocel[str_target_concept]
            assert hypotheses[0].quality >= hypotheses[1].quality
            assert hypotheses[1].quality >= hypotheses[2].quality
