""" Test DRILL"""
# @TODO, write such tests for Drill after, experiments are completed.
"""
#import random
#import pandas as pd

from ontolearn import KnowledgeBase
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.rl import DrillAverage
from ontolearn.utils import setup_logging

setup_logging("logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'

drill_pretrained_model_path = 'agent_pre_trained/model.pth'
family_embeddings_path = 'embeddings/dismult_family_benchmark/instance_emb.csv'
synthetic_problems_path = 'examples/synthetic_problems.json'

with open(synthetic_problems_path) as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(path=PATH_FAMILY)
rho = LengthBasedRefinement(knowledge_base=kb)

instance_emb = pd.read_csv(family_embeddings_path, index_col=0)


class TestDrill:
    def test_drill_regression(self):
        exp_f1_scores = {'Aunt': .80392, 'Brother': 1.0,
                         'Cousin': .72626, 'Granddaughter': 1.0,
                         'Uncle': .88372, 'Grandgrandfather': 0.94444}
        model = DrillAverage(knowledge_base=kb, refinement_operator=rho,
                             terminate_on_goal=True, instance_embeddings=instance_emb)

        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            print('Target concept: ', str_target_concept)
            concepts_to_ignore = set()
            # lets inject more background info
            if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
                concepts_to_ignore.update(
                    {'http://www.benchmark.org/family#Brother', 'Father',
                     'Grandparent'})  # Use URI, or concept with length 1.

            returned_val = model.fit(pos=p, neg=n,ignore=concepts_to_ignore)
            assert returned_val == model
            hypotheses = list(model.best_hypotheses(n=5))
            assert hypotheses[0].quality >= exp_f1_scores[str_target_concept]
            assert hypotheses[0].quality >= hypotheses[1].quality
"""
