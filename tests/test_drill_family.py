""" Test DRILL"""
from ontolearn import *
import json
import random
import pandas as pd

PATH_FAMILY = 'data/family-benchmark_rich_background.owl'
drill_pretrained_model_path = 'agent_pre_trained/model.pth'
family_embeddings_path = 'embeddings/dismult_family_benchmark/instance_emb.csv'
synthetic_problems_path = 'examples/synthetic_problems.json'

with open(synthetic_problems_path) as json_file:
    settings = json.load(json_file)

kb = KnowledgeBase(PATH_FAMILY)
rho = LengthBasedRefinement(kb=kb)

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
            hypotheses = model.best_hypotheses(n=5)
            assert hypotheses[0].quality >= exp_f1_scores[str_target_concept]
            assert hypotheses[0].quality >= hypotheses[1].quality
