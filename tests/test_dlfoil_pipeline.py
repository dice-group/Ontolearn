""" Test the default pipeline for structured machine learning"""

import json

from ontolearn import CustomConceptLearner, KnowledgeBase, F1

PATH_FAMILY = 'data/family-benchmark_rich_background.owl'
with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
kb = KnowledgeBase(path=PATH_FAMILY)


def test_dfoil():
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('Target concept: ', str_target_concept)
        model = CustomConceptLearner(
            knowledge_base=kb,
            quality_func=F1(),
            terminate_on_goal=True,
            iter_bound=1_00,
            verbose=True)

        returned_param = model.fit(pos=p, neg=n)
        assert returned_param == model
