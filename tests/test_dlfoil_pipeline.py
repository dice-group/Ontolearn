""" Test the default pipeline for structured machine learning"""

import json

from ontolearn import CustomConceptLearner
from ontolearn import CustomRefinementOperator
from ontolearn import DLFOILHeuristic
from ontolearn import F1
from ontolearn import KnowledgeBase
from ontolearn import SearchTree

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# because '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=settings['data_path'][3:])


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