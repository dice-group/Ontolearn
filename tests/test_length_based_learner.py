""" Test the default pipeline for structured machine learning"""

import json
from functools import partial
from operator import concat

from ontolearn import KnowledgeBase
from ontolearn.concept_learner import LengthBaseLearner
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.metrics import F1
from ontolearn.model_adapter import ModelAdapter
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.search import SearchTreePriorityQueue
from ontolearn.utils import setup_logging
from owlapy.model import OWLNamedIndividual, OWLClass, IRI

setup_logging("logging_test.conf")

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'
with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
kb = KnowledgeBase(path=PATH_FAMILY)


def test_lengthbasedlearner():
    for str_target_concept, examples in settings['problems'].items():
        p = set(map(OWLNamedIndividual, map(IRI.create, set(examples['positive_examples']))))
        n = set(map(OWLNamedIndividual, map(IRI.create, set(examples['negative_examples']))))
        print('Target concept: ', str_target_concept)
        concepts_to_ignore = set()
        # lets inject more background info
        if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
            concepts_to_ignore.update(
                map(OWLClass, map(IRI.create, map(partial(concat, "http://www.benchmark.org/family#"),
                                                  {'Brother', 'Father', 'Grandparent'}))))
            model = ModelAdapter(
                learner_type=LengthBaseLearner,
                knowledge_base=kb,
                refinement_operator_type=LengthBasedRefinement,
                quality_type=F1,
                min_length=1,
                heuristic_type=CELOEHeuristic,
                search_tree_type=SearchTreePriorityQueue,
                terminate_on_goal=True,
                iter_bound=1_000,
                max_num_of_concepts_tested=5_000,
                ignored_concepts=concepts_to_ignore,
                verbose=True)

            returned_val = model.fit(pos=p, neg=n)
            assert returned_val == model


if __name__ == '__main__':
    test_lengthbasedlearner()
