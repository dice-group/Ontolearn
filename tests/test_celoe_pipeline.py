""" Test the default pipeline for structured machine learning"""

import json

from ontolearn import CELOE
from ontolearn import CELOEHeuristic
from ontolearn import CELOESearchTree
from ontolearn import F1
from ontolearn import KnowledgeBase
from ontolearn import ModifiedCELOERefinement

with open('examples/synthetic_problems.json') as json_file:
    settings = json.load(json_file)
# because '../data/family-benchmark_rich_background.owl'
kb = KnowledgeBase(path=settings['data_path'][3:])


def test_celoeminimal():
    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('Target concept: ', str_target_concept)
        model = CELOE(knowledge_base=kb,
                      refinement_operator=ModifiedCELOERefinement(kb=kb),
                      quality_func=F1(),
                      min_horizontal_expansion=0,
                      heuristic_func=CELOEHeuristic(),
                      search_tree=CELOESearchTree(),
                      terminate_on_goal=True,
                      iter_bound=100,
                      verbose=False)
        model.fit(pos=p, neg=n)


def test_celoe():
    with open('examples/synthetic_problems.json') as json_file:
        settings = json.load(json_file)


    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('Target concept: ', str_target_concept)
        concepts_to_ignore = set()
        # lets inject more background info
        if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
            concepts_to_ignore.update(
                {'Brother', 'Father', 'Grandparent'})

        model = CELOE(knowledge_base=kb,
                      refinement_operator=ModifiedCELOERefinement(kb=kb),
                      quality_func=F1(),
                      min_horizontal_expansion=0,
                      heuristic_func=CELOEHeuristic(),
                      search_tree=CELOESearchTree(),
                      terminate_on_goal=True,
                      iter_bound=100,
                      ignored_concepts=concepts_to_ignore,
                      verbose=False)

        #model.fit(pos=p, neg=n)
