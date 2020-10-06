""" Test the default pipeline for structured machine learning"""

import json

from ontolearn import CELOE
from ontolearn import CELOEHeuristic
from ontolearn import CELOESearchTree
from ontolearn import F1
from ontolearn import KnowledgeBase
from ontolearn import ModifiedCELOERefinement


def test_celoeminimal():
    with open('examples/synthetic_problems.json') as json_file:
        settings = json.load(json_file)

    # because '../data/family-benchmark_rich_background.owl'
    kb = KnowledgeBase(path=settings['data_path'][3:])

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

        model.predict(pos=p, neg=n)

        model.show_best_predictions(top_n=10, key='quality')


def test_celoe():
    with open('examples/synthetic_problems.json') as json_file:
        settings = json.load(json_file)

    # because '../data/family-benchmark_rich_background.owl'
    kb = KnowledgeBase(path=settings['data_path'][3:])

    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('Target concept: ', str_target_concept)
        concepts_to_ignore = set()
        # lets inject more background info
        if str_target_concept in ['Granddaughter', 'Aunt', 'Sister']:
            concepts_to_ignore.update(
                {'Brother', 'Father', 'Uncle', 'Grandparent'})

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

        model.predict(pos=p, neg=n)
        model.show_best_predictions(top_n=10_000, key='heuristic')


def test_celoe_predictions():
    with open('examples/synthetic_problems.json') as json_file:
        settings = json.load(json_file)

    # because '../data/family-benchmark_rich_background.owl'
    kb = KnowledgeBase(path=settings['data_path'][3:])

    for str_target_concept, examples in settings['problems'].items():
        if str_target_concept == 'Aunt':
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

            best_pred = model.predict(pos=p, neg=n)
            model.show_best_predictions(top_n=10, key='quality')

            # TODO: The quality should be at least 0.8 as in the previous
            #  version (without the world object)
            assert (best_pred.quality > 0.7)
