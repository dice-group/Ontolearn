""" Test the default pipeline for structured machine learning"""

from ontolearn import *
import json


def test_celoe():
    with open('examples/synthetic_problems.json') as json_file:
        settings = json.load(json_file)
    kb = KnowledgeBase(path=settings['data_path'][3:]) # because '../data/family-benchmark_rich_background.owl'

    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('Target concept: ', str_target_concept)
        model = CELOE(knowledge_base=kb,
                      refinement_operator=ModifiedCELOERefinement(kb=kb),
                      quality_func=F1(),
                      min_horiziontal_expansion=0,
                      heuristic_func=CELOEHeuristic(),
                      search_tree=CELOESearchTree(),
                      terminate_on_goal=True,
                      iter_bound=100,
                      verbose=False)

        model.predict(pos=p, neg=n)
        model.show_best_predictions(top_n=10)

def test_dfoil():
    with open('examples/synthetic_problems.json') as json_file:
        settings = json.load(json_file)

    kb = KnowledgeBase(path=settings['data_path'][3:]) # because '../data/family-benchmark_rich_background.owl'

    for str_target_concept, examples in settings['problems'].items():
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        print('Target concept: ', str_target_concept)
        model = CustomConceptLearner(knowledge_base=kb,
                                     refinement_operator=CustomRefinementOperator(kb=kb),
                                     quality_func=F1(),
                                     heuristic_func=DLFOILHeuristic(),
                                     search_tree=SearchTree(),
                                     terminate_on_goal=True,
                                     iter_bound=1_00,
                                     verbose=True)

        model.predict(pos=p, neg=n)
        model.show_best_predictions(top_n=10)