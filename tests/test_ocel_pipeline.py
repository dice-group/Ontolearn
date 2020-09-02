""" Test the default pipeline for structured machine learning"""

import json

from ontolearn import CELOESearchTree
from ontolearn import F1
from ontolearn import KnowledgeBase
from ontolearn import ModifiedCELOERefinement
from ontolearn import OCEL
from ontolearn import OCELHeuristic


def test_ocel():
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

        model = OCEL(knowledge_base=kb,
                     refinement_operator=ModifiedCELOERefinement(kb=kb),
                     quality_func=F1(),
                     min_horizontal_expansion=0,
                     heuristic_func=OCELHeuristic(),
                     search_tree=CELOESearchTree(),
                     terminate_on_goal=True,
                     iter_bound=100,
                     ignored_concepts=concepts_to_ignore,
                     verbose=False)

        model.predict(pos=p, neg=n)
        model.show_best_predictions(top_n=10_000, key='heuristic')
