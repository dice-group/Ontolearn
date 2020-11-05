# -*- coding: utf-8 -*-
import json

from owlready2 import sync_reasoner
from ontolearn import CELOE
from ontolearn import CELOEHeuristic
from ontolearn import CELOESearchTree, SearchTreePriorityQueue
from ontolearn import F1, Accuracy
from ontolearn import KnowledgeBase
from ontolearn import ModifiedCELOERefinement, LengthBasedRefinement

PATH_DATA = '../data/father.owl'
settings = {}
settings['data_path'] = PATH_DATA

def test_celoe_father():
    kb = KnowledgeBase(path=settings['data_path'][3:])
    #with (kb.onto):
    #    sync_reasoner()
    #sync_reasoner()

    examples = {
        'positive_examples' : ["http://example.com/father#stefan","http://example.com/father#markus","http://example.com/father#martin"],
        'negative_examples' : ["http://example.com/father#heinz","http://example.com/father#anna","http://example.com/father#michelle"],
    }

    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])
    concepts_to_ignore = set()

    model = CELOE(knowledge_base=kb,
              refinement_operator=ModifiedCELOERefinement(kb=kb),
              quality_func=F1(),
              min_horizontal_expansion=0.02,
              heuristic_func=CELOEHeuristic(),
              search_tree=CELOESearchTree(),
              terminate_on_goal=True,
              iter_bound=137141,
              max_num_of_concepts_tested=3000,
              ignored_concepts=concepts_to_ignore,
              verbose=False)

    model.fit(pos=p, neg=n)
    best_pred = model.best_hypotheses(n=1)[0]
    print(best_pred)
    assert(best_pred.quality == 1.0)
    assert(best_pred.concept.str == '(male  ⊓  (∃hasChild.Thing))')
