# -*- coding: utf-8 -*-
import json
import types

from owlready2 import sync_reasoner, owl
from ontolearn import CELOE
from ontolearn import CELOEHeuristic
from ontolearn import CELOESearchTree, SearchTreePriorityQueue
from ontolearn import F1, Accuracy
from ontolearn import KnowledgeBase
from ontolearn import ModifiedCELOERefinement, LengthBasedRefinement
from ontolearn.export import export_concepts

PATH_DATA = '../data/father.owl'
settings = {}
settings['data_path'] = PATH_DATA

def test_father_serialization():
    kb = KnowledgeBase(path=settings['data_path'][3:])

    examples = {
        'positive_examples' : ["http://example.com/father#stefan","http://example.com/father#markus","http://example.com/father#martin"],
        'negative_examples' : ["http://example.com/father#heinz","http://example.com/father#anna","http://example.com/father#michelle"],
    }

    p = set(examples['positive_examples'])
    n = set(examples['negative_examples'])

    model = CELOE(knowledge_base=kb, terminate_on_goal=True)

    model.fit(pos=p, neg=n)
    best_n_pred = model.best_hypotheses(n=3)
    print(best_n_pred)

    export_concepts(best_n_pred, file_path = "father_predictions.owl")

