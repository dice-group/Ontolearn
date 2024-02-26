import json
import time

import numpy as np
import pandas as pd
from owlapy.model import IRI, OWLNamedIndividual
from sklearn.model_selection import StratifiedKFold

from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.triple_store import TripleStoreKnowledgeBase
from ontolearn.utils.static_funcs import compute_f1_score

with open('synthetic_problems.json') as json_file:
    settings = json.load(json_file)


# See our guide on how to load and launch a triplestore server:
# https://ontolearn-docs-dice-group.netlify.app/usage/06_concept_learners#loading-and-launching-a-triplestore


kb = TripleStoreKnowledgeBase("http://localhost:3030/family/sparql")

tdl = TDL(knowledge_base=kb,
          dataframe_triples=pd.DataFrame(
              data=sorted([(t[0], t[1], t[2]) for t in kb.triples(mode='iri')], key=lambda x: len(x)),
              columns=['subject', 'relation', 'object'], dtype=str),
          kwargs_classifier={"random_state": 0},
          max_runtime=15)


data = dict()
for str_target_concept, examples in settings['problems'].items():
    print('Target concept: ', str_target_concept)
    p = examples['positive_examples']
    n = examples['negative_examples']

    # 5 splits by default for each lp
    kf = StratifiedKFold(shuffle=True)
    X = np.array(p + n)
    y = np.array([1.0 for _ in p] + [0.0 for _ in n])

    for (ith, (train_index, test_index)) in enumerate(kf.split(X, y)):

        data.setdefault("LP", []).append(str_target_concept)
        data.setdefault("Fold", []).append(ith)
        # () Extract positive and negative examples from train fold
        train_pos = {pos_individual for pos_individual in X[train_index][y[train_index] == 1]}
        train_neg = {neg_individual for neg_individual in X[train_index][y[train_index] == 0]}

        # Sanity checking for individuals used for training.
        assert train_pos.issubset(examples['positive_examples'])
        assert train_neg.issubset(examples['negative_examples'])

        # () Extract positive and negative examples from test fold
        test_pos = {pos_individual for pos_individual in X[test_index][y[test_index] == 1]}
        test_neg = {neg_individual for neg_individual in X[test_index][y[test_index] == 0]}

        # Sanity checking for individuals used for testing.
        assert test_pos.issubset(examples['positive_examples'])
        assert test_neg.issubset(examples['negative_examples'])
        train_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, train_pos))),
                                    neg=set(map(OWLNamedIndividual, map(IRI.create, train_neg))))

        test_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, test_pos))),
                                   neg=set(map(OWLNamedIndividual, map(IRI.create, test_neg))))
        start_time = time.time()
        # () Fit model training dataset
        pred_tdl = tdl.fit(train_lp).best_hypotheses(n=1)
        print("TDL ends..", end="\t")
        rt_tdl = time.time() - start_time

        # () Quality on the training data
        train_f1_tdl = compute_f1_score(individuals={i for i in kb.individuals(pred_tdl)},
                                        pos=train_lp.pos,
                                        neg=train_lp.neg)
        # () Quality on test data
        test_f1_tdl = compute_f1_score(individuals={i for i in kb.individuals(pred_tdl)},
                                       pos=test_lp.pos,
                                       neg=test_lp.neg)

        data.setdefault("Train-F1-TDL", []).append(train_f1_tdl)
        data.setdefault("Test-F1-TDL", []).append(test_f1_tdl)
        data.setdefault("RT-TDL", []).append(rt_tdl)
        print(f"TDL Train Quality: {train_f1_tdl:.3f}", end="\t")
        print(f"TDL Test Quality: {test_f1_tdl:.3f}", end="\t")
        print(f"TDL Runtime: {rt_tdl:.3f}")
