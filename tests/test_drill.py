import json
import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold
from ontolearn.utils.static_funcs import compute_f1_score
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.learners import Drill
from ontolearn.metrics import F1
from ontolearn.heuristics import CeloeBasedReward
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer


class TestDrill(unittest.TestCase):

    kg_path = "KGs/Family/family-benchmark_rich_background.owl"
    embeddings_path = "embeddings/Keci_entity_embeddings.csv"
    lp_path = "examples/synthetic_problems.json"

    def test_regression_family(self):
        kb = KnowledgeBase(path=self.kg_path)
        drill = Drill(knowledge_base=kb,
                      path_embeddings=self.embeddings_path,
                      refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                      quality_func=F1(),
                      reward_func=CeloeBasedReward(),
                      epsilon_decay=.01,
                      learning_rate=.01,
                      num_of_sequential_actions=1,
                      num_episode=1,
                      iter_bound=10_000,
                      max_runtime=30)

        with open(self.lp_path) as json_file:
            examples = json.load(json_file)
        p = examples["problems"]["Uncle"]['positive_examples']
        n = examples["problems"]["Uncle"]['negative_examples']

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        X = np.array(p + n)
        Y = np.array([1.0 for _ in p] + [0.0 for _ in n])
        dl_render = DLSyntaxObjectRenderer()
        total_test_f1 = 0.0
        for (ith, (train_index, test_index)) in enumerate(kf.split(X, Y)):
            train_pos = {pos_individual for pos_individual in X[train_index][Y[train_index] == 1]}
            train_neg = {neg_individual for neg_individual in X[train_index][Y[train_index] == 0]}
            test_pos = {pos_individual for pos_individual in X[test_index][Y[test_index] == 1]}
            test_neg = {neg_individual for neg_individual in X[test_index][Y[test_index] == 0]}
            train_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, train_pos))),
                                        neg=set(map(OWLNamedIndividual, map(IRI.create, train_neg))))

            test_lp = PosNegLPStandard(pos=set(map(OWLNamedIndividual, map(IRI.create, test_pos))),
                                       neg=set(map(OWLNamedIndividual, map(IRI.create, test_neg))))

            pred_drill = drill.fit(train_lp).best_hypotheses()
            train_f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                                              pos=train_lp.pos,
                                              neg=train_lp.neg)
            # () Quality on test data
            test_f1_drill = compute_f1_score(individuals=frozenset({i for i in kb.individuals(pred_drill)}),
                                             pos=test_lp.pos,
                                             neg=test_lp.neg)
            total_test_f1 += test_f1_drill
            print(f"Prediction: {dl_render.render(pred_drill)} | Train Quality: {train_f1_drill:.3f} | "
                  f"Test Quality: {test_f1_drill:.3f} \n")
        print(total_test_f1/5)
        assert total_test_f1/5 >= 0.8
