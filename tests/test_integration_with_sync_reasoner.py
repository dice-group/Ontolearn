""" Test for checking integration with SyncReasoner"""
import unittest

import numpy as np
from owlapy.class_expression import OWLThing
from owlapy.owl_property import OWLObjectProperty
from owlapy.owl_reasoner import SyncReasoner
from sklearn.model_selection import StratifiedKFold

from ontolearn.heuristics import CeloeBasedReward
from ontolearn.concept_learner import CELOE, EvoLearner
from ontolearn.metrics import F1
from ontolearn.quality_funcs import evaluate_concept
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.utils import compute_f1_score
import json
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import Drill
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass

PATH_FAMILY = 'KGs/Family/family-benchmark_rich_background.owl'

class TestIntegrationWithSyncReasoner(unittest.TestCase):

    def setUp(self):
        self.kb = KnowledgeBase(path=PATH_FAMILY,reasoner=SyncReasoner(PATH_FAMILY, "Pellet"))
        self.hasChild = OWLObjectProperty("http://www.benchmark.org/family#hasChild")
        self.father = OWLClass("http://www.benchmark.org/family#Father")
        with open('LPs/Family/lps.json') as json_file:
            settings = json.load(json_file)
        self.p = settings['problems']['Aunt']['positive_examples']
        self.n = settings['problems']['Aunt']['negative_examples']
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, self.p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, self.n)))
        self.lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        self.embeddings_path = "embeddings/Keci_entity_embeddings.csv"

    def test_celoe(self):
        model = CELOE(knowledge_base=self.kb, max_runtime=30)
        hypo = model.fit(self.lp).best_hypotheses()
        self.assertGreater(evaluate_concept(self.kb, hypo, F1() ,self.lp.encode_kb(self.kb)).q, 0.65)

    def test_evolearner(self):
        model = EvoLearner(knowledge_base=self.kb, max_runtime=30)
        hypo = model.fit(self.lp).best_hypotheses()
        self.assertGreater(evaluate_concept(self.kb, hypo, F1() ,self.lp.encode_kb(self.kb)).q, 0.8)

    def test_drill(self):
        model = Drill(knowledge_base=self.kb,
                      path_embeddings=self.embeddings_path,
                      refinement_operator=LengthBasedRefinement(knowledge_base=self.kb),
                      quality_func=F1,
                      reward_func=CeloeBasedReward(),
                      epsilon_decay=.01,
                      learning_rate=.01,
                      num_of_sequential_actions=1,
                      num_episode=1,
                      iter_bound=10_000,
                      max_runtime=30)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        X = np.array(self.p + self.n)
        Y = np.array([1.0 for _ in self.p] + [0.0 for _ in self.n])
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

            pred_drill = model.fit(train_lp).best_hypotheses()
            # () Quality on test data
            test_f1_drill = compute_f1_score(individuals=frozenset({i for i in self.kb.individuals(pred_drill)}),
                                             pos=test_lp.pos,
                                             neg=test_lp.neg)
            total_test_f1 += test_f1_drill
        self.assertGreater(total_test_f1 / 5, 0.5)

    def test_kb_methods(self):
        # Checking only for error-free execution (just some random method calls)
        print(self.kb.individuals())
        print(self.kb.triples())
        print(self.kb.most_general_object_properties(domain=OWLThing))
        print(self.kb.get_object_property_domains(self.hasChild))
        print(self.kb.get_object_property_ranges(self.hasChild))
        print(self.kb.get_all_direct_sub_concepts(OWLThing))
        print(self.kb.contains_class(self.father))