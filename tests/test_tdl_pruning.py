"""Tests for ``TDL(hypothesis_extraction="pruned_leaves")``.

This mode is an Ontolearn addition that is NOT part of the published, peer-reviewed TDL algorithm (see
``ontolearn/learners/tdl_pruning.py``). The tests check what it promises: it never changes the training F1 at
``prune_tolerance=0``, never makes the expression longer, and leaves the default behaviour of TDL untouched.
"""
import json
import os
import unittest

import numpy as np

from ontolearn.learners.tdl_pruning import prune_rules, rule_mask

FAMILY_KB = "KGs/Family/family-benchmark_rich_background.owl"
FAMILY_LPS = "LPs/Family/lps.json"


class TestPruneRules(unittest.TestCase):
    # columns: 0 = A, 1 = B, 2 = C. The positives are exactly the rows with A.
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    def test_redundant_literals_are_dropped_without_changing_the_cover(self):
        pruned = prune_rules([((0, True), (1, False)), ((0, True), (1, True))], self.X, self.y)
        self.assertEqual(pruned, [((0, True),)])

    def test_a_literal_that_is_needed_is_kept(self):
        rules = [((0, True), (1, False))]  # covers 2 of the 4 positives and no negative
        self.assertEqual(prune_rules(rules, self.X, self.y), [((0, True),)])  # dropping B only improves F1
        rules = [((1, False), (2, False))]  # covers one positive and one negative
        pruned = prune_rules(rules, self.X, self.y)
        self.assertEqual(len(pruned), 1)
        self.assertGreaterEqual(len(pruned[0]), 1)

    def test_pruning_never_lowers_the_f1_at_zero_tolerance(self):
        rng = np.random.RandomState(0)
        for _ in range(30):
            X = rng.randint(0, 2, size=(40, 6))
            y = rng.randint(0, 2, size=40)
            rules = [tuple((int(c), bool(rng.randint(2))) for c in rng.choice(6, size=3, replace=False))
                     for _ in range(4)]
            pruned = prune_rules(rules, X, y)

            def f1(rs):
                covered = np.any([rule_mask(X, r) for r in rs], axis=0)
                tp = int((covered & (y == 1)).sum())
                return 0.0 if tp == 0 else 2 * tp / (int(covered.sum()) + int((y == 1).sum()))
            self.assertGreaterEqual(f1(pruned) + 1e-12, f1(rules))
            self.assertLessEqual(sum(map(len, pruned)), sum(map(len, rules)))
            self.assertTrue(all(len(r) >= 1 for r in pruned) and len(pruned) >= 1)

    def test_no_positives_returns_the_rules_unchanged(self):
        rules = [((0, True), (1, False))]
        self.assertEqual(prune_rules(rules, self.X, np.zeros(8, dtype=int)), rules)

    def test_tolerance_is_measured_against_the_unpruned_rules(self):
        rules = [((0, True),), ((2, True), (1, True))]
        strict = prune_rules(rules, self.X, self.y, 0.0)
        loose = prune_rules(rules, self.X, self.y, 0.5)
        self.assertLessEqual(sum(map(len, loose)), sum(map(len, strict)))


@unittest.skipUnless(os.path.exists(FAMILY_KB) and os.path.exists(FAMILY_LPS), "Family data not downloaded")
class TestPrunedLeavesOnFamily(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ontolearn.knowledge_base import KnowledgeBase
        cls.kb = KnowledgeBase(path=FAMILY_KB)
        with open(FAMILY_LPS) as f:
            cls.problems = json.load(f)["problems"]

    def fit(self, problem, mode, **kwargs):
        from owlapy.owl_individual import OWLNamedIndividual
        from ontolearn.learners import TDL
        from ontolearn.learning_problem import PosNegLPStandard
        ex = self.problems[problem]
        lp = PosNegLPStandard(pos={OWLNamedIndividual(i) for i in ex["positive_examples"]},
                              neg={OWLNamedIndividual(i) for i in ex["negative_examples"]})
        return TDL(knowledge_base=self.kb, kwargs_classifier={"random_state": 1}, verbose=0,
                   report_classification=False, hypothesis_extraction=mode, **kwargs).fit(lp), lp

    def train_f1(self, model, lp):
        covered = set(self.kb.individuals(model.best_hypotheses(1)))
        tp, fp = len(covered & lp.pos), len(covered & lp.neg)
        fn = len(lp.pos) - tp
        return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

    def test_zero_tolerance_keeps_the_training_f1_and_is_not_longer(self):
        from owlapy import owl_expression_to_dl
        for problem in ("Uncle", "Cousin", "Aunt"):
            base, lp = self.fit(problem, "predicted_leaves")
            pruned, _ = self.fit(problem, "pruned_leaves")
            self.assertAlmostEqual(self.train_f1(pruned, lp), self.train_f1(base, lp), places=9)
            self.assertLessEqual(len(owl_expression_to_dl(pruned.best_hypotheses(1))),
                                 len(owl_expression_to_dl(base.best_hypotheses(1))))

    def test_default_mode_is_unchanged(self):
        from ontolearn.learners import TDL
        self.assertEqual(TDL(knowledge_base=self.kb).hypothesis_extraction, "positive_paths")

    def test_unknown_mode_is_rejected(self):
        from ontolearn.learners import TDL
        with self.assertRaises(ValueError):
            TDL(knowledge_base=self.kb, hypothesis_extraction="nonsense")


if __name__ == "__main__":
    unittest.main()
