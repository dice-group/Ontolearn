"""Regression tests for the ``hypothesis_extraction`` modes of TDL.

The three modes turn the same trained decision tree into a class expression (see ``HYPOTHESIS_EXTRACTION_MODES`` in
``ontolearn/learners/tree_learner.py`` for a drawn example):

- ``positive_paths``  : union of the paths of the positive training examples (original behaviour, default)
- ``predicted_leaves``: union of the leaves that predict the positive class
- ``factored_tree``   : the same predictions as ``predicted_leaves`` in a shorter, factored notation

Quality is the F1 score on held-out individuals. The expected scores below were measured with this code and are
pinned (to 3 decimals) so that a change of the extraction logic shows up as a test failure. Sections:

1. the worked example of the documentation, built by hand;
2. a synthetic KB (self-contained): noisy training labels, true labels for testing;
3. the Family benchmark (needs ``KGs/Family`` and ``LPs/Family``): 5-fold cross-validation on three problems;
4. the Carcinogenesis benchmark (~10 minutes, opt in with ``TDL_SLOW_TESTS=1``).
"""
import json
import os
import random
import tempfile
import unittest

import numpy as np
import pandas as pd
from owlapy import owl_expression_to_dl
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from sklearn import tree
from sklearn.model_selection import StratifiedKFold

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL
from ontolearn.learners.tree_learner import HYPOTHESIS_EXTRACTION_MODES
from ontolearn.learning_problem import PosNegLPStandard

NS = "http://example.com/tdl#"
# The three modes of the published TDL; "pruned_leaves" is an Ontolearn addition with its own tests (test_tdl_pruning.py)
MODES = HYPOTHESIS_EXTRACTION_MODES[:3]
POSITIVE_PATHS, PREDICTED_LEAVES, FACTORED_TREE = MODES


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def f1(kb, expression, pos, neg):
    """F1 of `expression` as a classifier of `pos` against `neg` (sets of IRI strings), using KB retrieval."""
    covered = {i.str for i in kb.individuals_set(expression)}
    tp, fp, fn = len(covered & set(pos)), len(covered & set(neg)), len(set(pos) - covered)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def fit(kb, pos, neg, mode, kwargs_classifier=None, **tdl_kwargs):
    """Fit TDL with the given extraction mode and return the learner."""
    model = TDL(knowledge_base=kb, kwargs_classifier={"random_state": 1, **(kwargs_classifier or {})}, verbose=0,
                report_classification=False, hypothesis_extraction=mode, **tdl_kwargs)
    return model.fit(PosNegLPStandard(pos={OWLNamedIndividual(i) for i in pos},
                                      neg={OWLNamedIndividual(i) for i in neg}))


def evaluate(kb, pos, neg, test, mode, kwargs_classifier=None):
    """Return (F1 on `test`, size of the hypothesis as the length of its DL string)."""
    h = fit(kb, pos, neg, mode, kwargs_classifier).best_hypotheses()
    return f1(kb, h, *test), len(owl_expression_to_dl(h))


def cross_validate(kb, pos, neg, kwargs_classifier=None, folds=5):
    """5-fold stratified CV (seed 0). Returns {mode: (mean held-out F1, mean size)}.

    Per fold the tree is trained once; the three modes then extract their hypothesis from the very same tree
    (a paired comparison, and three times cheaper than refitting).
    """
    pos, neg = np.array(pos), np.array(neg)
    examples = np.concatenate([pos, neg])
    labels = np.array([1] * len(pos) + [0] * len(neg))
    scores = {m: [] for m in MODES}
    sizes = {m: [] for m in MODES}
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=0).split(examples, labels):
        train_pos, train_neg = examples[train][labels[train] == 1], examples[train][labels[train] == 0]
        test_pos, test_neg = set(examples[test][labels[test] == 1]), set(examples[test][labels[test] == 0])
        model = fit(kb, train_pos, train_neg, MODES[0], kwargs_classifier)
        for m in MODES:
            model.hypothesis_extraction = m
            model._finalize_predictions(model.X, model.y)
            h = model.best_hypotheses()
            scores[m].append(f1(kb, h, test_pos, test_neg))
            sizes[m].append(len(owl_expression_to_dl(h)))
    return {m: (float(np.mean(scores[m])), float(np.mean(sizes[m]))) for m in scores}


def report(title, results):
    print(f"\n{title}")
    for m, (score, size) in results.items():
        print(f"  {m:17s} F1={score:.3f} size={size:.0f}")


# ----------------------------------------------------------------------------------------------------------------------
# 1. The worked example of the documentation
# ----------------------------------------------------------------------------------------------------------------------
class TestWorkedExample(unittest.TestCase):
    """Reproduces the drawn tree: root C; "no" -> B with leaves (5/0)+ and (1/3)-; "yes" -> leaf (8/0)+.

    Leaf labels are #positives/#negatives.
    """

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        path = os.path.join(cls.dir.name, "kb.owl")
        write_kb(path)  # only needed for a KnowledgeBase providing classes A, B, C and Thing/Nothing
        cls.kb = KnowledgeBase(path=path)
        A, B, C = (OWLClass(IRI(NS, n)) for n in "ABC")
        # rows: (A, B, C, label)
        rows = [(0, 0, 0, 1)] * 5 + [(0, 1, 0, 1)] * 1 + [(0, 1, 0, 0)] * 3 + [(0, 1, 1, 1)] * 4 + [(1, 1, 1, 1)] * 4
        data = np.array(rows)
        cls.X = pd.DataFrame(data[:, :3], columns=[A, B, C])
        cls.y = pd.DataFrame(data[:, 3:], columns=["label"], index=range(len(rows)))
        cls.X.index = cls.y.index = [OWLNamedIndividual(IRI(NS, f"e{i}")) for i in range(len(rows))]

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def hypothesis(self, mode):
        model = TDL(knowledge_base=self.kb, hypothesis_extraction=mode, verbose=0)
        model.clf = tree.DecisionTreeClassifier(random_state=1).fit(self.X.values, self.y.values.ravel())
        model.X, model.y = self.X, self.y
        # The tree must look like the drawing: C at the root, B below its "no" branch, a pure positive "yes" leaf.
        self.assertEqual(model.clf.tree_.feature[0], 2)
        model._finalize_predictions(self.X, self.y)
        self.model = model
        return model.disjunction_of_conjunctive_concepts

    def dl(self, mode):
        return owl_expression_to_dl(self.hypothesis(mode))

    def paths(self, mode):
        """DL strings of the conjunctions that make up the hypothesis."""
        self.hypothesis(mode)
        return [owl_expression_to_dl(c) for c in self.model.conjunctive_concepts]

    def test_positive_paths_keeps_the_impure_leaf(self):
        # The (1/3) leaf holds a positive example, so its path ¬C ⊓ B is part of the hypothesis.
        self.assertEqual(sorted(self.paths(POSITIVE_PATHS)), sorted(["(¬C) ⊓ (¬B)", "(¬C) ⊓ B", "C"]))

    def test_predicted_leaves_drops_the_impure_leaf(self):
        self.assertEqual(self.paths(PREDICTED_LEAVES), ["(¬C) ⊓ (¬B)", "C"])

    def test_factored_tree_is_factored(self):
        # (¬C ⊓ ¬B) ⊔ C  ->  C ⊔ ¬B  (absorption); the complement ¬(¬C ⊓ B) is longer and not used.
        self.assertEqual(self.dl(FACTORED_TREE), "C ⊔ (¬B)")

    def test_hypotheses_are_equivalent_where_documented(self):
        # predicted_leaves and factored_tree are the same function of (B, C): C or not B.
        for b in (0, 1):
            for c in (0, 1):
                expected = bool(c or not b)
                row = {"C": c, "B": b}
                self.assertEqual(self.evaluate_on_features(self.dl(PREDICTED_LEAVES), row), expected)
                self.assertEqual(self.evaluate_on_features(self.dl(FACTORED_TREE), row), expected)

    @staticmethod
    def evaluate_on_features(dl_string, row):
        """Evaluate a DL string over atomic classes B, C as a propositional formula."""
        formula = dl_string.replace("⊓", " and ").replace("⊔", " or ").replace("¬", " not ")
        return bool(eval(formula, {}, row))


# ----------------------------------------------------------------------------------------------------------------------
# 2. Synthetic knowledge base
# ----------------------------------------------------------------------------------------------------------------------
def write_kb(path, n=200, n_train=60, noise=0.1, seed=0, target=lambda a, b, c, d: (a and not b) or c):
    """Write a KB of `n` persons typed by combinations of the classes A, B, C, D.

    The target concept is ``(A and not B) or C``. The first `n_train` persons are the training set; a fraction
    `noise` of their labels is flipped. The remaining persons keep their true labels for testing.
    Returns (train_pos, train_neg, test_pos, test_neg) as lists of IRI strings.
    """
    rng = random.Random(seed)
    lines = ['<?xml version="1.0"?>',
             f'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
             f'xmlns:owl="http://www.w3.org/2002/07/owl#" xml:base="{NS}" xmlns="{NS}">',
             '<owl:Ontology rdf:about="http://example.com/tdl"/>']
    lines += [f'<owl:Class rdf:about="{NS}{c}"/>' for c in "ABCD"]
    train_pos, train_neg, test_pos, test_neg = [], [], [], []
    for i in range(n):
        flags = [rng.random() < 0.5 for _ in range(4)]
        label = bool(target(*flags))
        is_train = i < n_train
        if is_train and rng.random() < noise:
            label = not label
        bucket = (train_pos if label else train_neg) if is_train else (test_pos if label else test_neg)
        bucket.append(f"{NS}p{i}")
        types = "".join(f'<rdf:type rdf:resource="{NS}{c}"/>' for c, f in zip("ABCD", flags) if f)
        lines.append(f'<owl:NamedIndividual rdf:about="{NS}p{i}">{types}</owl:NamedIndividual>')
    lines.append("</rdf:RDF>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return train_pos, train_neg, test_pos, test_neg


class TestSyntheticKB(unittest.TestCase):
    """Held-out F1 on a KB with noisy training labels. Expected values (positive_paths, predicted_leaves):

    ==================  ==============  ================
    tree                positive_paths  predicted_leaves
    ==================  ==============  ================
    max_depth=1             0.809           0.895
    max_depth=2             0.931           0.931
    min_samples_leaf=8      0.931           0.931
    fully grown, noisy      1.000           0.962
    ==================  ==============  ================

    factored_tree always equals predicted_leaves in F1.
    """

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.dir.name, "kb.owl")
        cls.pos, cls.neg, cls.test_pos, cls.test_neg = write_kb(cls.path)
        cls.kb = KnowledgeBase(path=cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def scores(self, **kwargs_classifier):
        out = {m: evaluate(self.kb, self.pos, self.neg, (self.test_pos, self.test_neg), m, kwargs_classifier)
               for m in MODES}
        report(f"synthetic KB {kwargs_classifier or 'fully grown tree'}", out)
        return out

    def check(self, expected_positive_paths, expected_predicted_leaves, **kwargs_classifier):
        out = self.scores(**kwargs_classifier)
        self.assertAlmostEqual(out[POSITIVE_PATHS][0], expected_positive_paths, places=3)
        self.assertAlmostEqual(out[PREDICTED_LEAVES][0], expected_predicted_leaves, places=3)
        self.assertAlmostEqual(out[FACTORED_TREE][0], expected_predicted_leaves, places=3)
        self.assertLessEqual(out[FACTORED_TREE][1], out[PREDICTED_LEAVES][1])

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            TDL(knowledge_base=self.kb, hypothesis_extraction="nope")

    def test_default_is_original_behaviour(self):
        self.assertEqual(TDL(knowledge_base=self.kb).hypothesis_extraction, POSITIVE_PATHS)

    def test_depth_1_impure_leaves(self):
        # positive_paths returns C ⊔ ¬C (everything); the leaf-based modes return C.
        self.check(0.809, 0.895, max_depth=1)

    def test_depth_2(self):
        self.check(0.931, 0.931, max_depth=2)

    def test_factored_tree_prefers_a_shorter_complement(self):
        # Fully grown noisy tree: many positive leaves, few negative ones, so ¬(negative region) is the shorter notation.
        h = fit(self.kb, self.pos, self.neg, FACTORED_TREE).best_hypotheses()
        self.assertTrue(owl_expression_to_dl(h).startswith("¬("), owl_expression_to_dl(h))

    def test_min_samples_leaf(self):
        self.check(0.931, 0.931, min_samples_leaf=8)

    def test_fully_grown_noisy_tree(self):
        # Here the recall-oriented positive_paths is better: it keeps leaves holding a single (noisy) positive.
        self.check(1.000, 0.962)

    def test_noise_free_fully_grown_tree_modes_agree(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "kb.owl")
            pos, neg, test_pos, test_neg = write_kb(path, noise=0.0)
            kb = KnowledgeBase(path=path)
            out = {m: evaluate(kb, pos, neg, (test_pos, test_neg), m) for m in MODES}
            for m in (PREDICTED_LEAVES, FACTORED_TREE):
                self.assertAlmostEqual(out[m][0], out[POSITIVE_PATHS][0], places=9)
            self.assertLessEqual(out[FACTORED_TREE][1], out[POSITIVE_PATHS][1])

    def test_single_leaf_tree(self):
        # All examples positive: informative features exist but the tree is a single positive leaf -> Thing.
        # (positive_paths cannot handle this: every path is empty.)
        for m in (PREDICTED_LEAVES, FACTORED_TREE):
            h = fit(self.kb, self.pos + self.neg, [], m).best_hypotheses()
            self.assertEqual(h, self.kb.generator.thing, msg=m)

    def test_no_positive_leaf(self):
        for m in (PREDICTED_LEAVES, FACTORED_TREE):
            h = fit(self.kb, [], self.pos + self.neg, m).best_hypotheses()
            self.assertEqual(h, self.kb.generator.nothing, msg=m)

    def test_deterministic_output(self):
        for m in (PREDICTED_LEAVES, FACTORED_TREE):
            first = fit(self.kb, self.pos, self.neg, m).best_hypotheses()
            second = fit(self.kb, self.pos, self.neg, m).best_hypotheses()
            self.assertEqual(first, second)


# ----------------------------------------------------------------------------------------------------------------------
# 3. Family benchmark
# ----------------------------------------------------------------------------------------------------------------------
FAMILY_KB = "KGs/Family/family-benchmark_rich_background.owl"
FAMILY_LPS = "LPs/Family/lps.json"


@unittest.skipUnless(os.path.exists(FAMILY_KB) and os.path.exists(FAMILY_LPS), "Family data not downloaded")
class TestFamily(unittest.TestCase):
    """5-fold CV mean held-out F1 (positive_paths, predicted_leaves = factored_tree).

    =========  ==========================  ==========================
    problem    full tree                   max_depth=3
    =========  ==========================  ==========================
    Aunt       0.928 / 0.928               0.804 / 0.873
    Cousin     0.777 / 0.777               0.670 / 0.640
    Uncle      0.906 / 0.906               0.893 / 0.879
    =========  ==========================  ==========================

    Over all 18 problems the means are 0.966 for all modes (full tree) and 0.949 / 0.951 / 0.951 (depth 3);
    factored_tree is ~35% shorter than the other modes on a fully grown tree (mean 73 vs 113 characters).
    """

    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase(path=FAMILY_KB)
        cls.problems = json.load(open(FAMILY_LPS))["problems"]

    def check(self, problem, kwargs_classifier, expected_positive_paths, expected_leaf_based):
        lp = self.problems[problem]
        out = cross_validate(self.kb, lp["positive_examples"], lp["negative_examples"], kwargs_classifier)
        report(f"Family {problem} {kwargs_classifier or 'full tree'}", out)
        self.assertAlmostEqual(out[POSITIVE_PATHS][0], expected_positive_paths, places=3)
        self.assertAlmostEqual(out[PREDICTED_LEAVES][0], expected_leaf_based, places=3)
        self.assertAlmostEqual(out[FACTORED_TREE][0], expected_leaf_based, places=3)
        self.assertLessEqual(out[FACTORED_TREE][1], out[PREDICTED_LEAVES][1])

    def test_aunt_full(self):
        self.check("Aunt", {}, 0.9277, 0.9277)

    def test_aunt_depth3(self):
        self.check("Aunt", {"max_depth": 3}, 0.8044, 0.8728)  # leaf-based is better

    def test_cousin_full(self):
        self.check("Cousin", {}, 0.7768, 0.7768)

    def test_cousin_depth3(self):
        self.check("Cousin", {"max_depth": 3}, 0.6703, 0.6399)  # positive_paths is better

    def test_uncle_full(self):
        self.check("Uncle", {}, 0.9060, 0.9060)

    def test_uncle_depth3(self):
        self.check("Uncle", {"max_depth": 3}, 0.8927, 0.8794)  # positive_paths is better


# ----------------------------------------------------------------------------------------------------------------------
# 4. Carcinogenesis benchmark (slow)
# ----------------------------------------------------------------------------------------------------------------------
CARCINOGENESIS_KB = "KGs/Carcinogenesis/carcinogenesis.owl"
CARCINOGENESIS_LPS = "LPs/Carcinogenesis/lps.json"


@unittest.skipUnless(os.environ.get("TDL_SLOW_TESTS") and os.path.exists(CARCINOGENESIS_KB)
                     and os.path.exists(CARCINOGENESIS_LPS),
                     "slow (~10 min): set TDL_SLOW_TESTS=1 and download Carcinogenesis")
class TestCarcinogenesis(unittest.TestCase):
    """5-fold CV on NOTKNOWN (55 positives, 46 negatives).

    ===========  ==============  ================  =============  =====================
    tree         positive_paths  predicted_leaves  factored_tree  size factored/positive
    ===========  ==============  ================  =============  =====================
    full tree        0.620           0.620             0.620          ~0.36 (2.7x shorter)
    max_depth=3      0.705           0.592             0.592          ~0.48
    ===========  ==============  ================  =============  =====================

    On this problem the leaf-based modes are worse with a shallow tree: majority-vote leaves lose recall.
    """

    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase(path=CARCINOGENESIS_KB)
        lp = json.load(open(CARCINOGENESIS_LPS))["problems"]["NOTKNOWN"]
        cls.pos, cls.neg = lp["positive_examples"], lp["negative_examples"]

    def test_full_tree(self):
        out = cross_validate(self.kb, self.pos, self.neg)
        report("Carcinogenesis full tree", out)
        for m in MODES:
            self.assertAlmostEqual(out[m][0], 0.620, places=3)
        self.assertLess(out[FACTORED_TREE][1], 0.5 * out[POSITIVE_PATHS][1])  # 560 vs 1632 characters

    def test_depth3(self):
        out = cross_validate(self.kb, self.pos, self.neg, {"max_depth": 3})
        report("Carcinogenesis max_depth=3", out)
        self.assertAlmostEqual(out[POSITIVE_PATHS][0], 0.705, places=3)
        self.assertAlmostEqual(out[PREDICTED_LEAVES][0], 0.592, places=3)
        self.assertAlmostEqual(out[FACTORED_TREE][0], 0.592, places=3)


if __name__ == "__main__":
    unittest.main()
