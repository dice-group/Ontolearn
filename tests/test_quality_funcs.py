import unittest

from owlapy.class_expression import OWLClass
from owlapy.iri import IRI

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from ontolearn.quality_funcs import _tp_tn_fp_fn, acc, evaluate_concept, f1

PATH_FAMILY = "KGs/Family/family-benchmark_rich_background.owl"
NS = "http://www.benchmark.org/family#"


class TestQualityFuncs(unittest.TestCase):

    def test_confusion_counts(self):
        tp, tn, fp, fn = _tp_tn_fp_fn({1, 2, 5}, pos={1, 2, 3}, neg={4, 5})
        self.assertEqual((tp, tn, fp, fn), (2, 1, 1, 1))

    def test_confusion_counts_require_sets(self):
        with self.assertRaises(AssertionError):
            _tp_tn_fp_fn([1], {1}, {2})
        with self.assertRaises(AssertionError):
            _tp_tn_fp_fn({1}, [1], {2})
        with self.assertRaises(AssertionError):
            _tp_tn_fp_fn({1}, {1}, [2])

    def test_f1_perfect_and_partial(self):
        self.assertEqual(f1(individuals={1, 2}, pos={1, 2}, neg={3}), 1.0)
        # tp=1, fp=1, fn=1 -> precision = recall = 0.5
        self.assertAlmostEqual(f1(individuals={1, 3}, pos={1, 2}, neg={3}), 0.5, places=4)

    def test_f1_not_applicable_returns_zero(self):
        # no true positives -> F1 not applicable
        self.assertEqual(f1(individuals={3}, pos={1, 2}, neg={3}), 0)
        # empty prediction
        self.assertEqual(f1(individuals=set(), pos={1, 2}, neg={3}), 0)

    def test_accuracy(self):
        self.assertEqual(acc(individuals={1, 2}, pos={1, 2}, neg={3, 4}), 1.0)
        # tp=1, tn=1, fp=1, fn=1
        self.assertAlmostEqual(acc(individuals={1, 3}, pos={1, 2}, neg={3, 4}), 0.5, places=4)

    def test_evaluate_concept(self):
        kb = KnowledgeBase(path=PATH_FAMILY)
        female = OWLClass(IRI.create(NS + "Female"))
        male = OWLClass(IRI.create(NS + "Male"))
        females = set(kb.individuals(female))
        males = set(kb.individuals(male))
        self.assertTrue(females and males)

        lp = PosNegLPStandard(pos=set(list(females)[:5]), neg=set(list(males)[:5])).encode_kb(kb)
        for scorer in (F1(), Accuracy()):
            e = evaluate_concept(kb, female, scorer, lp)
            self.assertEqual(e.inds, kb.individuals_set(female))
            self.assertEqual(e.ic, len(e.inds))
            self.assertIsNotNone(e.q)
            self.assertTrue(0.0 <= e.q <= 1.0)
        # Female covers all 5 positives and none of the negatives -> perfect F1.
        self.assertEqual(evaluate_concept(kb, female, F1(), lp).q, 1.0)


if __name__ == "__main__":
    unittest.main()
