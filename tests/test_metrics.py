import unittest

from ontolearn.metrics import Recall, Precision, F1, Accuracy, WeightedAccuracy


class MetricsTest(unittest.TestCase):

    def test_recall(self):

        score = Recall().score2(10, 10, 6, 2)
        self.assertEqual(score, (True, 0.5))

        score = Recall().score2(0, 0, 6, 2)
        self.assertEqual(score, (False, 0))

    def test_precision(self):

        score = Precision().score2(10, 10, 6, 2)
        self.assertEqual(score, (True, 0.625))

        score = Precision().score2(0, 10, 0, 2)
        self.assertEqual(score, (False, 0))

    def test_f1(self):

        score = F1().score2(10, 10, 6, 2)
        self.assertEqual(score, (True, 0.55556))

        score = F1().score2(0, 0, 0, 2)
        self.assertEqual(score, (False, 0))

    def test_accuracy(self):

        score = Accuracy().score2(10, 10, 6, 2)
        self.assertEqual(score, (True, 0.42857))

    def test_wighted_accuracy(self):

        score = WeightedAccuracy().score2(10, 10, 6, 2)
        self.assertEqual(score, (True, 0.375))

