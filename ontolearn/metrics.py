from typing import Final, Tuple

from .abstracts import AbstractScorer
from .learning_problem import EncodedPosNegLPStandard


class Recall(AbstractScorer):
    __slots__ = ()

    name: Final = 'Recall'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        try:
            recall = tp / (tp + fn)
            return True, round(recall, 5)
        except ZeroDivisionError:
            return False, 0


class Precision(AbstractScorer):
    __slots__ = ()

    name: Final = 'Precision'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        try:
            precision = tp / (tp + fp)
            return True, round(precision, 5)
        except ZeroDivisionError:
            return False, 0


class F1(AbstractScorer):
    __slots__ = ()

    name: Final = 'F1'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            return False, 0

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            return False, 0

        if precision == 0 or recall == 0:
            return False, 0

        f_1 = 2 * ((precision * recall) / (precision + recall))
        return True, round(f_1, 5)


class Accuracy(AbstractScorer):
    """
    Accuracy is          acc = (tp + tn) / (tp + tn + fp+ fn). However,
    Concept learning papers (e.g. Learning OWL Class expression) appear to invernt their own accuracy metrics.

    In OCEL =>    Accuracy of a concept = 1 - ( \\|E^+ \\ R(C)\\|+ \\|E^- AND R(C)\\|) / \\|E\\|)


    In CELOE  =>    Accuracy of a concept C = 1 - ( \\|R(A) \\ R(C)\\| + \\|R(C) \\ R(A)\\|)/n



    1) R(.) is the retrieval function, A is the class to describe and C in CELOE.

    2) E^+ and E^- are the positive and negative examples probided. E = E^+ OR E^- .
    """
    __slots__ = ()

    name: Final = 'Accuracy'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        acc = (tp + tn) / (tp + tn + fp + fn)
        # acc = 1 - ((fp + fn) / len(self.pos) + len(self.neg)) # from Learning OWL Class Expressions.

        return True, round(acc, 5)


class WeightedAccuracy(AbstractScorer):
    __slots__ = ()

    name: Final = 'WeightedAccuracy'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        ap = tp + fn
        an = fp + tn

        wacc = ((tp/ap) + (tn/an)) / ((tp/ap) + (tn/an) + (fp/an) + (fn/ap))

        return True, round(wacc, 5)
