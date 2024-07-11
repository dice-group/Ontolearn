# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""Quality metrics for concept learners."""
from typing import Final, Tuple

from .abstracts import AbstractScorer


class Recall(AbstractScorer):
    """Recall quality function.

    Attribute:
        name: name of the metric = 'Recall'.
    """
    __slots__ = ()

    name: Final = 'Recall'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        try:
            recall = tp / (tp + fn)
            return True, round(recall, 5)
        except ZeroDivisionError:
            return False, 0


class Precision(AbstractScorer):
    """Precision quality function.

    Attribute:
        name: name of the metric = 'Precision'.
    """
    __slots__ = ()

    name: Final = 'Precision'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        try:
            precision = tp / (tp + fp)
            return True, round(precision, 5)
        except ZeroDivisionError:
            return False, 0


class F1(AbstractScorer):
    """F1-score quality function.

    Attribute:
        name: name of the metric = 'F1'.
    """
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
    Accuracy quality function.
    Accuracy is acc = (tp + tn) / (tp + tn + fp+ fn).
    However, Concept learning papers (e.g. Learning OWL Class expression) appear to invent their own accuracy metrics.

    In OCEL =>    Accuracy of a concept = 1 - ( \\|E^+ \\ R(C)\\|+ \\|E^- AND R(C)\\|) / \\|E\\|).


    In CELOE  =>    Accuracy of a concept C = 1 - ( \\|R(A) \\ R(C)\\| + \\|R(C) \\ R(A)\\|)/n.



    1) R(.) is the retrieval function, A is the class to describe and C in CELOE.

    2) E^+ and E^- are the positive and negative examples probided. E = E^+ OR E^- .

    Attribute:
        name: name of the metric = 'Accuracy'.
    """
    __slots__ = ()

    name: Final = 'Accuracy'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        acc = (tp + tn) / (tp + tn + fp + fn)
        # acc = 1 - ((fp + fn) / len(self.pos) + len(self.neg)) # from Learning OWL Class Expressions.

        return True, round(acc, 5)


class WeightedAccuracy(AbstractScorer):
    """
    WeightedAccuracy quality function.

    Attribute:
        name: name of the metric = 'WeightedAccuracy'.
    """
    __slots__ = ()

    name: Final = 'WeightedAccuracy'

    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, float]:
        ap = tp + fn
        an = fp + tn

        wacc = ((tp/ap) + (tn/an)) / ((tp/ap) + (tn/an) + (fp/an) + (fn/ap))

        return True, round(wacc, 5)
