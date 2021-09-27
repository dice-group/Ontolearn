from typing import Final, Tuple

from .abstracts import AbstractScorer
from .learning_problem import EncodedPosNegLPStandard


class Recall(AbstractScorer):
    __slots__ = ()

    name: Final = 'Recall'

    def score(self, instances, learning_problem: EncodedPosNegLPStandard):
        if len(instances) == 0:
            return False, 0
        tp = len(learning_problem.kb_pos.intersection(instances))
        fn = len(learning_problem.kb_pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            return True, round(recall, 5)
        except ZeroDivisionError:
            return False, 0


class Precision(AbstractScorer):
    __slots__ = ()

    name: Final = 'Precision'

    def score(self, instances, learning_problem: EncodedPosNegLPStandard):
        if len(instances) == 0:
            return False, 0
        tp = len(learning_problem.kb_pos.intersection(instances))
        fp = len(learning_problem.kb_neg.intersection(instances))
        try:
            precision = tp / (tp + fp)
            return True, round(precision, 5)
        except ZeroDivisionError:
            return False, 0


class F1(AbstractScorer):
    __slots__ = ()

    name: Final = 'F1'

    def score(self, instances, learning_problem: EncodedPosNegLPStandard):
        if len(instances) == 0:
            return False, 0
        tp = len(learning_problem.kb_pos.intersection(instances))
        # tn = len(learning_problem.kb_neg.difference(instances))

        fp = len(learning_problem.kb_neg.intersection(instances))
        fn = len(learning_problem.kb_pos.difference(instances))

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

    def score(self, instances, learning_problem: EncodedPosNegLPStandard) -> Tuple[bool, float]:
        if len(instances) == 0:
            return False, 0

        tp = len(learning_problem.kb_pos.intersection(instances))
        tn = len(learning_problem.kb_neg.difference(instances))

        # FP corresponds to CN in Learning OWL Class Expressions OCEL paper, i.e., cn = |R(C) \AND
        # E^-| covered negatives
        fp = len(learning_problem.kb_neg.intersection(instances))
        # FN corresponds to UP in Learning OWL Class Expressions OCEL paper, i.e., up = |E^+ \ R(C)|
        fn = len(learning_problem.kb_pos.difference(instances))
        # uncovered positives

        acc = (tp + tn) / (tp + tn + fp + fn)
        # acc = 1 - ((fp + fn) / len(self.pos) + len(self.neg)) # from Learning OWL Class Expressions.

        return True, round(acc, 5)


class WeightedAccuracy(AbstractScorer):
    __slots__ = ()

    name: Final = 'WeightedAccuracy'

    def score(self, instances, learning_problem: EncodedPosNegLPStandard):
        if len(instances) == 0:
            return False, 0

        ap = len(learning_problem.kb_pos)
        an = len(learning_problem.kb_neg)

        tp = len(learning_problem.kb_pos.intersection(instances))
        tn = len(learning_problem.kb_neg.difference(instances))

        # FP corresponds to CN in Learning OWL Class Expressions OCEL paper, i.e., cn = |R(C) \AND
        # E^-| covered negatives
        fp = len(learning_problem.kb_neg.intersection(instances))
        # FN corresponds to UP in Learning OWL Class Expressions OCEL paper, i.e., up = |E^+ \ R(C)|
        fn = len(learning_problem.kb_pos.difference(instances))
        # uncovered positives

        wacc = ((tp/ap) + (tn/an)) / ((tp/ap) + (tn/an) + (fp/an) + (fn/ap))
        # acc = 1 - ((fp + fn) / len(self.pos) + len(self.neg)) # from Learning OWL Class Expressions.

        return True, round(wacc, 5)
