from .abstracts import AbstractScorer, AbstractLearningProblem, AbstractNode
from .learning_problem import PosNegLPStandard
from .search import Node
from typing import Set, ClassVar, Final


class Recall(AbstractScorer):
    __slots__ = ()

    name: Final = 'Recall'

    lp: PosNegLPStandard

    def __init__(self, learning_problem: PosNegLPStandard):
        super().__init__(lp)

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg
        if len(instances) == 0:
            return 0
        tp = len(self.pos.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            return round(recall, 5)
        except ValueError:
            return 0

    def apply(self, node, instances):
        self.applied += 1

        if len(instances) == 0:
            node.quality = 0
            return False
        tp = len(self.lp.kb_pos.intersection(instances))
        fn = len(self.lp.kb_pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            node.quality = round(recall, 5)
        except ZeroDivisionError:
            node.quality = 0
            return False


class Precision(AbstractScorer):
    __slots__ = ()

    name: Final = 'Precision'

    lp: PosNegLPStandard

    def __init__(self, learning_problem: PosNegLPStandard):
        super().__init__(lp)

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg
        if len(instances) == 0:
            return 0
        tp = len(self.pos.intersection(instances))
        fp = len(self.neg.intersection(instances))
        try:
            precision = tp / (tp + fp)
            return round(precision, 5)
        except ValueError:
            return 0

    def apply(self, node, instances):
        self.applied += 1

        if len(instances) == 0:
            node.quality = 0
            return False
        tp = len(self.lp.kb_pos.intersection(instances))
        fp = len(self.lp.kb_neg.intersection(instances))
        try:
            precision = tp / (tp + fp)
            node.quality = round(precision, 5)
        except ZeroDivisionError:
            node.quality = 0
            return False


class F1(AbstractScorer):
    __slots__ = ()

    name: Final = 'F1'

    lp: PosNegLPStandard

    def __init__(self, learning_problem: PosNegLPStandard):
        super().__init__(learning_problem=learning_problem)

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg

        tp = len(self.pos.intersection(instances))
        tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f_1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f_1 = 0

        return round(f_1, 5)

    def apply(self, node, instances):
        self.applied += 1

        if len(instances) == 0:
            node.quality = 0
            return False

        tp = len(self.lp.kb_pos.intersection(instances))
        # tn = len(self.lp.kb_neg.difference(instances))

        fp = len(self.lp.kb_neg.intersection(instances))
        fn = len(self.lp.kb_pos.difference(instances))

        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            node.quality = 0
            return False

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            node.quality = 0
            return False

        if precision == 0 or recall == 0:
            node.quality = 0
            return False

        f_1 = 2 * ((precision * recall) / (precision + recall))
        node.quality = round(f_1, 5)

        assert node.quality


class Accuracy(AbstractScorer):
    """
    Accuracy is          acc = (tp + tn) / (tp + tn + fp+ fn). However,
    Concept learning papers (e.g. Learning OWL Class expression) appear to invernt their own accuracy metrics.

    In OCEL =>    Accuracy of a concept = 1 - ( |E^+ \\ R(C)|+ |E^- AND R(C)|) / |E|)


    In CELOE  =>    Accuracy of a concept C = 1 - ( |R(A) \\ R(C)| + |R(C) \\ R(A)|)/n



    1) R(.) is the retrieval function, A is the class to describe and C in CELOE.

    2) E^+ and E^- are the positive and negative examples probided. E = E^+ OR E^- .
    """
    __slots__ = ()

    name: Final = 'Accuracy'

    lp: PosNegLPStandard

    def score(self, pos, neg, instances):
        self.pos = pos
        self.neg = neg

        tp = len(self.pos.intersection(instances))
        tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            acc = (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError as e:
            print(e)
            print(tp)
            print(tn)
            print(fp)
            print(fn)
            acc = 0
        return acc

    def __init__(self, learning_problem: PosNegLPStandard):
        super().__init__(learning_problem)

    def apply(self, node: AbstractNode, instances):
        assert isinstance(node, AbstractNode)
        self.applied += 1

        if len(instances) == 0:
            node.quality = 0
            return False

        tp = len(self.lp.kb_pos.intersection(instances))
        tn = len(self.lp.kb_neg.difference(instances))

        # FP corresponds to CN in Learning OWL Class Expressions OCEL paper, i.e., cn = |R(C) \AND
        # E^-| covered negatives
        fp = len(self.lp.kb_neg.intersection(instances))
        # FN corresponds to UP in Learning OWL Class Expressions OCEL paper, i.e., up = |E^+ \ R(C)|
        fn = len(self.lp.kb_pos.difference(instances))
        # uncovered positives

        acc = (tp + tn) / (tp + tn + fp + fn)
        # acc = 1 - ((fp + fn) / len(self.pos) + len(self.neg)) # from Learning OWL Class Expressions.

        node.quality = round(acc, 5)
