from .abstracts import AbstractScorer
from .search import Node
from typing import Set, ClassVar, Final


class Recall(AbstractScorer):
    name: Final = 'Recall'

    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)

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
        tp = len(self.pos.intersection(instances))
        fn = len(self.pos.difference(instances))
        try:
            recall = tp / (tp + fn)
            node.quality = round(recall, 5)
        except ZeroDivisionError:
            node.quality = 0
            return False


class Precision(AbstractScorer):
    name: Final = 'Precision'

    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)

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
        tp = len(self.pos.intersection(instances))
        fp = len(self.neg.intersection(instances))
        try:
            precision = tp / (tp + fp)
            node.quality = round(precision, 5)
        except ZeroDivisionError:
            node.quality = 0
            return False


class F1(AbstractScorer):
    name: Final = 'F1'

    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.beta = 0
        self.noise = 0

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

        tp = len(self.pos.intersection(instances))
        # tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))

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
    name: Final = 'Accuracy'

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

    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)

    def apply(self, node: Node, instances):
        assert isinstance(node, Node)
        self.applied += 1

        if len(instances) == 0:
            node.quality = 0
            return False

        tp = len(self.pos.intersection(instances))
        tn = len(self.neg.difference(instances))

        # FP corresponds to CN in Learning OWL Class Expressions OCEL paper, i.e., cn = |R(C) \AND
        # E^-| covered negatives
        fp = len(self.neg.intersection(instances))
        # FN corresponds to UP in Learning OWL Class Expressions OCEL paper, i.e., up = |E^+ \ R(C)|
        fn = len(self.pos.difference(instances))
        # uncovered positives

        acc = (tp + tn) / (tp + tn + fp + fn)
        # acc = 1 - ((fp + fn) / len(self.pos) + len(self.neg)) # from Learning OWL Class Expressions.

        node.quality = round(acc, 5)
