from .abstracts import AbstractScorer
from .search import Node
from typing import Set


class F1(AbstractScorer):
    def __init__(self, pos, neg, N=None):
        super().__init__(pos, neg)
        self.num_concept_tested=0

    def get_quality(self, n: Node, beta=0, noise=0):
        assert isinstance(n, Node)

        individuals = n.ce.indvs

        if len(individuals) == 0:
            return 0
        tp = len(self.pos.intersection(individuals))
        tn = len(self.neg.difference(individuals))

        fp = len(self.neg.intersection(individuals))
        fn = len(self.pos.difference(individuals))

        try:
            recall = tp / (tp + fn)
        except:
            recall = 0

        if beta == 0:
            if (recall == 0) or (recall < 1 - noise):
                return 0
        else:
            raise NotImplemented

        try:
            precision = tp / (tp + fp)
        except:
            precision = 0

        if beta == 0:
            f_1 = 2 * ((precision * recall) / (precision + recall))
        else:
            raise NotImplemented

        return round(f_1, 5)

    def apply(self, n):
        self.num_concept_tested+=1
        if isinstance(n, Set):
            for i in n:
                i.score = round(self.get_quality(i), 5)
        elif isinstance(n, Node):
            n.score = round(self.get_quality(n), 5)
        else:
            raise ValueError

        """
        individuals = n.ce.indvs

        if len(individuals) == 0:
            return 0

        tp = len(self.pos.intersection(individuals))
        tn = len(self.neg.difference(individuals))

        fp = len(self.neg.intersection(individuals))
        fn = len(self.pos.difference(individuals))

        try:
            recall = tp / (tp + fn)
        except:
            recall = 0

        try:
            precision = tp / (tp + fp)
        except:
            precision = 0

        try:
            f_1 = 2 * ((precision * recall) / (precision + recall))

        except:
            f_1 = 0

        n.score = round(f_1, 5)
        """


class PredictiveAccuracy(AbstractScorer):
    def __init__(self, pos, neg, N):
        super().__init__(pos, neg)
        self.total_num_of_instances = N

    def apply(self, n: Node):
        assert isinstance(n, Node)

        individuals = n.ce.indvs

        positives = len(self.pos)
        negatives = len(self.neg)

        tp = len(self.pos.intersection(individuals))
        tn = len(self.neg.difference(individuals))

        fp = len(self.neg.intersection(individuals))
        fn = len(self.neg.difference(individuals))

        acc = (tp + tn) / (positives + negatives)

        n.score = round((2 * tp) / (2 * tp + fp + fn), 4)


class NoQuality(AbstractScorer):
    def __init__(self):
        super().__init__({}, {})

    def apply(self, n):
        if isinstance(n, Set):
            for i in n:
                i.score = 0.0
        elif isinstance(n, Node):
            n.score = 0.0
        else:
            raise ValueError



class CeloeMetric:
    def __init__(self):
        self.classLength = 1
        self.objectComplementLength = 1
        self.objectSomeValuesLength = 1
        self.objectPropertyLength = 1

        self.objectCardinalityLength = 2
        self.objectAllValuesLength = 1
        self.objectInverseLength = 2
