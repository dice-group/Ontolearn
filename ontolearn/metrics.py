from .abstracts import AbstractScorer
from .search import Node
from typing import Set


class F1(AbstractScorer):
    def __init__(self, pos=None, neg=None):
        super().__init__(pos, neg)
        self.applied = 0
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
        except ValueError:
            f_1 = 0

        return round(f_1, 5)

    def apply(self, node):
        """
        TODO: do it more intelligently so that we can use multiprocessing.
        @param node:
        @return:
        """
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False

        tp = len(self.pos.intersection(instances))
        tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(instances))
        fn = len(self.pos.difference(instances))

        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            node.quality = 0
            return False

        try:
            precision = tp / (tp + fp)
        except:
            node.quality = 0
            return False

        if precision == 0 or recall == 0:
            node.quality = 0
            return False

        f_1 = 2 * ((precision * recall) / (precision + recall))
        node.quality = round(f_1, 5)

        assert node.quality


class CELOEHeuristic:
    def __init__(self):
        self.gainBonusFactor = 0.3
        self.startNodeBonus = 0.1
        self.nodeRefinementPenalty = 0.001
        self.expansionPenaltyFactor = 0.1
        self.applied = 0

    def apply(self, node):
        self.applied += 1

        try:
            assert node.quality
        except AssertionError:
            print(node)
            print(node.parent_node)
            exit(1)
        heuristic_val = 0
        heuristic_val += node.quality

        assert id(heuristic_val) != node.quality

        if node.parent_node:
            heuristic_val += (node.parent_node.quality - node.quality) * self.gainBonusFactor
        else:
            heuristic_val += self.startNodeBonus

        # penalty for horizontal expansion
        heuristic_val -= node.h_exp * self.expansionPenaltyFactor
        # // penalty for having many child nodes (stuck prevention)
        heuristic_val -= node.refinement_count * self.nodeRefinementPenalty
        node.heuristic = round(heuristic_val, 5)


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
