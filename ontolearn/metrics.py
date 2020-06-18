from .abstracts import AbstractScorer, AbstractHeuristic
from .search import Node
from typing import Set


class F1(AbstractScorer):
    def __init__(self, pos=None, neg=None):
        super().__init__(pos, neg)
        self.name = 'F1'
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
        self.applied += 1

        instances = node.concept.instances
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


class PredictiveAccuracy(AbstractScorer):
    """
    Accuracy is          acc = (tp + tn) / (tp + tn + fp+ fn). However,
    Concept learning papers (e.g. Learning OWL Class expression) appear to invernt their own accuracy metrics.

    In OCEL =>    Accuracy of a concept = 1 - ( |E^+ \ R(C)|+ |E^- AND R(C)|) / |E|)


    In CELOE  =>    Accuracy of a concept C = 1 - ( |R(A) \ R(C)| + |R(C) \ R(A)|)/n



    1) R(.) is the retrieval function, A is the class to describe and C in CELOE.

    2) E^+ and E^- are the positive and negative examples probided. E = E^+ OR E^- .
    """

    def __init__(self, pos=None, neg=None):
        super().__init__(pos, neg)
        self.name = 'Accuracy'

    def apply(self, node: Node):
        assert isinstance(node, Node)
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.quality = 0
            return False

        tp = len(self.pos.intersection(instances))
        tn = len(self.neg.difference(instances))

        fp = len(self.neg.intersection(
            instances))  # FP corresponds to CN in Learning OWL Class Expressions OCEL paper, i.e., cn = |R(C) \AND E^-| covered negatives
        fn = len(self.pos.difference(
            instances))  # FN corresponds to UP in Learning OWL Class Expressions OCEL paper, i.e., up = |E^+ \ R(C)|
        # uncovered positives

        acc = (tp + tn) / (tp + tn + fp + fn)
        # acc = 1 - ((fp + fn) / len(self.pos) + len(self.neg)) # from Learning OWL Class Expressions.

        node.quality = round(acc, 5)


class CELOEHeuristic(AbstractHeuristic):
    def __init__(self,pos=None,neg=None,unlabelled=None):
        super().__init__(pos, neg,unlabelled)
        self.name = 'CELOE'

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


class DLFOILHeuristic(AbstractHeuristic):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'DL-FOIL'

    def apply(self, node):
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.heuristic = 0
            return False

        p_1 = len(self.pos.intersection(instances))  # number of positive examples covered by the concept
        n_1 = len(self.neg.intersection(instances))  # number of negative examples covered by the concept
        u_1 = len(self.unlabelled.intersection(instances))

        import numpy as np

        if node.parent_node:
            parent_inst = node.parent_node.concept.instances
            p_0 = len(self.pos.intersection(parent_inst))  # number of positive examples covered by the concept
            n_0 = len(self.neg.intersection(parent_inst))  # number of negative examples covered by the concept
            u_0 = len(self.unlabelled.intersection(parent_inst))
        else:
            p_0, n_0, u_0 = 0, 0, 0

        try:
            term1 = np.log((p_1) / (p_1 + n_1 + u_1))
        except:
            term1 = 0

        try:
            term2 = np.log((p_0) / (p_0 + n_0 + u_0))
        except:
            term2 = 0

        gain = p_1 * (term1 - term2)
        node.heuristic = round(gain, 5)


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
