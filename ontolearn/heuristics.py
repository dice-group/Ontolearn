from .abstracts import AbstractScorer
import numpy as np


class CELOEHeuristic(AbstractScorer):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'CELOE_Heuristic'

        self.gainBonusFactor = 0.3
        self.startNodeBonus = 0.1
        self.nodeRefinementPenalty = 0.001
        self.expansionPenaltyFactor = 0.1
        self.applied = 0

    def score(self):
        pass

    def apply(self, node, parent_node=None):
        self.applied += 1

        heuristic_val = 0
        heuristic_val += node.quality

        assert id(heuristic_val) != node.quality

        if node.parent_node is not None:
            # heuristic_val += (parent_node.quality - node.quality) * self.gainBonusFactor
            heuristic_val += (node.quality - parent_node.quality) * self.gainBonusFactor
        else:
            heuristic_val += self.startNodeBonus

        # penalty for horizontal expansion
        heuristic_val -= node.h_exp * self.expansionPenaltyFactor
        # // penalty for having many child nodes (stuck prevention)
        heuristic_val -= node.refinement_count * self.nodeRefinementPenalty
        node.heuristic = round(heuristic_val, 5)


class DLFOILHeuristic(AbstractScorer):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'DL-FOIL_Heuristic'
        # @todo Needs to be tested.

    def score(self):
        pass

    def apply(self, node, parent_node=None):
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.heuristic = 0
            return False

        p_1 = len(self.pos.intersection(instances))  # number of positive examples covered by the concept
        n_1 = len(self.neg.intersection(instances))  # number of negative examples covered by the concept
        u_1 = len(self.unlabelled.intersection(instances))
        term1 = np.log(p_1 / (p_1 + n_1 + u_1))

        if parent_node:
            parent_inst = parent_node.concept.instances
            p_0 = len(self.pos.intersection(parent_inst))  # number of positive examples covered by the concept
            n_0 = len(self.neg.intersection(parent_inst))  # number of negative examples covered by the concept
            u_0 = len(self.unlabelled.intersection(parent_inst))
            term2 = np.log(p_0 / (p_0 + n_0 + u_0))
        else:
            term2 = 0

        gain = round(p_1 * (term1 - term2), 5)
        node.heuristic = gain


class OCELHeuristic(AbstractScorer):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'OCEL_Heuristic'
        self.applied = 0

        self.gainBonusFactor = 0.5  # called alpha in the paper and gainBonusFactor in the original code
        self.expansionPenaltyFactor = 0.02  # called beta in the paper
        self.applied = 0

    def score(self):
        pass

    def apply(self, node, parent_node=None):
        self.applied += 1

        heuristic_val = 0
        accuracy = 0
        accuracy_gain = 0

        uncovered_positives = len(self.pos.difference(node.concept.instances))
        covered_negatives = len(self.neg.intersection(node.concept.instances))

        accuracy += 1 - (uncovered_positives + covered_negatives) / (
                len(self.pos) + len(self.neg))  # ACCURACY of Concept

        accuracy_gain += accuracy
        if node.parent_node is not None:
            uncovered_positives_parent = len(self.pos.difference(node.parent_node.concept.instances))
            covered_negatives_parent = len(self.neg.intersection(node.parent_node.concept.instances))

            parent_accuracy = 1 - (
                    uncovered_positives_parent + covered_negatives_parent) / (
                                      len(self.pos) + len(self.neg))  # ACCURACY of Concept
            accuracy_gain -= parent_accuracy

        heuristic_val += accuracy + self.gainBonusFactor * accuracy_gain - node.h_exp * self.expansionPenaltyFactor
        node.heuristic = round(heuristic_val, 5)


class Reward(AbstractScorer):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'F1'
        self.beta = 0
        self.noise = 0

        self.reward_of_goal = 100.0
        self.gainBonusFactor = self.reward_of_goal * .1

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

    def apply(self, node):
        """
        Calculate F1-score and assigns it into quality variable of node.
        """
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

    def calculate(self, current_state, next_state=None) -> float:
        self.apply(current_state)
        self.apply(next_state)
        if next_state.quality == 1.0:
            return self.reward_of_goal
        reward = 0
        reward += next_state.quality
        if next_state.quality > current_state.quality:
            reward += (next_state.quality - current_state.quality) * self.gainBonusFactor
        return reward
