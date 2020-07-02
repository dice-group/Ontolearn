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
            heuristic_val += (parent_node.quality - node.quality) * self.gainBonusFactor
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

    def score(self):
        pass

    def apply(self, node):
        self.applied += 1

        instances = node.concept.instances
        if len(instances) == 0:
            node.heuristic = 0
            return False

        p_1 = len(self.pos.intersection(instances))  # number of positive examples covered by the concept
        n_1 = len(self.neg.intersection(instances))  # number of negative examples covered by the concept
        u_1 = len(self.unlabelled.intersection(instances))

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

        accuracy += 1 - (uncovered_positives + covered_negatives) / (len(self.pos)+len(self.neg))  # ACCURACY of Concept

        accuracy_gain += accuracy
        if node.parent_node is not None:
            uncovered_positives_parent = len(self.pos.difference(node.parent_node.concept.instances))
            covered_negatives_parent = len(self.neg.intersection(node.parent_node.concept.instances))

            parent_accuracy = 1 - (
                    uncovered_positives_parent + covered_negatives_parent) / (len(self.pos)+len(self.neg))  # ACCURACY of Concept
            accuracy_gain -= parent_accuracy

        heuristic_val += accuracy + self.gainBonusFactor * accuracy_gain - node.h_exp * self.expansionPenaltyFactor
        node.heuristic = round(heuristic_val, 5)
