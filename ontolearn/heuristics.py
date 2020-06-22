from .abstracts import AbstractHeuristic
class CELOEHeuristic(AbstractHeuristic):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'CELOE'

        self.gainBonusFactor = 0.3
        self.startNodeBonus = 0.1
        self.nodeRefinementPenalty = 0.001
        self.expansionPenaltyFactor = 0.1
        self.applied = 0

    def apply(self, node):
        self.applied += 1

        heuristic_val = 0
        heuristic_val += node.quality

        assert id(heuristic_val) != node.quality

        if node.parent_node is not None:
            try:
                heuristic_val += (node.parent_node.quality - node.quality) * self.gainBonusFactor
            except TypeError as ty:
                print(node)
                print(node.parent_node)

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
