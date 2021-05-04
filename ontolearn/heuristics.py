from typing import Final

import numpy as np

from .abstracts import AbstractScorer, AbstractHeuristic, AbstractOEHeuristicNode, AbstractLearningProblem
from .metrics import Accuracy
from .search import LBLNode, OENode


class CELOEHeuristic(AbstractHeuristic[AbstractOEHeuristicNode]):
    """Heuristic like the CELOE Heuristic in DL-Learner"""
    __slots__ = 'gainBonusFactor', 'startNodeBonus', 'nodeRefinementPenalty', 'expansionPenaltyFactor'

    name: Final = 'CELOE_Heuristic'

    gainBonusFactor: Final[float]
    startNodeBonus: Final[float]
    nodeRefinementPenalty: Final[float]
    expansionPenaltyFactor: Final[float]

    def __init__(self, *,
                 gainBonusFactor: float = 0.3,
                 startNodeBonus: float = 0.1,
                 nodeRefinementPenalty: float = 0.001,
                 expansionPenaltyFactor: float = 0.1):
        """Create a new CELOE Heuristic

        Args:
            gainBonusFactor: factor that weighs the increase in quality compared to the parent node
            startNodeBonus: special value added to the root node
            nodeRefinementPenalty: value that is substracted from the heuristic for each refinement attempt of this node
            expansionPenaltyFactor: value that is substracted from the heuristic for each horizontal expansion of this
                node
        """
        super().__init__()
        self.gainBonusFactor = gainBonusFactor
        self.startNodeBonus = startNodeBonus
        self.nodeRefinementPenalty = nodeRefinementPenalty
        self.expansionPenaltyFactor = expansionPenaltyFactor

    def apply(self, node: AbstractOEHeuristicNode, instances=None):
        self.applied += 1

        heuristic_val = 0
        heuristic_val += node.quality

        if node.is_root:
            heuristic_val += self.startNodeBonus
        else:
            heuristic_val += (node.quality - node.parent_node.quality) * self.gainBonusFactor

        # penalty for horizontal expansion
        heuristic_val -= (node.h_exp - 1) * self.expansionPenaltyFactor
        # // penalty for having many child nodes (stuck prevention)
        heuristic_val -= node.refinement_count * self.nodeRefinementPenalty
        node.heuristic = round(heuristic_val, 5)

    def clean(self):
        super().clean()


class DLFOILHeuristic(AbstractHeuristic):
    def __init__(self, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)
        self.name = 'custom_dl_foil'
        # @todo Needs to be tested.

    def apply(self, node, instances=None):
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


class OCELHeuristic(AbstractHeuristic):
    __slots__ = 'lp', 'accuracy', 'gainBonusFactor', 'expansionPenaltyFactor'

    name: Final = 'OCEL_Heuristic'

    def __init__(self, *, learning_problem: AbstractLearningProblem, gainBonusFactor: float = 0.5,
                 expansionPenaltyFactor: float = 0.02):
        super().__init__()
        self.lp = learning_problem
        self.accuracy_method = Accuracy(learning_problem=learning_problem)

        self.gainBonusFactor = gainBonusFactor   # called alpha in the paper and gainBonusFactor in the original code
        self.expansionPenaltyFactor = expansionPenaltyFactor  # called beta in the paper

    def apply(self, node, instances=None):
        assert isinstance(node, LBLNode), "OCEL Heuristic requires instances information of a node"

        self.applied += 1

        heuristic_val = 0
        accuracy_gain = 0
        _, accuracy = self.accuracy_method.score(instances)

        if node.parent_node is not None:
            _, parent_accuracy = self.accuracy_method.score(node.parent_node.individuals)
            accuracy_gain = accuracy - parent_accuracy

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
