"""Heuristic functions."""

from typing import Final

import numpy as np

from .abstracts import AbstractHeuristic, AbstractOEHeuristicNode, EncodedLearningProblem
from .learning_problem import EncodedPosNegUndLP, EncodedPosNegLPStandard
from .metrics import Accuracy
from .search import LBLNode, RL_State


class CELOEHeuristic(AbstractHeuristic[AbstractOEHeuristicNode]):
    """Heuristic like the CELOE Heuristic in DL-Learner."""
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
        """Create a new CELOE Heuristic.

        Args:
            gainBonusFactor: Factor that weighs the increase in quality compared to the parent node.
            startNodeBonus: Special value added to the root node.
            nodeRefinementPenalty: Value that is subtracted from the heuristic for each refinement attempt of this node.
            expansionPenaltyFactor: Value that is subtracted from the heuristic for each horizontal expansion of this
                node.
        """
        self.gainBonusFactor = gainBonusFactor
        self.startNodeBonus = startNodeBonus
        self.nodeRefinementPenalty = nodeRefinementPenalty
        self.expansionPenaltyFactor = expansionPenaltyFactor

    def apply(self, node: AbstractOEHeuristicNode, instances, learning_problem: EncodedLearningProblem):
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


class DLFOILHeuristic(AbstractHeuristic):
    """DLFOIL Heuristic."""
    __slots__ = ()

    name: Final = 'custom_dl_foil'

    def __init__(self):
        # @todo Needs to be tested.
        ...

    def apply(self, node, instances, learning_problem: EncodedPosNegUndLP):

        instances = node.concept.instances
        if len(instances) == 0:
            node.heuristic = 0
            return False

        p_1 = len(learning_problem.kb_pos.intersection(instances))  # number of positive examples covered by the concept
        n_1 = len(learning_problem.kb_neg.intersection(instances))  # number of negative examples covered by the concept
        u_1 = len(learning_problem.kb_unlabelled.intersection(instances))
        term1 = np.log(p_1 / (p_1 + n_1 + u_1))

        if node.parent_node:
            parent_inst = node.parent_node.individuals
            p_0 = len(
                learning_problem.kb_pos.intersection(parent_inst))  # number of positive examples covered by the concept
            n_0 = len(
                learning_problem.kb_neg.intersection(parent_inst))  # number of negative examples covered by the concept
            u_0 = len(learning_problem.kb_unlabelled.intersection(parent_inst))
            term2 = np.log(p_0 / (p_0 + n_0 + u_0))
        else:
            term2 = 0

        gain = round(p_1 * (term1 - term2), 5)
        node.heuristic = gain


class OCELHeuristic(AbstractHeuristic):
    """OCEL Heuristic."""
    __slots__ = 'accuracy_method', 'gainBonusFactor', 'expansionPenaltyFactor'

    name: Final = 'OCEL_Heuristic'

    def __init__(self, *, gainBonusFactor: float = 0.5,
                 expansionPenaltyFactor: float = 0.02):
        super().__init__()
        self.accuracy_method = Accuracy()

        self.gainBonusFactor = gainBonusFactor  # called alpha in the paper and gainBonusFactor in the original code
        self.expansionPenaltyFactor = expansionPenaltyFactor  # called beta in the paper

    def apply(self, node: LBLNode, instances, learning_problem: EncodedPosNegLPStandard):
        assert isinstance(node, LBLNode), "OCEL Heuristic requires instances information of a node"

        heuristic_val = 0
        accuracy_gain = 0
        _, accuracy = self.accuracy_method.score_elp(node.individuals, learning_problem)

        if node.parent_node is not None:
            _, parent_accuracy = self.accuracy_method.score_elp(node.parent_node.individuals, learning_problem)
            accuracy_gain = accuracy - parent_accuracy

        heuristic_val += accuracy + self.gainBonusFactor * accuracy_gain - node.h_exp * self.expansionPenaltyFactor
        node.heuristic = round(heuristic_val, 5)


class CeloeBasedReward:
    """Reward function for DRILL."""
    def __init__(self, reward_of_goal=5.0, beta=.04, alpha=.5):
        self.name = 'DRILL_Reward'
        self.lp = None
        self.reward_of_goal = reward_of_goal
        self.beta = beta
        self.alpha = alpha

    @property
    def learning_problem(self):
        return self.lp

    @learning_problem.setter
    def learning_problem(self, x):
        assert isinstance(x, EncodedLearningProblem)
        self.lp = x

    def apply(self, rl_state: RL_State, next_rl_state: RL_State):
        assert next_rl_state.quality is not None
        assert rl_state.quality is not None
        reward = next_rl_state.quality
        if next_rl_state.quality == 1.0:
            reward = self.reward_of_goal
        else:
            # Reward => being better than parent.
            reward += (next_rl_state.quality - rl_state.quality) * self.alpha
        # Regret => Length penalization.
        reward -= next_rl_state.length * self.beta
        return max(reward, 0)
