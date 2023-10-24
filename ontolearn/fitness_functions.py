"""Fitness functions."""

from typing import Final
from ontolearn.abstracts import AbstractFitness
from ontolearn.ea_utils import Tree


class LinearPressureFitness(AbstractFitness):
    """Linear parametric parsimony pressure."""

    __slots__ = 'gain', 'penalty'

    name: Final = 'Linear_Pressure_Fitness'

    def __init__(self, gain: float = 2048.0, penalty: float = 1.0):
        self.gain = gain
        self.penalty = penalty

    def apply(self, individual: Tree):
        quality = individual.quality.values[0]
        fitness = self.gain*quality - self.penalty*len(individual)
        individual.fitness.values = (round(fitness, 5),)
