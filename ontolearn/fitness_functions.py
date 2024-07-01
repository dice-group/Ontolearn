# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

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
