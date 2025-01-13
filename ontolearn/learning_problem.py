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

"""Learning problem in Ontolearn."""
import logging
import random
from typing import Set, Optional
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.abstracts import AbstractLearningProblem, EncodedLearningProblem, EncodedPosNegLPStandardKind, \
    AbstractKnowledgeBase
from owlapy.owl_individual import OWLNamedIndividual

logger = logging.getLogger(__name__)


class EncodedPosNegLPStandard(EncodedPosNegLPStandardKind):
    """Encoded learning problem standard.

    Attributes:
        kb_pos (set): Positive examples.
        kb_neg (set): Negative examples.
        kb_diff (set): kb_all - (kb_pos + kb_neg).
        kb_all (set): All examples/ all individuals set.
    """
    __slots__ = 'kb_pos', 'kb_neg', 'kb_diff', 'kb_all'

    kb_pos: set
    kb_neg: set
    kb_diff: set
    kb_all: set

    def __init__(self, kb_pos, kb_neg, kb_diff, kb_all):
        """Create a new instance of EncodedPosNegLPStandard.

        Args:
            kb_pos (set): Positive examples.
            kb_neg (set): Negative examples.
            kb_diff (set): kb_all - (kb_pos + kb_neg).
            kb_all (set): All examples/ all individuals set.
        """
        self.kb_pos = kb_pos
        self.kb_neg = kb_neg
        self.kb_diff = kb_diff
        self.kb_all = kb_all


class PosNegLPStandard(AbstractLearningProblem):
    """Positive-Negative learning problem standard.
    Attributes:
        pos: Positive examples.
        neg: Negative examples.
        all: All examples.
    """
    __slots__ = 'pos', 'neg', 'all'

    def __init__(self,
                 pos: Set[OWLNamedIndividual],
                 neg: Set[OWLNamedIndividual],
                 all_instances: Optional[Set[OWLNamedIndividual]] = None):
        """
        Determine the learning problem and initialize the search.
        1) Convert the string representation of an individuals into the owlready2 representation.
        2) Sample negative examples if necessary.
        3) Initialize the root and search tree.

        Args:
            pos: Positive examples.
            neg: Negative examples.
            all_instances: All examples.
        """
        assert isinstance(pos, set) and isinstance(neg, set)
        self.pos = frozenset(pos)
        self.neg = frozenset(neg)
        if all_instances is None:
            self.all = None
        else:
            self.all = frozenset(all_instances)

    # def encode_kb(self, knowledge_base: 'KnowledgeBase') -> EncodedPosNegLPStandard:
    #     return knowledge_base.encode_learning_problem(self)

    def encode_kb(self, kb: 'AbstractKnowledgeBase') -> EncodedPosNegLPStandard:
        """
        Provides the encoded learning problem (lp), i.e. the class containing the set of OWLNamedIndividuals
        as follows:
            kb_pos --> the positive examples set,
            kb_neg --> the negative examples set,
            kb_all --> all lp individuals / all individuals set,
            kb_diff --> kb_all - (kb_pos + kb_neg).
        Args:
            kb (PosNegLPStandard): The knowledge base to encode the learning problem.
        Return:
            EncodedPosNegLPStandard: The encoded learning problem.
        """
        if self.all is None:
            kb_all = set(kb.individuals())
        else:
            kb_all = set(kb.individuals_set(self.all))

        assert 0 < len(self.pos) < len(kb_all) and len(kb_all) > len(self.neg)
        if logger.isEnabledFor(logging.INFO):
            r = DLSyntaxObjectRenderer()
            logger.info('E^+:[ {0} ]'.format(', '.join(map(r.render, self.pos))))
            logger.info('E^-:[ {0} ]'.format(', '.join(map(r.render, self.neg))))

        kb_pos = kb.individuals_set(self.pos)
        if len(self.neg) == 0:  # if negatives are not provided, randomly sample.
            kb_neg = type(kb_all)(random.sample(list(kb_all), len(kb_pos)))
        else:
            kb_neg = kb.individuals_set(self.neg)

        try:
            assert len(kb_pos) == len(self.pos)
        except AssertionError:
            print(self.pos)
            print(kb_pos)
            print(kb_all)
            print('Assertion error. Exiting.')
            raise
        if self.neg:
            assert len(kb_neg) == len(self.neg)

        return EncodedPosNegLPStandard(
            kb_pos=kb_pos,
            kb_neg=kb_neg,
            kb_all=kb_all,
            kb_diff=kb_all.difference(kb_pos.union(kb_neg)))

class EncodedPosNegUndLP(EncodedLearningProblem):
    """To be implemented."""
    ...
    # XXX: TODO


class PosNegUndLP(AbstractLearningProblem):
    """To be implemented."""
    ...
    # XXX: TODO
