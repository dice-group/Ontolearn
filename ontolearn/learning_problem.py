import logging
import random
from typing import Set, Optional

from ontolearn import KnowledgeBase
from ontolearn.abstracts import AbstractLearningProblem, EncodedLearningProblem
from owlapy.model import OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer

logger = logging.getLogger(__name__)


class EncodedPosNegLPStandard(EncodedLearningProblem):
    __slots__ = 'kb_pos', 'kb_neg', 'kb_diff', 'kb_all'

    kb_pos: set
    kb_neg: set
    kb_diff: set
    kb_all: set

    def __init__(self, kb_pos, kb_neg, kb_diff, kb_all):
        self.kb_pos = kb_pos
        self.kb_neg = kb_neg
        self.kb_diff = kb_diff
        self.kb_all = kb_all


class PosNegLPStandard(AbstractLearningProblem):
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
        """
        assert isinstance(pos, set) and isinstance(neg, set)
        self.pos = frozenset(pos)
        self.neg = frozenset(neg)
        if all_instances is None:
            self.all = None
        else:
            self.all = frozenset(all_instances)

    def encode_kb(self, knowledge_base: KnowledgeBase) -> EncodedPosNegLPStandard:
        assert len(knowledge_base.class_hierarchy()) > 0

        if self.all is None:
            kb_all = knowledge_base.all_individuals_set()
        else:
            kb_all = knowledge_base.individuals_set(self.all)

        assert 0 < len(self.pos) < len(kb_all) and len(kb_all) > len(self.neg)
        if logger.isEnabledFor(logging.INFO):
            r = DLSyntaxObjectRenderer()
            logger.info('E^+:[ {0} ]'.format(', '.join(map(r.render, self.pos))))
            logger.info('E^-:[ {0} ]'.format(', '.join(map(r.render, self.neg))))

        kb_pos = knowledge_base.individuals_set(self.pos)
        if len(self.neg) == 0:  # if negatives are not provided, randomly sample.
            kb_neg = type(kb_all)(random.sample(list(kb_all), len(kb_pos)))
        else:
            kb_neg = knowledge_base.individuals_set(self.neg)

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
    ...
    # XXX: TODO


class PosNegUndLP(AbstractLearningProblem):
    ...
    # XXX: TODO
