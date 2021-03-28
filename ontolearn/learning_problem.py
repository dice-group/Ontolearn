import logging
import random
from typing import Set, Optional

from ontolearn import KnowledgeBase
from ontolearn.abstracts import AbstractLearningProblem
from owlapy.model import OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer

logger = logging.getLogger(__name__)


class PosNegLPStandard(AbstractLearningProblem):
    __slots__ = 'kb_pos', 'kb_neg', 'kb_diff', 'kb_all'

    kb: KnowledgeBase

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 pos: Set[OWLNamedIndividual],
                 neg: Set[OWLNamedIndividual],
                 all_instances: Optional[Set[OWLNamedIndividual]] = None):
        """
        Determine the learning problem and initialize the search.
        1) Convert the string representation of an individuals into the owlready2 representation.
        2) Sample negative examples if necessary.
        3) Initialize the root and search tree.
        """
        super().__init__(knowledge_base)

        assert len(self.kb.class_hierarchy()) > 0

        if all_instances is None:
            kb_all = self.kb.all_individuals_set()
        else:
            kb_all = self.kb.individuals_set(all_instances)

        assert isinstance(pos, set) and isinstance(neg, set)
        assert 0 < len(pos) < len(kb_all) and len(kb_all) > len(neg)
        if logger.isEnabledFor(logging.INFO):
            r = DLSyntaxObjectRenderer()
            logger.info('E^+:[ {0} ]'.format(', '.join(map(r.render, pos))))
            logger.info('E^-:[ {0} ]'.format(', '.join(map(r.render, neg))))

        kb_pos = self.kb.individuals_set(pos)
        if len(neg) == 0:  # if negatives are not provided, randomly sample.
            kb_neg = type(kb_all)(random.sample(list(kb_all), len(kb_pos)))
        else:
            kb_neg = self.kb.individuals_set(neg)

        try:
            assert len(kb_pos) == len(pos)
        except AssertionError:
            print(pos)
            print(kb_pos)
            print(kb_all)
            print('Assertion error. Exiting.')
            raise
        assert len(kb_neg) == len(neg)

        self.kb_pos = kb_pos
        self.kb_neg = kb_neg
        self.kb_all = kb_all
        self.kb_diff = kb_all.difference(kb_pos.union(kb_neg))
