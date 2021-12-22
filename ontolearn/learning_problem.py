import logging
from typing import Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ontolearn import KnowledgeBase
from ontolearn.abstracts import AbstractLearningProblem, EncodedLearningProblem, EncodedPosNegLPStandardKind
from owlapy.model import OWLNamedIndividual

logger = logging.getLogger(__name__)


class EncodedPosNegLPStandard(EncodedPosNegLPStandardKind):
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

    def encode_kb(self, knowledge_base: 'KnowledgeBase') -> EncodedPosNegLPStandard:
        return knowledge_base.encode_learning_problem(self)


class EncodedPosNegUndLP(EncodedLearningProblem):
    ...
    # XXX: TODO


class PosNegUndLP(AbstractLearningProblem):
    ...
    # XXX: TODO
