"""Learning problem in Ontolearn."""
import logging
from typing import Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.abstracts import AbstractLearningProblem, EncodedLearningProblem, EncodedPosNegLPStandardKind
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

    def encode_kb(self, knowledge_base: 'KnowledgeBase') -> EncodedPosNegLPStandard:
        return knowledge_base.encode_learning_problem(self)


class EncodedPosNegUndLP(EncodedLearningProblem):
    """To be implemented."""
    ...
    # XXX: TODO


class PosNegUndLP(AbstractLearningProblem):
    """To be implemented."""
    ...
    # XXX: TODO
