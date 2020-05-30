from abc import ABCMeta, abstractmethod, ABC
from functools import total_ordering
from abc import ABCMeta, abstractmethod
from owlready2 import ThingClass
from .util import get_full_iri
from typing import Set


@total_ordering
class BaseConcept(metaclass=ABCMeta):
    """Base class for Concept."""
    __slots__ = ['owl', 'full_iri', 'str', 'is_atomic', '__instances', 'length', 'form', 'role', 'filler', 'concept_a',
                 'concept_b']

    @abstractmethod
    def __init__(self, concept: ThingClass, kwargs):
        assert isinstance(concept, ThingClass)
        assert kwargs['form'] in ['Class', 'ObjectIntersectionOf', 'ObjectUnionOf', 'ObjectComplementOf',
                                  'ObjectSomeValuesFrom', 'ObjectAllValuesFrom']

        self.owl = concept
        self.full_iri = get_full_iri(concept)  # .namespace.base_iri + concept.name
        self.str = concept.name
        self.form = kwargs['form']

        self.is_atomic = self.__is_atomic()  # TODO consider the necessity.
        self.length = self.__calculate_length()

        self.__instances = None

    @property
    def instances(self) -> Set:
        """ Returns all instances belonging to the concept."""
        if self.__instances:
            return self.__instances
        self.__instances = {jjj for jjj in self.owl.instances()}  # be sure of the memory usage.
        return self.__instances

    @instances.setter
    def instances(self, x: Set):
        """ Setter of instances."""
        self.__instances = x

    def __str__(self):
        return '{self.__repr__}\t{self.full_iri}'.format(self=self)

    def __len__(self):
        return self.length

    def __calculate_length(self):
        """
        The length of a concept is defined as
        the sum of the numbers of
            concept names, role names, quantifiers,and connective symbols occurring in the concept

        The length |A| of a concept CAis defined inductively:
        |A| = |\top| = |\bot| = 1
        |¬D| = |D| + 1
        |D \sqcap E| = |D \sqcup E| = 1 + |D| + |E|
        |∃r.D| = |∀r.D| = 2 + |D|
        :return:
        """
        num_of_exists = self.str.count("∃")
        num_of_for_all = self.str.count("∀")
        num_of_negation = self.str.count("¬")
        is_dot_here = self.str.count('.')

        num_of_operand_and_operator = len(self.str.split())
        count = num_of_negation + num_of_operand_and_operator + num_of_exists + is_dot_here + num_of_for_all
        return count

    def __is_atomic(self):
        """
        @todo Atomic class definition must be explicitly defined.
        Currently we consider all concepts having length=1 as atomic.
        :return: True if self is atomic otherwise False.
        """
        if '∃' in self.str or '∀' in self.str:
            return False
        elif '⊔' in self.str or '⊓' in self.str or '¬' in self.str:
            return False
        return True

    def __lt__(self, other):
        return self.length < other.length

    def __gt__(self, other):
        return self.length > other.length


class AbstractScorer(ABC):
    @abstractmethod
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def apply(self, n):
        pass


class AbstractRefinement(ABC):
    @abstractmethod
    def __init__(self, kb):
        self.kb = kb

    @abstractmethod
    def refine(self, concept):
        pass

    @abstractmethod
    def refine_atomic_concept(self, concept):
        pass

    @abstractmethod
    def refine_complement_of(self, concept):
        pass

    @abstractmethod
    def refine_object_some_values_from(self, concept):
        pass

    @abstractmethod
    def refine_object_all_values_from(self, concept):
        pass

    @abstractmethod
    def refine_object_union_of(self, concept):
        pass

    @abstractmethod
    def refine_object_intersection_of(self, concept):
        pass


class AbstractNode(ABC):
    pass


class AbstractTree(ABC):
    pass
