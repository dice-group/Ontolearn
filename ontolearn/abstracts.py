from collections import OrderedDict
from functools import total_ordering
from abc import ABCMeta, abstractmethod, ABC
from owlready2 import ThingClass
from .util import get_full_iri
from typing import Set
import random

random.seed(0)


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

        self.is_atomic = True if self.form=='Class' else False#self.__is_atomic()  # TODO consider the necessity.
        self.length = self.__calculate_length()

        self.__instances = None

    @property
    def instances(self) -> Set:
        """ Returns all instances belonging to the concept."""
        if self.__instances:
            return self.__instances
        self.__instances = {get_full_iri(jjj) for jjj in self.owl.instances()}  # be sure of the memory usage.
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
    """
    An abstract class for quality and heuristic functions.
    """

    @abstractmethod
    def __init__(self, pos, neg, unlabelled):
        self.pos = pos
        self.neg = neg
        self.unlabelled = unlabelled
        self.applied = 0

    def set_positive_examples(self, instances):
        self.pos = instances

    def set_negative_examples(self, instances):
        self.neg = instances

    def set_unlabelled_examples(self, instances):
        self.unlabelled = instances


"""
class AbstractHeuristic(ABC):
    @abstractmethod
    def __init__(self, pos, neg, unlabelled):
        self.pos = pos
        self.neg = neg
        self.unlabelled = unlabelled
        self.applied = 0

    def set_positive_examples(self, instances):
        self.pos = instances

    def set_negative_examples(self, instances):
        self.neg = instances

    def set_unlabelled_examples(self, instances):
        self.unlabelled = instances

"""


class BaseRefinement(metaclass=ABCMeta):
    """
    Base class for Refinement Operators.

    Let C, D \in N_c where N_c os a finite set of concepts.

    * Proposition 3.3 (Complete and Finite Refinement Operators) [1]
        ** ρ(C) = {C ⊓ T} ∪ {D | D is not empty AND D \sqset C}
        *** The operator is finite,
        *** The operator is complete as given a concept C, we can reach an arbitrary concept D such that D subset of C.

    *) Theoretical Foundations of Refinement Operators [1].




    *) Defining a top-down refimenent operator that is a proper is crutial.
        4.1.3 Achieving Properness [1]
    *) Figure 4.1 [1] defines of the refinement operator

    [1] Learning OWL Class Expressions
    """

    @abstractmethod
    def __init__(self, kb):
        self.kb = kb
        self.concepts_to_nodes = dict()

    def set_kb(self, kb):
        self.kb = kb

    def set_concepts_node_mapping(self, m: dict):
        self.concepts_to_nodes = m

    @abstractmethod
    def getNode(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_atomic_concept(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_complement_of(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_some_values_from(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_all_values_from(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_union_of(self, *args, **kwargs):
        pass

    @abstractmethod
    def refine_object_intersection_of(self, *args, **kwargs):
        pass


class BaseNode(metaclass=ABCMeta):
    """Base class for Concept."""
    __slots__ = ['concept', '__heuristic_score', '__horizontal_expansion',
                 '__quality_score', 'parent_node', '___refinement_count',
                 '__refinement_count', '__depth', '__children']

    @abstractmethod
    def __init__(self, concept, parent_node, is_root=False):
        self.__quality_score, self.__heuristic_score = None, None
        self.__horizontal_expansion, self.__refinement_count = 0, 0
        self.concept = concept
        self.parent_node = parent_node
        self.__children = set()

        if self.parent_node is None:
            assert len(concept) == 1 and is_root
            self.__depth = 0
        else:
            self.__depth = self.parent_node.depth + 1

    def __len__(self):
        return len(self.concept)

    @property
    def children(self):
        return self.__children

    def add_children(self, n):
        self.__children.add(n)

    @property
    def refinement_count(self):
        return self.__refinement_count

    @refinement_count.setter
    def refinement_count(self, n):
        self.__refinement_count = n

    @property
    def depth(self):
        return self.__depth

    @depth.setter
    def depth(self, n: int):
        self.__depth = n

    @property
    def h_exp(self):
        return self.__horizontal_expansion

    @property
    def heuristic(self) -> float:
        return self.__heuristic_score

    @heuristic.setter
    def heuristic(self, val: float):
        self.__heuristic_score = val

    @property
    def quality(self) -> float:
        return self.__quality_score

    @quality.setter
    def quality(self, val: float):
        self.__quality_score = val

    def increment_h_exp(self, val=0):
        self.__horizontal_expansion += val + 1


class AbstractTree(ABC):
    @abstractmethod
    def __init__(self, quality_func, heuristic_func):
        self.expressionTests = 0
        self.quality_func = quality_func
        self.heuristic_func = heuristic_func
        self._nodes = dict()

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, item):
        return self._nodes[item]

    def __setitem__(self, k, v):
        self._nodes[k] = v

    def __iter__(self):
        for k, node in self._nodes.items():
            yield node

    def set_positive_negative_examples(self, *, p, n, unlabelled):
        """
        Assing positives and negatives
        """
        assert len(p) > 0
        if len(n) == 0:
            # randomly sample from unlabelled.
            n = random.sample(unlabelled)
        self.quality_func.set_positive_examples(p)
        self.quality_func.set_negative_examples(n)
        self.heuristic_func.set_positive_examples(p)
        self.heuristic_func.set_negative_examples(n)
        self.heuristic_func.set_unlabelled_examples(unlabelled)

    def set_quality_func(self, f: AbstractScorer):
        self.quality_func = f

    def set_heuristic_func(self, h):
        self.heuristic_func = h

    def redundancy_check(self, n):
        if n in self._nodes:
            return False
        return True

    @property
    def nodes(self):
        return self._nodes

    @abstractmethod
    def add_node(self, *args, **kwargs):
        pass

    def sort_search_tree_by_descending_heuristic_score(self):

        sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].heuristic, reverse=True)
        self._nodes = OrderedDict(sorted_x)

    def sort_search_tree_by_descending_quality(self):
        sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].quality, reverse=True)
        self._nodes = OrderedDict(sorted_x)

    def show_search_tree(self, ith, top_n=10):
        """
        Show search tree.
        """
        print('######## ', ith, 'step Search Tree ###########')
        counter = 1
        predictions=[]
        for k, v in enumerate(self):
            print(
                '{0}-\t{1}\t{2}:{3}\tHeuristic:{4}:'.format(counter, v.concept.str, self.quality_func.name,
                                                            v.quality, v.heuristic))
            # print('\t\t\t\t\t', counter, '-', v)  # , ' - acc:', v.accuracy)
            counter += 1
            predictions.append(v)
            if counter == top_n:
                break
        print('######## Search Tree ###########\n')
        return predictions
    def show_best_nodes(self, top_n):
        print('Number of times quality function applied: ', self.quality_func.applied)
        sorted_x = sorted(self.nodes.items(), key=lambda kv: kv[1].quality, reverse=True)
        self._nodes = OrderedDict(sorted_x)
        return self.show_search_tree('Final', top_n=top_n + 1)
