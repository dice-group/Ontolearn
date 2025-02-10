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

"""The main abstract classes."""

import logging
from abc import ABCMeta, abstractmethod
from typing import Set, List, Tuple, Iterable, TypeVar, Generic, ClassVar, Optional
from collections import OrderedDict
from owlapy.class_expression import OWLClassExpression
from owlapy.abstracts import AbstractOWLOntology
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.utils import iter_count
from .utils.static_funcs import concept_len

_N = TypeVar('_N')  #:

logger = logging.getLogger(__name__)

# @TODO:CD: Each Class definiton in abstract.py should share a prefix, e.g., BaseX or AbstractX.
# @TODO:CD: All imports must be located on top of the script
from owlapy import owl_expression_to_dl
class EncodedLearningProblem(metaclass=ABCMeta):
    """Encoded Abstract learning problem for use in Scorers."""
    __slots__ = ()


class EncodedPosNegLPStandardKind(EncodedLearningProblem, metaclass=ABCMeta):
    """Encoded Abstract learning problem following pos-neg lp standard."""
    __slots__ = ()


# @TODO: Why we need Generic[_N] and if we need it why we di not use it in all other abstract classes?
class AbstractScorer(Generic[_N], metaclass=ABCMeta):
    """
    An abstract class for quality functions.
    """
    __slots__ = ()

    name: ClassVar[str]

    def __init__(self, *args, **kwargs):
        """Create a new quality function."""
        pass

    def score_elp(self, instances: set, learning_problem: EncodedLearningProblem) -> Tuple[bool, Optional[float]]:
        """Quality score for a set of instances with regard to the learning problem.

        Args:
            instances (set): Instances to calculate a quality score for.
            learning_problem: Underlying learning problem to compare the quality to.

        Returns:
             Tuple, first position indicating if the function could be applied, second position the quality value
                in the range 0.0--1.0.
        """
        if len(instances) == 0:
            return False, 0
        # @TODO: It must be moved to the top of the abstracts.py
        from ontolearn.learning_problem import EncodedPosNegLPStandard
        if isinstance(learning_problem, EncodedPosNegLPStandard):
            tp = len(learning_problem.kb_pos.intersection(instances))
            tn = len(learning_problem.kb_neg.difference(instances))

            fp = len(learning_problem.kb_neg.intersection(instances))
            fn = len(learning_problem.kb_pos.difference(instances))
            return self.score2(tp=tp, tn=tn, fp=fp, fn=fn)
        else:
            raise NotImplementedError(learning_problem)

    @abstractmethod
    def score2(self, tp: int, fn: int, fp: int, tn: int) -> Tuple[bool, Optional[float]]:
        """Quality score for a coverage count.

        Args:
            tp: True positive count.
            fn: False negative count.
            fp: False positive count.
            tn: True negative count.

        Returns:
             Tuple, first position indicating if the function could be applied, second position the quality value
                in the range 0.0--1.0.
        """
        pass

    # @TODO:CD: Why there is '..' in AbstractNode
    def apply(self, node: 'AbstractNode', instances, learning_problem: EncodedLearningProblem) -> bool:  # pragma: no cover
        """Apply the quality function to a search tree node after calculating the quality score on the given instances.

        Args:
            node: search tree node to set the quality on.
            instances (set): Instances to calculate the quality for.
            learning_problem: Underlying learning problem to compare the quality to.

        Returns:
            True if the quality function was applied successfully
        """

        assert isinstance(learning_problem, EncodedLearningProblem), \
            f'Expected EncodedLearningProblem but got {type(learning_problem)}'
        assert isinstance(node, AbstractNode), \
            f'Expected AbstractNode but got {type(node)}'
        # @TODO: It must be moved to the top of the abstracts.py
        from ontolearn.search import _NodeQuality
        assert isinstance(node, _NodeQuality), \
            f'Expected _NodeQuality but got {type(_NodeQuality)}'

        ret, q = self.score_elp(instances, learning_problem)
        if q is not None:
            node.quality = q
        return ret


class AbstractHeuristic(Generic[_N], metaclass=ABCMeta):
    """Abstract base class for heuristic functions.

    Heuristic functions can guide the search process."""
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        """Create a new heuristic function."""
        pass

    @abstractmethod
    def apply(self, node: _N, instances, learning_problem: EncodedLearningProblem):
        """Apply the heuristic on a search tree node and set its heuristic property to the calculated value.

        Args:
            node: Node to set the heuristic on.
            instances (set, optional): Set of instances covered by this node.
            learning_problem: Underlying learning problem to compare the heuristic to.
        """
        pass


class AbstractFitness(metaclass=ABCMeta):
    """Abstract base class for fitness functions.

    Fitness functions guide the evolutionary process."""
    __slots__ = ()

    name: ClassVar[str]

    @abstractmethod
    def __init__(self):
        """Create a new fitness function."""
        pass

    @abstractmethod
    def apply(self, individual):
        """Apply the fitness function on an individual and set its fitness attribute to the calculated value.

        Args:
            individual: Individual to set the fitness on.
        """
        pass


class AbstractNode(metaclass=ABCMeta):
    """Abstract search tree node."""
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        """Create an abstract search tree node."""
        pass

    def __str__(self):
        """String representation of node, by default its internal memory address."""
        addr = hex(id(self))
        addr = addr[0:2] + addr[6:-1]
        return f'{type(self)} at {addr}'

    def __repr__(self):
        return self.__str__()

class AbstractOEHeuristicNode(metaclass=ABCMeta):
    """Abstract Node for the CELOEHeuristic heuristic function.

    This node must support quality, horizontal expansion (h_exp), is_root, parent_node and refinement_count.
    """
    __slots__ = ()

    @property
    @abstractmethod
    def quality(self) -> Optional[float]:
        """Get the quality of the node.

        Returns:
            Quality of the node.
        """
        pass

    @property
    @abstractmethod
    def h_exp(self) -> int:
        """Get horizontal expansion.

        Returns:
            Horizontal expansion.
        """
        pass

    @property
    @abstractmethod
    def is_root(self) -> bool:
        """Is this the root node?

        Returns:
            True if this is the root node, otherwise False.
        """
        pass

    @property
    @abstractmethod
    def parent_node(self: _N) -> Optional[_N]:
        """Get the parent node.

        Returns:
            Parent node.
        """
        pass

    @property
    @abstractmethod
    def refinement_count(self) -> int:
        """Get the refinement count for this node.

        Returns:
            Refinement count.
        """
        pass

    @property
    @abstractmethod
    def heuristic(self) -> Optional[float]:
        """Get the heuristic value.

        Returns:
            Heuristic value.
        """
        pass

    @heuristic.setter
    @abstractmethod
    def heuristic(self, v: float):
        """Set the heuristic value."""
        pass


class AbstractConceptNode(metaclass=ABCMeta):
    """Abstract search tree node which has a concept."""
    __slots__ = ()

    @property
    @abstractmethod
    def concept(self) -> OWLClassExpression:
        """Get the concept representing this node.

        Returns:
            The concept representing this node.
        """
        pass


class AbstractKnowledgeBase(metaclass=ABCMeta):
    """Abstract knowledge base."""
    __slots__ = ()

    # CD: This function is used as "a get method". Insteadf either access the atttribute directly
    # or use it as a property @abstractmethod
    def ontology(self) -> AbstractOWLOntology:
        """The base ontology of this knowledge base."""
        pass

    def describe(self) -> None:
        """Print a short description of the Knowledge Base to the info logger output."""
        properties_count = iter_count(self.ontology.object_properties_in_signature()) + iter_count(
            self.ontology.data_properties_in_signature())
        logger.info(f'Number of named classes: {iter_count(self.ontology.classes_in_signature())}\n'
                    f'Number of individuals: {self.individuals_count()}\n'
                    f'Number of properties: {properties_count}')

    @abstractmethod
    def individuals_count(self) -> int:
        """Total number of individuals in this knowledge base."""
        pass

    @abstractmethod
    def individuals_set(self, *args, **kwargs) -> Set:
        """Encode an individual, an iterable of individuals or the individuals that are instances of a given concept
        into a set.

        Args:
            arg (OWLNamedIndividual): Individual to encode.
            arg (Iterable[OWLNamedIndividual]): Individuals to encode.
            arg (OWLClassExpression): Encode individuals that are instances of this concept.

        Returns:
            Encoded set representation of individual(s).
        """
        pass

    @abstractmethod
    def individuals(self, concept: Optional[OWLClassExpression] = None, named_individuals: bool = False) -> Iterable[OWLNamedIndividual]:
        pass

    @abstractmethod
    def abox(self, *args, **kwargs):
        pass

    @abstractmethod
    def tbox(self, *args, **kwargs):
        pass

    @abstractmethod
    def triples(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_object_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def data_properties_for_domain(self, *args, **kwargs):
        pass

    @abstractmethod
    def least_general_named_concepts(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_classes(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_object_property_domains(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_object_property_ranges(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_data_property_domains(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_data_property_ranges(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_boolean_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_numeric_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_time_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_existential_restrictions(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_universal_restrictions(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_existential_restrictions_inverse(self, *args, **kwargs):
        pass

    @abstractmethod
    def most_general_universal_restrictions_inverse(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_direct_parents(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_all_direct_sub_concepts(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_all_sub_concepts(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_concepts(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def concepts(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def object_properties(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_object_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_boolean_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_numeric_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_double_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_time_data_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_types(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_object_properties_for_ind(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_data_properties_for_ind(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_object_property_values(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_data_property_values(self, *args, **kwargs):
        pass

    @abstractmethod
    def contains_class(self, *args, **kwargs):
        pass

    @abstractmethod
    def are_owl_concept_disjoint(self, *args, **kwargs):
        pass


class BaseRefinement(Generic[_N], metaclass=ABCMeta):
    """
    Base class for Refinement Operators.

    Let C, D \\in N_c where N_c os a finite set of concepts.

    * Proposition 3.3 (Complete and Finite Refinement Operators) [1]
      * ρ(C) = {C ⊓ T} ∪ {D \\| D is not empty AND D \\sqset C}
        * The operator is finite,
        * The operator is complete as given a concept C, we can reach an arbitrary concept D such that D subset of C.

    *) Theoretical Foundations of Refinement Operators [1].

    *) Defining a top-down refimenent operator that is a proper is crutial.
        4.1.3 Achieving Properness [1]
    *) Figure 4.1 [1] defines of the refinement operator.

    [1] Learning OWL Class Expressions.

    Attributes:
        kb (AbstractKnowledgeBase): The knowledge base used by this refinement operator.
    """
    __slots__ = 'kb'

    kb: AbstractKnowledgeBase

    @abstractmethod
    def __init__(self, knowledge_base: AbstractKnowledgeBase):
        """Construct a new base refinement operator.

        Args:
            knowledge_base: Knowledge base to operate on.
        """
        self.kb = knowledge_base

    @abstractmethod
    def refine(self, *args, **kwargs) -> Iterable[OWLClassExpression]:
        """Refine a given concept.

        Args:
            ce (OWLClassExpression): Concept to refine.

        Returns:
            New refined concepts.
        """
        pass

    def len(self, concept: OWLClassExpression) -> int:
        """The length of a concept.

        Args:
            concept: The concept to measure the length for.

        Returns:
            Length of concept according to some metric configured in the knowledge base.
        """
        return concept_len(concept)


class AbstractLearningProblem(metaclass=ABCMeta):
    """Abstract learning problem."""
    __slots__ = ()

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Create a new abstract learning problem."""
        pass

    @abstractmethod
    def encode_kb(self, knowledge_base: AbstractKnowledgeBase) -> 'EncodedLearningProblem':
        """Encode the learning problem into the knowledge base."""
        pass


class LBLSearchTree(Generic[_N], metaclass=ABCMeta):
    """Abstract search tree for the Length based learner."""

    @abstractmethod
    def get_most_promising(self) -> _N:
        """Find most "promising" node in the search tree that should be refined next.

        Returns:
            Most promising search tree node.
        """
        pass

    @abstractmethod
    def add_node(self, node: _N, parent_node: _N, kb_learning_problem: EncodedLearningProblem):
        """Add a node to the search tree.

        Args:
            node: Node to add.
            parent_node: Parent of that node.
            kb_learning_problem: Underlying learning problem to compare the quality to.
        """
        pass

    @abstractmethod
    def clean(self):
        """Reset the search tree state."""
        pass

    @abstractmethod
    def get_top_n(self, n: int) -> List[_N]:
        """Retrieve the best n search tree nodes.

        Args:
            n: Maximum number of nodes.

        Returns:
            List of top n search tree nodes.
        """
        pass

    @abstractmethod
    def show_search_tree(self, root_concept: OWLClassExpression, heading_step: str):
        """Debugging function to print the search tree to standard output.

        Args:
            root_concept: The tree is printed starting from this search tree node.
            heading_step: Message to print at top of the output.
        """
        pass

    @abstractmethod
    def add_root(self, node: _N, kb_learning_problem: EncodedLearningProblem):
        """Add the root node to the search tree.

        Args:
            node: Root node to add.
            kb_learning_problem: Underlying learning problem to compare the quality to.
        """
        pass


class DRILLAbstractTree:  # pragma: no cover
    """Abstract Tree for DRILL."""
    @abstractmethod
    def __init__(self):
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

    def get_top_n_nodes(self, n: int, key='quality'):
        self.sort_search_tree_by_decreasing_order(key=key)
        for ith, dict_ in enumerate(self._nodes.items()):
            if ith >= n:
                break
            k, node = dict_
            yield node

    def redundancy_check(self, n):
        if n in self._nodes:
            return False
        return True

    @property
    def nodes(self):
        return self._nodes

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    def sort_search_tree_by_decreasing_order(self, *, key: str):
        if key == 'heuristic':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].heuristic, reverse=True)
        elif key == 'quality':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: kv[1].quality, reverse=True)
        elif key == 'length':
            sorted_x = sorted(self._nodes.items(), key=lambda kv: len(kv[1]), reverse=True)
        else:
            raise ValueError('Wrong Key. Key must be heuristic, quality or concept_length')

        self._nodes = OrderedDict(sorted_x)

    def best_hypotheses(self, n=10) -> List:
        assert self.search_tree is not None
        assert len(self.search_tree) > 1
        return [i for i in self.search_tree.get_top_n_nodes(n)]

    def show_search_tree(self, top_n=100):
        """
        Show search tree.
        """
        predictions = list(self.get_top_n_nodes(top_n))
        print('######## Search Tree ###########\n')
        for ith, node in enumerate(predictions):
            print(f"{ith + 1}-\t{owl_expression_to_dl(node.concept)} | Quality:{node.quality}| Heuristic:{node.heuristic}")
        print('\n######## Search Tree ###########\n')
        return predictions



    def show_best_nodes(self, top_n, key=None):
        assert key
        self.sort_search_tree_by_decreasing_order(key=key)
        return self.show_search_tree('Final', top_n=top_n + 1)

    @staticmethod
    def save_current_top_n_nodes(key=None, n=10, path=None):
        """
        Save current top_n nodes.
        """
        assert path
        assert key
        assert isinstance(n, int)
        pass

    def clean(self):
        self._nodes.clear()
