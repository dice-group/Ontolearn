"""Initialization for evolutionary algorithms."""

from dataclasses import dataclass
from functools import lru_cache
from enum import Enum, auto
from itertools import chain, cycle

from owlapy.class_expression import OWLClass, OWLClassExpression, OWLThing
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty

from ontolearn.ea_utils import OperatorVocabulary, Tree, escape, owlliteral_to_primitive_string
from ontolearn.knowledge_base import KnowledgeBase
import random
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Final, List, Set, Union
from deap.gp import Primitive, PrimitiveSetTyped


class RandomInitMethod(Enum):
    GROW: Final = auto()  #:
    FULL: Final = auto()  #:
    RAMPED_HALF_HALF: Final = auto()  #:


class AbstractEAInitialization(metaclass=ABCMeta):
    """Abstract base class for initialization methods for evolutionary algorithms.

    """
    __slots__ = ()

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_population(self, container: Callable, pset: PrimitiveSetTyped, population_size: int = 0) -> List[Tree]:
        pass

    @abstractmethod
    def get_expression(self, pset: PrimitiveSetTyped) -> Tree:
        pass


class EARandomInitialization(AbstractEAInitialization):
    """Rnndom initialization methods for evolutionary algorithms.

    """
    __slots__ = 'min_height', 'max_height', 'method'

    min_height: int
    max_height: int
    method: RandomInitMethod

    def __init__(self, min_height: int = 3, max_height: int = 6,
                 method: RandomInitMethod = RandomInitMethod.RAMPED_HALF_HALF):
        """
        Args:
            min_height: Minimum height of trees.
            max_height: Maximum height of trees.
            method: Random initialization method possible values: rhh, grow, full.
        """
        self.min_height = min_height
        self.max_height = max_height
        self.method = method

    def get_population(self, container: Callable, pset: PrimitiveSetTyped, population_size: int = 0) -> List[Tree]:
        return [container(self.get_expression(pset)) for _ in range(population_size)]

    def get_expression(self, pset: PrimitiveSetTyped, type_: type = None) -> Tree:
        if type_ is None:
            type_ = pset.ret

        use_grow = (self.method == RandomInitMethod.GROW or
                    (self.method == RandomInitMethod.RAMPED_HALF_HALF and random.random() < 0.5))

        expr: Tree = []
        height = random.randint(self.min_height, self.max_height)
        self._build_tree(expr, pset, height, 0, type_, use_grow)
        return expr

    def _build_tree(self, tree,
                    pset: PrimitiveSetTyped,
                    height: int,
                    current_height: int,
                    type_: type,
                    use_grow: bool):

        if current_height == height or len(pset.primitives[type_]) == 0:
            tree.append(random.choice(pset.terminals[type_]))
        else:
            operators = []
            if use_grow and current_height >= self.min_height:
                operators = pset.primitives[type_] + pset.terminals[type_]
            else:
                operators = pset.primitives[type_]

            operator = random.choice(operators)
            tree.append(operator)

            if isinstance(operator, Primitive):
                for arg_type in operator.args:
                    self._build_tree(tree, pset, height, current_height+1, arg_type, use_grow)


Property = Union[OWLObjectProperty, OWLDataProperty]
Object = Union[OWLNamedIndividual, OWLLiteral]


@dataclass(frozen=True)
class PropObjPair:
    property_: Property
    object_: Object


class EARandomWalkInitialization(AbstractEAInitialization):
    """Random walk initialization for description logic learning.

    """
    __slots__ = 'max_t', 'jump_pr', 'type_counts', 'dp_to_prim_type', 'dp_splits', 'kb'

    connection_pr: float = 0.5
    _cache_size: int = 2048

    max_t: int
    jump_pr: float

    type_counts: Dict[OWLClass, int]
    dp_to_prim_type: Dict[OWLDataProperty, Any]
    dp_splits: Dict[OWLDataProperty, List[OWLLiteral]]
    kb: KnowledgeBase

    def __init__(self, max_t: int = 2, jump_pr: float = 0.5):
        """
        Random walk initialization for description logic learning.
        Args:
            max_t: Number of paths.
            jump_pr: Probability to explore paths of length 2.
        """
        self.max_t = max_t
        self.jump_pr = jump_pr

        self.type_counts = dict()
        self.dp_to_prim_type = dict()
        self.dp_splits = dict()

    def get_population(self, container: Callable,
                       pset: PrimitiveSetTyped,
                       population_size: int = 0,
                       pos: List[OWLNamedIndividual] = None,
                       dp_to_prim_type: Dict[OWLDataProperty, Any] = None,
                       dp_splits: Dict[OWLDataProperty, List[OWLLiteral]] = None,
                       kb: KnowledgeBase = None) -> List[Tree]:
        assert pos is not None
        assert kb is not None
        assert dp_to_prim_type is not None
        assert dp_splits is not None

        self.dp_to_prim_type = dp_to_prim_type
        self.dp_splits = dp_splits
        self.kb = kb
        self.type_counts = self._compute_type_counts(pos)

        count = 0
        population = []
        for ind in cycle(pos):
            population.append(container(self.get_expression(pset, ind)))
            count += 1
            if count == population_size:
                break

        return population

    def get_expression(self, pset: PrimitiveSetTyped, ind: OWLNamedIndividual = None) -> Tree:
        assert ind is not None
        type_ = self._select_type(ind)
        pairs = self._select_pairs(self._get_properties(ind), ind)

        expr: Tree = []
        if len(pairs) > 0:
            self._add_intersection_or_union(expr, pset)
        self._add_object_terminal(expr, pset, type_)

        for idx, pair in enumerate(pairs):
            if idx != len(pairs) - 1:
                self._add_intersection_or_union(expr, pset)

            if isinstance(pair.property_, OWLObjectProperty):
                self._build_object_property(expr, ind, pair, pset)
            elif isinstance(pair.property_, OWLDataProperty):
                if pair.property_ in self.kb.get_boolean_data_properties():
                    self._build_bool_property(expr, pair, pset)
                elif pair.property_ in chain(self.kb.get_time_data_properties(), self.kb.get_numeric_data_properties()):
                    self._build_split_property(expr, pair, pset)
                else:
                    raise NotImplementedError(pair.property_)

        return expr

    def _compute_type_counts(self, pos: List[OWLNamedIndividual]) -> Dict[OWLClass, int]:
        types = chain.from_iterable((self._get_types(ind, direct=True) for ind in pos))
        type_counts = dict.fromkeys(types, 0)

        for ind in pos:
            common_types = type_counts.keys() & self._get_types(ind)
            for t in common_types:
                type_counts[t] += 1

        return type_counts

    def _select_type(self, ind: OWLNamedIndividual) -> OWLClass:
        types_ind = list(self.type_counts.keys() & self._get_types(ind))
        weights = [self.type_counts[t] for t in types_ind]
        return random.choices(types_ind, weights=weights)[0]

    @lru_cache(maxsize=_cache_size)
    def _get_types(self, ind: OWLNamedIndividual, direct: bool = False) -> Set[OWLClass]:
        inds = set(self.kb.get_types(ind, direct))
        return inds if inds else {OWLThing}

    @lru_cache(maxsize=_cache_size)
    def _get_properties(self, ind: OWLNamedIndividual) -> List[Property]:
        properties: List[Property] = list(self.kb.get_object_properties_for_ind(ind))
        for p in self.kb.get_data_properties_for_ind(ind):
            if p in self.dp_to_prim_type:
                properties.append(p)
        return properties

    def _select_pairs(self, properties: List[Property], ind: OWLNamedIndividual) -> List[PropObjPair]:
        ind_nbrs: Dict[Property, List[Object]] = dict()
        ind_nbrs = {p: self._get_property_values(ind, p) for p in properties}

        pairs = []
        if len(properties) < self.max_t:
            pairs = [PropObjPair(p, random.choice(ind_nbrs[p])) for p in properties]
        else:
            temp_props = random.sample(properties, k=self.max_t)
            pairs = [PropObjPair(p, random.choice(ind_nbrs[p])) for p in temp_props]

        # If not enough pairs selected, also taking duplicate properties to different objects
        temp_pairs = []
        if len(pairs) < self.max_t:
            temp_pairs = [PropObjPair(p, o) for p in properties for o in ind_nbrs[p] if PropObjPair(p, o) not in pairs]

            remaining_pairs = self.max_t - len(pairs)
            if len(temp_pairs) > remaining_pairs:
                pairs += random.sample(temp_pairs, k=remaining_pairs)
            else:
                pairs += temp_pairs

        return pairs

    def _build_object_property(self, expr: Tree, ind: OWLNamedIndividual, pair: PropObjPair, pset: PrimitiveSetTyped):
        assert isinstance(pair.property_, OWLObjectProperty)
        self._add_primitive(expr, pset, pair.property_, OperatorVocabulary.EXISTENTIAL)

        second_ind = pair.object_
        assert isinstance(second_ind, OWLNamedIndividual)

        properties = self._get_properties(second_ind)

        # Select next path while prohibiting a loop back to the first individual
        next_pair = None
        while next_pair is None and len(properties) > 1:
            temp_prop = random.choice(properties)
            objs = self._get_property_values(second_ind, temp_prop)

            if isinstance(temp_prop, OWLObjectProperty):
                try:
                    objs.remove(ind)
                except ValueError:
                    pass

            if len(objs) > 0:
                next_pair = PropObjPair(temp_prop, random.choice(objs))

            properties.remove(temp_prop)

        if next_pair is not None and random.random() < self.jump_pr:
            if isinstance(next_pair.property_, OWLObjectProperty):
                self._add_primitive(expr, pset, next_pair.property_, OperatorVocabulary.EXISTENTIAL)
                assert isinstance(next_pair.object_, OWLNamedIndividual)
                type_ = random.choice(list(self._get_types(next_pair.object_)))
                self._add_object_terminal(expr, pset, type_)
            elif isinstance(next_pair.property_, OWLDataProperty):
                if next_pair.property_ in self.kb.get_boolean_data_properties():
                    self._build_bool_property(expr, next_pair, pset)
                elif next_pair.property_ in chain(self.kb.get_time_data_properties(),
                                                  self.kb.get_numeric_data_properties()):
                    self._build_split_property(expr, next_pair, pset)
            else:
                raise NotImplementedError(next_pair.property_)

        else:
            type_ = random.choice(list(self._get_types(second_ind)))
            self._add_object_terminal(expr, pset, type_)

    def _build_bool_property(self, expr: Tree, pair: PropObjPair, pset: PrimitiveSetTyped):
        assert isinstance(pair.property_, OWLDataProperty)
        assert isinstance(pair.object_, OWLLiteral)

        self._add_primitive(expr, pset, pair.property_, OperatorVocabulary.DATA_HAS_VALUE)
        self._add_data_terminal(expr, pset, pair.property_, pair.object_)

    def _build_split_property(self, expr: Tree, pair: PropObjPair, pset: PrimitiveSetTyped):
        assert isinstance(pair.property_, OWLDataProperty)
        assert isinstance(pair.object_, OWLLiteral)

        splits = self.dp_splits[pair.property_]
        nearest_value = min(splits, key=lambda k: abs(k.to_python()-pair.object_.to_python())) if len(splits) > 0 else 0
        vocab = OperatorVocabulary.DATA_MIN_INCLUSIVE \
            if nearest_value.to_python() <= pair.object_.to_python() else OperatorVocabulary.DATA_MAX_INCLUSIVE

        self._add_primitive(expr, pset, pair.property_, vocab)
        self._add_data_terminal(expr, pset, pair.property_, nearest_value)

    @lru_cache(maxsize=_cache_size)
    def _get_property_values(self, ind: OWLNamedIndividual, property_: Property) -> List[Object]:
        if isinstance(property_, OWLObjectProperty):
            return list(self.kb.get_object_property_values(ind, property_))
        elif isinstance(property_, OWLDataProperty):
            return list(self.kb.get_data_property_values(ind, property_))
        else:
            raise NotImplementedError(property_)

    def _add_intersection_or_union(self, expr: Tree, pset: PrimitiveSetTyped):
        if random.random() <= EARandomWalkInitialization.connection_pr:
            expr.append(pset.primitives[OWLClassExpression][2])
        else:
            expr.append(pset.primitives[OWLClassExpression][1])

    def _add_object_terminal(self, expr: Tree, pset: PrimitiveSetTyped, type_: OWLClass):
        for t in pset.terminals[OWLClassExpression]:
            if t.name == escape(type_.iri.get_remainder()):
                expr.append(t)
                return

    def _add_data_terminal(self, expr: Tree, pset: PrimitiveSetTyped, property_: OWLDataProperty, object_: OWLLiteral):
        for t in pset.terminals[self.dp_to_prim_type[property_]]:
            if t.name == owlliteral_to_primitive_string(object_, property_):
                expr.append(t)
                return

    def _add_primitive(self, expr: Tree, pset: PrimitiveSetTyped, property_: Property, vocab: OperatorVocabulary):
        for p in pset.primitives[OWLClassExpression]:
            if p.name == vocab + escape(property_.iri.get_remainder()):
                expr.append(p)
                return
