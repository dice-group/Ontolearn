from enum import Enum
from typing import Callable, Final, List, Optional, Tuple, Union

from deap.gp import Primitive, Terminal
from owlapy.model import OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, OWLObjectUnionOf, \
    OWLClassExpression, OWLDataHasValue, OWLDataPropertyExpression, OWLDataSomeValuesFrom, OWLLiteral, \
    OWLObjectAllValuesFrom, OWLObjectIntersectionOf, NUMERIC_DATATYPES, OWLDataProperty, OWLObjectProperty, \
    OWLObjectExactCardinality, OWLObjectMaxCardinality, OWLObjectMinCardinality
from ontolearn.knowledge_base import KnowledgeBase
import re

from owlapy.model.providers import OWLDatatypeMinExclusiveRestriction, OWLDatatypeMinInclusiveRestriction, \
    OWLDatatypeMaxExclusiveRestriction, OWLDatatypeMaxInclusiveRestriction


Tree = List[Union[Primitive, Terminal]]


class PrimitiveFactory:

    __slots__ = 'knowledge_base'

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def create_union(self) -> Callable[[OWLClassExpression, OWLClassExpression], OWLObjectUnionOf]:

        def union(A: OWLClassExpression, B: OWLClassExpression) -> OWLObjectUnionOf:
            return self.knowledge_base.union([A, B])

        return union

    def create_intersection(self) -> Callable[[OWLClassExpression, OWLClassExpression], OWLObjectIntersectionOf]:

        def intersection(A: OWLClassExpression, B: OWLClassExpression) -> OWLObjectIntersectionOf:
            return self.knowledge_base.intersection([A, B])

        return intersection

    def create_existential_universal(self, property_: OWLObjectPropertyExpression) \
            -> Tuple[Callable[[OWLClassExpression], OWLObjectSomeValuesFrom],
                     Callable[[OWLClassExpression], OWLObjectAllValuesFrom]]:

        def existential_restriction(filler: OWLClassExpression) -> OWLObjectSomeValuesFrom:
            return self.knowledge_base.existential_restriction(filler, property_)

        def universal_restriction(filler: OWLClassExpression) -> OWLObjectAllValuesFrom:
            return self.knowledge_base.universal_restriction(filler, property_)

        return existential_restriction, universal_restriction

    def create_card_restrictions(self, property_: OWLObjectPropertyExpression) \
        -> Tuple[Callable[[int, OWLClassExpression], OWLObjectMinCardinality],
                 Callable[[int, OWLClassExpression], OWLObjectMaxCardinality],
                 Callable[[int, OWLClassExpression], OWLObjectExactCardinality]]:

        def min_cardinality(card: int, filler: OWLClassExpression) -> OWLObjectMinCardinality:
            return self.knowledge_base.min_cardinality_restriction(filler, property_, card)

        def max_cardinality(card: int, filler: OWLClassExpression) -> OWLObjectMaxCardinality:
            return self.knowledge_base.max_cardinality_restriction(filler, property_, card)

        def exact_cardinality(card: int, filler: OWLClassExpression) -> OWLObjectExactCardinality:
            return self.knowledge_base.exact_cardinality_restriction(filler, property_, card)

        return min_cardinality, max_cardinality, exact_cardinality

    def create_data_some_values(self, property_: OWLDataPropertyExpression) \
            -> Tuple[Callable[[OWLLiteral], OWLDataSomeValuesFrom], Callable[[OWLLiteral], OWLDataSomeValuesFrom],
                     Callable[[OWLLiteral], OWLDataSomeValuesFrom], Callable[[OWLLiteral], OWLDataSomeValuesFrom]]:

        def data_some_min_inclusive(value: OWLLiteral) -> OWLDataSomeValuesFrom:
            filler = OWLDatatypeMinInclusiveRestriction(value)
            return self.knowledge_base.data_existential_restriction(filler, property_)

        def data_some_max_inclusive(value: OWLLiteral) -> OWLDataSomeValuesFrom:
            filler = OWLDatatypeMaxInclusiveRestriction(value)
            return self.knowledge_base.data_existential_restriction(filler, property_)

        def data_some_min_exclusive(value: OWLLiteral) -> OWLDataSomeValuesFrom:
            filler = OWLDatatypeMinExclusiveRestriction(value)
            return self.knowledge_base.data_existential_restriction(filler, property_)

        def data_some_max_exclusive(value: OWLLiteral) -> OWLDataSomeValuesFrom:
            filler = OWLDatatypeMaxExclusiveRestriction(value)
            return self.knowledge_base.data_existential_restriction(filler, property_)

        return data_some_min_inclusive, data_some_max_inclusive, data_some_min_exclusive, data_some_max_exclusive

    def create_data_has_value(self, property_: OWLDataPropertyExpression) -> Callable[[OWLLiteral], OWLDataHasValue]:

        def data_has_value(value: OWLLiteral) -> OWLDataHasValue:
            return self.knowledge_base.data_has_value_restriction(value, property_)

        return data_has_value


class OperatorVocabulary(str, Enum):
    UNION: Final = "union"  #:
    INTERSECTION: Final = "intersection"  #:
    NEGATION: Final = "negation"  #:
    EXISTENTIAL: Final = "exists"  #:
    UNIVERSAL: Final = "forall"  #:
    INVERSE: Final = "inverse"  #:
    CARD_MIN: Final = "cardMin"  #:
    CARD_MAX: Final = "cardMax"  #:
    CARD_EXACT: Final = "cardExact"  #:
    DATA_MIN_INCLUSIVE: Final = "dataMinInc"  #:
    DATA_MAX_INCLUSIVE: Final = "dataMaxInc"  #:
    DATA_MIN_EXCLUSIVE: Final = "dataMinExc"  #:
    DATA_MAX_EXCLUSIVE: Final = "dataMaxExc"  #:
    DATA_HAS_VALUE: Final = "dataHasValue"  #:


class ToolboxVocabulary(str, Enum):
    MUTATION: Final = "mutate"  #:
    CROSSOVER: Final = "mate"  #:
    SELECTION: Final = "select"  #:
    COMPILE: Final = "compile"  #:
    INIT_POPULATION: Final = "population"  #:
    FITNESS_FUNCTION: Final = "apply_fitness"  #:
    HEIGHT_KEY: Final = "height"  #:


def escape(name: str) -> str:
    name = name.replace('-', 'minus')
    return re.sub(r'[\W+]', '', name)


def ind_to_string(ind: List[Tree]) -> str:
    return ''.join([prim.name for prim in ind])


# TODO: Ugly hack for now
def owlliteral_to_primitive_string(lit: OWLLiteral, pe: Optional[Union[OWLDataProperty, OWLObjectProperty]] = None) \
        -> str:
    str_ = type(lit.to_python()).__name__ + escape(lit.get_literal())
    if lit.get_datatype() in NUMERIC_DATATYPES:
        assert pe is not None
        return escape(pe.get_iri().get_remainder()) + str_
    return str_
