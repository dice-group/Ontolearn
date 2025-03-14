"""Utils for evolutionary algorithms."""

from enum import Enum
from typing import Callable, Final, List, Optional, Tuple, Union

from deap.gp import Primitive, Terminal
from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectUnionOf, OWLClassExpression, OWLDataHasValue, \
    OWLDataSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectIntersectionOf, OWLObjectExactCardinality, \
    OWLObjectMaxCardinality, OWLObjectMinCardinality
from owlapy.owl_literal import OWLLiteral, NUMERIC_DATATYPES
from owlapy.owl_property import OWLObjectPropertyExpression, OWLDataPropertyExpression, OWLDataProperty, \
    OWLObjectProperty

from ontolearn.concept_generator import ConceptGenerator
import re

from owlapy.providers import owl_datatype_min_exclusive_restriction, owl_datatype_min_inclusive_restriction, \
    owl_datatype_max_exclusive_restriction, owl_datatype_max_inclusive_restriction


Tree = List[Union[Primitive, Terminal]]


class PrimitiveFactory:

    __slots__ = 'generator'

    def __init__(self):
        self.generator = ConceptGenerator()

    def create_union(self) -> Callable[[OWLClassExpression, OWLClassExpression], OWLObjectUnionOf]:

        def union(A: OWLClassExpression, B: OWLClassExpression) -> OWLObjectUnionOf:
            return self.generator.union([A, B])

        return union

    def create_intersection(self) -> Callable[[OWLClassExpression, OWLClassExpression], OWLObjectIntersectionOf]:

        def intersection(A: OWLClassExpression, B: OWLClassExpression) -> OWLObjectIntersectionOf:
            return self.generator.intersection([A, B])

        return intersection

    def create_existential_universal(self, property_: OWLObjectPropertyExpression) \
            -> Tuple[Callable[[OWLClassExpression], OWLObjectSomeValuesFrom],
                     Callable[[OWLClassExpression], OWLObjectAllValuesFrom]]:

        def existential_restriction(filler: OWLClassExpression) -> OWLObjectSomeValuesFrom:
            return self.generator.existential_restriction(filler, property_)

        def universal_restriction(filler: OWLClassExpression) -> OWLObjectAllValuesFrom:
            return self.generator.universal_restriction(filler, property_)

        return existential_restriction, universal_restriction

    def create_card_restrictions(self, property_: OWLObjectPropertyExpression) \
        -> Tuple[Callable[[int, OWLClassExpression], OWLObjectMinCardinality],
                 Callable[[int, OWLClassExpression], OWLObjectMaxCardinality],
                 Callable[[int, OWLClassExpression], OWLObjectExactCardinality]]:

        def min_cardinality(card: int, filler: OWLClassExpression) -> OWLObjectMinCardinality:
            return self.generator.min_cardinality_restriction(filler, property_, card)

        def max_cardinality(card: int, filler: OWLClassExpression) -> OWLObjectMaxCardinality:
            return self.generator.max_cardinality_restriction(filler, property_, card)

        def exact_cardinality(card: int, filler: OWLClassExpression) -> OWLObjectExactCardinality:
            return self.generator.exact_cardinality_restriction(filler, property_, card)

        return min_cardinality, max_cardinality, exact_cardinality

    def create_data_some_values(self, property_: OWLDataPropertyExpression) \
            -> Tuple[Callable[[OWLLiteral], OWLDataSomeValuesFrom], Callable[[OWLLiteral], OWLDataSomeValuesFrom],
                     Callable[[OWLLiteral], OWLDataSomeValuesFrom], Callable[[OWLLiteral], OWLDataSomeValuesFrom]]:

        def data_some_min_inclusive(value: OWLLiteral) -> OWLDataSomeValuesFrom:
            filler = owl_datatype_min_inclusive_restriction(value)
            return self.generator.data_existential_restriction(filler, property_)

        def data_some_max_inclusive(value: OWLLiteral) -> OWLDataSomeValuesFrom:
            filler = owl_datatype_max_inclusive_restriction(value)
            return self.generator.data_existential_restriction(filler, property_)

        def data_some_min_exclusive(value: OWLLiteral) -> OWLDataSomeValuesFrom:
            filler = owl_datatype_min_exclusive_restriction(value)
            return self.generator.data_existential_restriction(filler, property_)

        def data_some_max_exclusive(value: OWLLiteral) -> OWLDataSomeValuesFrom:
            filler = owl_datatype_max_exclusive_restriction(value)
            return self.generator.data_existential_restriction(filler, property_)

        return data_some_min_inclusive, data_some_max_inclusive, data_some_min_exclusive, data_some_max_exclusive

    def create_data_has_value(self, property_: OWLDataPropertyExpression) -> Callable[[OWLLiteral], OWLDataHasValue]:

        def data_has_value(value: OWLLiteral) -> OWLDataHasValue:
            return self.generator.data_has_value_restriction(value, property_)

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
        return escape(pe.iri.get_remainder()) + str_
    return str_
