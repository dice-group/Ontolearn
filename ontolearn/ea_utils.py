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

"""Utils for evolutionary algorithms."""

import random
from enum import Enum
from typing import Any, Callable, Dict, Final, Iterable, List, Optional, Tuple, Union

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
    name = name.replace('http://', '')
    name = name.replace('https://', '')
    name = name.replace('#', '_')
    name = name.replace('.', '_')
    name = name.replace('/', '_')
    name = name.replace(':', '_')

    return re.sub(r'[\W+]', '', name)


def _order_key(item: Any) -> Tuple[str, str]:
    if isinstance(item, OWLLiteral):
        return item.get_datatype().str, item.get_literal()
    return '', item.str


def sorted_for_reproducibility(items: Iterable[Any]) -> List[Any]:
    """Sort OWL entities (by IRI) and literals (by datatype, then lexical form) into a fixed order.

    Knowledge bases and reasoners may return entities in a different order in each process (e.g. from sets,
    whose iteration order depends on hashing). EvoLearner makes random choices by list position, so it puts
    such collections in this order first; otherwise seeded runs would differ between processes. Only the order
    changes, not the probabilities.
    """
    return sorted(items, key=_order_key)


def cx_one_point_ordered(ind1: Tree, ind2: Tree) -> Tuple[Tree, Tree]:
    """One-point crossover for strongly typed trees, with a reproducible choice of node type.

    Follows the typed branch of :func:`deap.gp.cxOnePoint`: pick a node type present in both trees uniformly at
    random, then swap a random subtree of that type between them. DEAP collects the shared types in a ``set`` of
    Python classes, which hash by memory address, so the type it picks can differ between processes even when
    ``random`` is seeded. Here the shared types are listed in the order they first occur in ``ind1``, so seeded runs
    repeat exactly; the probabilities are unchanged.

    Differences from :func:`deap.gp.cxOnePoint`: DEAP's untyped branch (taken when ``ind1.root.ret`` is the
    untyped ``__type__``) is not implemented, since EvoLearner's trees are always typed; and because the types are
    drawn from a list instead of a set, the sequence of random draws differs, so seeded results are not comparable
    with runs that used :func:`deap.gp.cxOnePoint`.

    Args:
        ind1: First tree participating in the crossover.
        ind2: Second tree participating in the crossover.

    Returns:
        The two trees, modified in place.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    types1: Dict[type, List[int]] = {}
    types2: Dict[type, List[int]] = {}
    for idx, node in enumerate(ind1[1:], 1):
        types1.setdefault(node.ret, []).append(idx)
    for idx, node in enumerate(ind2[1:], 1):
        types2.setdefault(node.ret, []).append(idx)
    common_types = [t for t in types1 if t in types2]

    if common_types:
        type_ = random.choice(common_types)
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def ind_to_string(ind: List[Tree]) -> str:
    return ''.join([prim.name for prim in ind])


# TODO: Ugly hack for now
def owlliteral_to_primitive_string(lit: OWLLiteral, pe: Optional[Union[OWLDataProperty, OWLObjectProperty]] = None) \
        -> str:
    str_ = type(lit.to_python()).__name__ + escape(lit.get_literal())
    if lit.get_datatype() in NUMERIC_DATATYPES:
        assert pe is not None
        return escape(pe.str) + str_
    return str_
