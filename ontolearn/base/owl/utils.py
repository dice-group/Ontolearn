from collections import Counter
from functools import singledispatchmethod
from typing import Iterable, Generic, TypeVar, Callable, List
from owlapy.class_expression import OWLObjectOneOf, OWLClass, OWLObjectUnionOf, OWLObjectIntersectionOf, \
    OWLObjectSomeValuesFrom, OWLObjectComplementOf, OWLObjectAllValuesFrom, OWLDataSomeValuesFrom, \
    OWLDatatypeRestriction, OWLClassExpression, OWLDataAllValuesFrom, OWLDataHasValue, OWLDataOneOf, \
    OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality, \
    OWLObjectHasValue, OWLDataExactCardinality, OWLDataMaxCardinality, \
    OWLDataMinCardinality, OWLObjectHasSelf, OWLObjectCardinalityRestriction, \
    OWLDataCardinalityRestriction, OWLThing
from owlapy.owl_data_ranges import OWLDataRange, OWLDataComplementOf, OWLDataIntersectionOf, OWLDataUnionOf
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_object import OWLObject
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty, OWLObjectInverseOf

from owlapy.util import OrderedOWLObject, iter_count
from sortedcontainers import SortedSet


class OWLClassExpressionLengthMetric:
    """Length calculation of OWLClassExpression

    Args:
        class_length: Class: "C"
        object_intersection_length: Intersection: A ⨅ B
        object_union_length: Union: A ⨆ B
        object_complement_length: Complement: ¬ C
        object_some_values_length: Obj. Some Values: ∃ r.C
        object_all_values_length: Obj. All Values: ∀ r.C
        object_has_value_length: Obj. Has Value: ∃ r.{I}
        object_cardinality_length: Obj. Cardinality restriction: ≤n r.C
        object_has_self_length: Obj. Self restriction: ∃ r.Self
        object_one_of_length: Obj. One of: ∃ r.{X,Y,Z}
        data_some_values_length: Data Some Values: ∃ p.t
        data_all_values_length: Data All Values: ∀ p.t
        data_has_value_length: Data Has Value: ∃ p.{V}
        data_cardinality_length: Data Cardinality restriction: ≤n r.t
        object_property_length: Obj. Property: ∃ r.C
        object_inverse_length: Inverse property: ∃ r⁻.C
        data_property_length: Data Property: ∃ p.t
        datatype_length: Datatype: ^^datatype
        data_one_of_length: Data One of: ∃ p.{U,V,W}
        data_complement_length: Data Complement: ¬datatype
        data_intersection_length: Data Intersection: datatype ⨅ datatype
        data_union_length: Data Union: datatype ⨆ datatype
    """

    __slots__ = 'class_length', 'object_intersection_length', 'object_union_length', 'object_complement_length', \
                'object_some_values_length', 'object_all_values_length', 'object_has_value_length', \
                'object_cardinality_length', 'object_has_self_length', 'object_one_of_length', \
                'data_some_values_length', 'data_all_values_length', 'data_has_value_length', \
                'data_cardinality_length', 'object_property_length', 'object_inverse_length', 'data_property_length', \
                'datatype_length', 'data_one_of_length', 'data_complement_length', 'data_intersection_length', \
                'data_union_length'

    class_length: int
    object_intersection_length: int
    object_union_length: int
    object_complement_length: int
    object_some_values_length: int
    object_all_values_length: int
    object_has_value_length: int
    object_cardinality_length: int
    object_has_self_length: int
    object_one_of_length: int
    data_some_values_length: int
    data_all_values_length: int
    data_has_value_length: int
    data_cardinality_length: int
    object_property_length: int
    object_inverse_length: int
    data_property_length: int
    datatype_length: int
    data_one_of_length: int
    data_complement_length: int
    data_intersection_length: int
    data_union_length: int

    def __init__(self, *,
                 class_length: int,
                 object_intersection_length: int,
                 object_union_length: int,
                 object_complement_length: int,
                 object_some_values_length: int,
                 object_all_values_length: int,
                 object_has_value_length: int,
                 object_cardinality_length: int,
                 object_has_self_length: int,
                 object_one_of_length: int,
                 data_some_values_length: int,
                 data_all_values_length: int,
                 data_has_value_length: int,
                 data_cardinality_length: int,
                 object_property_length: int,
                 object_inverse_length: int,
                 data_property_length: int,
                 datatype_length: int,
                 data_one_of_length: int,
                 data_complement_length: int,
                 data_intersection_length: int,
                 data_union_length: int,
                 ):
        self.class_length = class_length
        self.object_intersection_length = object_intersection_length
        self.object_union_length = object_union_length
        self.object_complement_length = object_complement_length
        self.object_some_values_length = object_some_values_length
        self.object_all_values_length = object_all_values_length
        self.object_has_value_length = object_has_value_length
        self.object_cardinality_length = object_cardinality_length
        self.object_has_self_length = object_has_self_length
        self.object_one_of_length = object_one_of_length
        self.data_some_values_length = data_some_values_length
        self.data_all_values_length = data_all_values_length
        self.data_has_value_length = data_has_value_length
        self.data_cardinality_length = data_cardinality_length
        self.object_property_length = object_property_length
        self.object_inverse_length = object_inverse_length
        self.data_property_length = data_property_length
        self.datatype_length = datatype_length
        self.data_one_of_length = data_one_of_length
        self.data_complement_length = data_complement_length
        self.data_intersection_length = data_intersection_length
        self.data_union_length = data_union_length

    @staticmethod
    def get_default() -> 'OWLClassExpressionLengthMetric':
        return OWLClassExpressionLengthMetric(
            class_length=1,
            object_intersection_length=1,
            object_union_length=1,
            object_complement_length=1,
            object_some_values_length=1,
            object_all_values_length=1,
            object_has_value_length=2,
            object_cardinality_length=2,
            object_has_self_length=1,
            object_one_of_length=1,
            data_some_values_length=1,
            data_all_values_length=1,
            data_has_value_length=2,
            data_cardinality_length=2,
            object_property_length=1,
            object_inverse_length=2,
            data_property_length=1,
            datatype_length=1,
            data_one_of_length=1,
            data_complement_length=1,
            data_intersection_length=1,
            data_union_length=1,
        )

    # single dispatch is still not implemented in mypy, see https://github.com/python/mypy/issues/2904
    @singledispatchmethod
    def length(self, o: OWLObject) -> int:
        raise NotImplementedError

    @length.register
    def _(self, o: OWLClass) -> int:
        return self.class_length

    @length.register
    def _(self, p: OWLObjectProperty) -> int:
        return self.object_property_length

    @length.register
    def _(self, e: OWLObjectSomeValuesFrom) -> int:
        return self.object_some_values_length \
               + self.length(e.get_property()) \
               + self.length(e.get_filler())

    @length.register
    def _(self, e: OWLObjectAllValuesFrom) -> int:
        return self.object_all_values_length \
               + self.length(e.get_property()) \
               + self.length(e.get_filler())

    @length.register
    def _(self, c: OWLObjectUnionOf) -> int:
        length = -self.object_union_length
        for op in c.operands():
            length += self.length(op) + self.object_union_length

        return length

    @length.register
    def _(self, c: OWLObjectIntersectionOf) -> int:
        length = -self.object_intersection_length
        for op in c.operands():
            length += self.length(op) + self.object_intersection_length

        return length

    @length.register
    def _(self, n: OWLObjectComplementOf) -> int:
        return self.length(n.get_operand()) + self.object_complement_length

    @length.register
    def _(self, p: OWLObjectInverseOf) -> int:
        return self.object_inverse_length

    @length.register
    def _(self, e: OWLObjectCardinalityRestriction) -> int:
        return self.object_cardinality_length \
               + self.length(e.get_property()) \
               + self.length(e.get_filler())

    @length.register
    def _(self, s: OWLObjectHasSelf) -> int:
        return self.object_has_self_length + self.length(s.get_property())

    @length.register
    def _(self, v: OWLObjectHasValue) -> int:
        return self.object_has_value_length + self.length(v.get_property())

    @length.register
    def _(self, o: OWLObjectOneOf) -> int:
        return self.object_one_of_length

    @length.register
    def _(self, p: OWLDataProperty) -> int:
        return self.data_property_length

    @length.register
    def _(self, e: OWLDataSomeValuesFrom) -> int:
        return self.data_some_values_length \
               + self.length(e.get_property()) \
               + self.length(e.get_filler())

    @length.register
    def _(self, e: OWLDataAllValuesFrom) -> int:
        return self.data_all_values_length \
               + self.length(e.get_property()) \
               + self.length(e.get_filler())

    @length.register
    def _(self, e: OWLDataCardinalityRestriction) -> int:
        return self.data_cardinality_length \
               + self.length(e.get_property()) \
               + self.length(e.get_filler())

    @length.register
    def _(self, v: OWLDataHasValue) -> int:
        return self.data_has_value_length + self.length(v.get_property())

    @length.register
    def _(self, o: OWLDataOneOf) -> int:
        return self.data_one_of_length

    @length.register
    def _(self, n: OWLDatatypeRestriction) -> int:
        return iter_count(n.get_facet_restrictions())

    @length.register
    def _(self, n: OWLDataComplementOf) -> int:
        return self.data_complement_length + self.length(n.get_data_range())

    @length.register
    def _(self, c: OWLDataUnionOf) -> int:
        length = -self.data_union_length
        for op in c.operands():
            length += self.length(op) + self.data_union_length
        return length

    @length.register
    def _(self, c: OWLDataIntersectionOf) -> int:
        length = -self.data_intersection_length
        for op in c.operands():
            length += self.length(op) + self.data_intersection_length
        return length

    @length.register
    def _(self, t: OWLDatatype) -> int:
        return self.datatype_length


_N = TypeVar('_N')  #:
_O = TypeVar('_O')  #:


class EvaluatedDescriptionSet(Generic[_N, _O]):
    __slots__ = 'items', '_max_size', '_Ordering'

    items: 'SortedSet[_N]'
    _max_size: int
    _Ordering: Callable[[_N], _O]

    def __init__(self, ordering: Callable[[_N], _O], max_size: int = 10):
        self._max_size = max_size
        self._Ordering = ordering
        self.items = SortedSet(key=self._Ordering)

    def maybe_add(self, node: _N):
        if len(self.items) == self._max_size:
            worst = self.items[0]
            if self._Ordering(node) > self._Ordering(worst):
                self.items.pop(0)
                self.items.add(node)
                return True
        else:
            self.items.add(node)
            return True
        return False

    def clean(self):
        self.items.clear()

    def worst(self):
        return self.items[0]

    def best(self):
        return self.items[-1]

    def best_quality_value(self) -> float:
        return self.items[-1].quality

    def __iter__(self) -> Iterable[_N]:
        yield from reversed(self.items)


def _avoid_overly_redundand_operands(operands: List[_O], max_count: int = 2) -> List[_O]:
    _max_count = max_count
    r = []
    counts = Counter(operands)
    for op in sorted(operands, key=OrderedOWLObject):
        for _ in range(min(_max_count, counts[op])):
            r.append(op)
    return r


def _sort_by_ordered_owl_object(i: Iterable[_O]) -> Iterable[_O]:
    return sorted(i, key=OrderedOWLObject)


class ConceptOperandSorter:
    # single dispatch is still not implemented in mypy, see https://github.com/python/mypy/issues/2904
    @singledispatchmethod
    def sort(self, o: _O) -> _O:
        raise NotImplementedError(o)

    @sort.register
    def _(self, o: OWLClass) -> OWLClass:
        return o

    @sort.register
    def _(self, p: OWLObjectProperty) -> OWLObjectProperty:
        return p

    @sort.register
    def _(self, p: OWLDataProperty) -> OWLDataProperty:
        return p

    @sort.register
    def _(self, i: OWLNamedIndividual) -> OWLNamedIndividual:
        return i

    @sort.register
    def _(self, i: OWLLiteral) -> OWLLiteral:
        return i

    @sort.register
    def _(self, e: OWLObjectSomeValuesFrom) -> OWLObjectSomeValuesFrom:
        t = OWLObjectSomeValuesFrom(property=e.get_property(), filler=self.sort(e.get_filler()))
        if t == e:
            return e
        else:
            return t

    @sort.register
    def _(self, e: OWLObjectAllValuesFrom) -> OWLObjectAllValuesFrom:
        t = OWLObjectAllValuesFrom(property=e.get_property(), filler=self.sort(e.get_filler()))
        if t == e:
            return e
        else:
            return t

    @sort.register
    def _(self, c: OWLObjectUnionOf) -> OWLObjectUnionOf:
        t = OWLObjectUnionOf(_sort_by_ordered_owl_object(c.operands()))
        if t == c:
            return c
        else:
            return t

    @sort.register
    def _(self, c: OWLObjectIntersectionOf) -> OWLObjectIntersectionOf:
        t = OWLObjectIntersectionOf(_sort_by_ordered_owl_object(c.operands()))
        if t == c:
            return c
        else:
            return t

    @sort.register
    def _(self, n: OWLObjectComplementOf) -> OWLObjectComplementOf:
        return n

    @sort.register
    def _(self, p: OWLObjectInverseOf) -> OWLObjectInverseOf:
        return p

    @sort.register
    def _(self, r: OWLObjectMinCardinality) -> OWLObjectMinCardinality:
        t = OWLObjectMinCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                    filler=self.sort(r.get_filler()))
        if t == r:
            return r
        else:
            return t

    @sort.register
    def _(self, r: OWLObjectExactCardinality) -> OWLObjectExactCardinality:
        t = OWLObjectExactCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                      filler=self.sort(r.get_filler()))
        if t == r:
            return r
        else:
            return t

    @sort.register
    def _(self, r: OWLObjectMaxCardinality) -> OWLObjectMaxCardinality:
        t = OWLObjectMaxCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                    filler=self.sort(r.get_filler()))
        if t == r:
            return r
        else:
            return t

    @sort.register
    def _(self, r: OWLObjectHasSelf) -> OWLObjectHasSelf:
        return r

    @sort.register
    def _(self, r: OWLObjectHasValue) -> OWLObjectHasValue:
        return r

    @sort.register
    def _(self, r: OWLObjectOneOf) -> OWLObjectOneOf:
        t = OWLObjectOneOf(_sort_by_ordered_owl_object(r.individuals()))
        if t == r:
            return r
        else:
            return t

    @sort.register
    def _(self, e: OWLDataSomeValuesFrom) -> OWLDataSomeValuesFrom:
        t = OWLDataSomeValuesFrom(property=e.get_property(), filler=self.sort(e.get_filler()))
        if t == e:
            return e
        else:
            return t

    @sort.register
    def _(self, e: OWLDataAllValuesFrom) -> OWLDataAllValuesFrom:
        t = OWLDataAllValuesFrom(property=e.get_property(), filler=self.sort(e.get_filler()))
        if t == e:
            return e
        else:
            return t

    @sort.register
    def _(self, c: OWLDataUnionOf) -> OWLDataUnionOf:
        t = OWLDataUnionOf(_sort_by_ordered_owl_object(c.operands()))
        if t == c:
            return c
        else:
            return t

    @sort.register
    def _(self, c: OWLDataIntersectionOf) -> OWLDataIntersectionOf:
        t = OWLDataIntersectionOf(_sort_by_ordered_owl_object(c.operands()))
        if t == c:
            return c
        else:
            return t

    @sort.register
    def _(self, n: OWLDataComplementOf) -> OWLDataComplementOf:
        return n

    @sort.register
    def _(self, n: OWLDatatypeRestriction) -> OWLDatatypeRestriction:
        t = OWLDatatypeRestriction(n.get_datatype(), _sort_by_ordered_owl_object(n.get_facet_restrictions()))
        if t == n:
            return n
        else:
            return t

    @sort.register
    def _(self, d: OWLDatatype) -> OWLDatatype:
        return d

    @sort.register
    def _(self, r: OWLDataMinCardinality) -> OWLDataMinCardinality:
        t = OWLDataMinCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                  filler=self.sort(r.get_filler()))
        if t == r:
            return r
        else:
            return t

    @sort.register
    def _(self, r: OWLDataExactCardinality) -> OWLDataExactCardinality:
        t = OWLDataExactCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                    filler=self.sort(r.get_filler()))
        if t == r:
            return r
        else:
            return t

    @sort.register
    def _(self, r: OWLDataMaxCardinality) -> OWLDataMaxCardinality:
        t = OWLDataMaxCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                  filler=self.sort(r.get_filler()))
        if t == r:
            return r
        else:
            return t

    @sort.register
    def _(self, r: OWLDataHasValue) -> OWLDataHasValue:
        return r

    @sort.register
    def _(self, n: OWLDataOneOf) -> OWLDataOneOf:
        t = OWLDataOneOf(_sort_by_ordered_owl_object(n.values()))
        if t == n:
            return n
        else:
            return t


class OperandSetTransform:
    def simplify(self, o: OWLClassExpression) -> OWLClassExpression:
        return self._simplify(o).get_nnf()

    # single dispatch is still not implemented in mypy, see https://github.com/python/mypy/issues/2904
    @singledispatchmethod
    def _simplify(self, o: _O) -> _O:
        raise NotImplementedError(o)

    @_simplify.register
    def _(self, o: OWLClass) -> OWLClass:
        return o

    @_simplify.register
    def _(self, p: OWLObjectProperty) -> OWLObjectProperty:
        return p

    @_simplify.register
    def _(self, p: OWLDataProperty) -> OWLDataProperty:
        return p

    @_simplify.register
    def _(self, i: OWLNamedIndividual) -> OWLNamedIndividual:
        return i

    @_simplify.register
    def _(self, i: OWLLiteral) -> OWLLiteral:
        return i

    @_simplify.register
    def _(self, e: OWLObjectSomeValuesFrom) -> OWLObjectSomeValuesFrom:
        return OWLObjectSomeValuesFrom(property=e.get_property(), filler=self._simplify(e.get_filler()))

    @_simplify.register
    def _(self, e: OWLObjectAllValuesFrom) -> OWLObjectAllValuesFrom:
        return OWLObjectAllValuesFrom(property=e.get_property(), filler=self._simplify(e.get_filler()))

    @_simplify.register
    def _(self, c: OWLObjectUnionOf) -> OWLClassExpression:
        s = set(map(self._simplify, set(c.operands())))
        if OWLThing in s:
            return OWLThing
        elif len(s) == 1:
            return s.pop()
        return OWLObjectUnionOf(_sort_by_ordered_owl_object(s))

    @_simplify.register
    def _(self, c: OWLObjectIntersectionOf) -> OWLClassExpression:
        s = set(map(self._simplify, set(c.operands())))
        s.discard(OWLThing)
        if not s:
            return OWLThing
        elif len(s) == 1:
            return s.pop()
        return OWLObjectIntersectionOf(_sort_by_ordered_owl_object(s))

    @_simplify.register
    def _(self, n: OWLObjectComplementOf) -> OWLObjectComplementOf:
        return n

    @_simplify.register
    def _(self, p: OWLObjectInverseOf) -> OWLObjectInverseOf:
        return p

    @_simplify.register
    def _(self, r: OWLObjectMinCardinality) -> OWLObjectMinCardinality:
        return OWLObjectMinCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                       filler=self._simplify(r.get_filler()))

    @_simplify.register
    def _(self, r: OWLObjectExactCardinality) -> OWLObjectExactCardinality:
        return OWLObjectExactCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                         filler=self._simplify(r.get_filler()))

    @_simplify.register
    def _(self, r: OWLObjectMaxCardinality) -> OWLObjectMaxCardinality:
        return OWLObjectMaxCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                       filler=self._simplify(r.get_filler()))

    @_simplify.register
    def _(self, r: OWLObjectHasSelf) -> OWLObjectHasSelf:
        return r

    @_simplify.register
    def _(self, r: OWLObjectHasValue) -> OWLObjectHasValue:
        return r

    @_simplify.register
    def _(self, r: OWLObjectOneOf) -> OWLObjectOneOf:
        return OWLObjectOneOf(_sort_by_ordered_owl_object(set(r.individuals())))

    @_simplify.register
    def _(self, e: OWLDataSomeValuesFrom) -> OWLDataSomeValuesFrom:
        return OWLDataSomeValuesFrom(property=e.get_property(), filler=self._simplify(e.get_filler()))

    @_simplify.register
    def _(self, e: OWLDataAllValuesFrom) -> OWLDataAllValuesFrom:
        return OWLDataAllValuesFrom(property=e.get_property(), filler=self._simplify(e.get_filler()))

    @_simplify.register
    def _(self, c: OWLDataUnionOf) -> OWLDataRange:
        s = set(map(self._simplify, set(c.operands())))
        if len(s) == 1:
            return s.pop()
        return OWLDataUnionOf(_sort_by_ordered_owl_object(s))

    @_simplify.register
    def _(self, c: OWLDataIntersectionOf) -> OWLDataRange:
        s = set(map(self._simplify, set(c.operands())))
        if len(s) == 1:
            return s.pop()
        return OWLDataIntersectionOf(_sort_by_ordered_owl_object(s))

    @_simplify.register
    def _(self, n: OWLDatatypeRestriction) -> OWLDatatypeRestriction:
        return n

    @_simplify.register
    def _(self, n: OWLDataComplementOf) -> OWLDataComplementOf:
        return n

    @_simplify.register
    def _(self, r: OWLDataMinCardinality) -> OWLDataMinCardinality:
        return OWLDataMinCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                     filler=self._simplify(r.get_filler()))

    @_simplify.register
    def _(self, r: OWLDataExactCardinality) -> OWLDataExactCardinality:
        return OWLDataExactCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                       filler=self._simplify(r.get_filler()))

    @_simplify.register
    def _(self, r: OWLDataMaxCardinality) -> OWLDataMaxCardinality:
        return OWLDataMaxCardinality(cardinality=r.get_cardinality(), property=r.get_property(),
                                     filler=self._simplify(r.get_filler()))

    @_simplify.register
    def _(self, r: OWLDataHasValue) -> OWLDataHasValue:
        return r

    @_simplify.register
    def _(self, r: OWLDataOneOf) -> OWLDataOneOf:
        return OWLDataOneOf(_sort_by_ordered_owl_object(set(r.values())))
