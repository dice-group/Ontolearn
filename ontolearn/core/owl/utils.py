from functools import singledispatchmethod, total_ordering
from typing import cast, Set, Iterable

from ontolearn.owlapy import HasIRI
from ontolearn.owlapy.base import HasIndex
from ontolearn.owlapy.model import OWLObject, OWLClass, OWLObjectProperty, OWLObjectSomeValuesFrom, \
    OWLObjectAllValuesFrom, OWLObjectUnionOf, OWLObjectIntersectionOf, OWLObjectComplementOf, OWLObjectInverseOf, \
    OWLObjectCardinalityRestriction, OWLObjectHasSelf, \
    OWLObjectHasValue, OWLObjectOneOf, OWLObjectRestriction, HasFiller, HasCardinality, HasOperands
from ontolearn.owlapy.utils import as_index, popcount, iter_bits


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
        object_propery_length: Obj. Property: ∃ r.C
        object_inverse_length: Inverse property: ∃ r⁻.C
        data_propery_length: Data Property: ∃ p.t
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
                'data_cardinality_length', 'object_propery_length', 'object_inverse_length', 'data_propery_length', \
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
    object_propery_length: int
    object_inverse_length: int
    data_propery_length: int
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
                 object_propery_length: int,
                 object_inverse_length: int,
                 data_propery_length: int,
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
        self.object_propery_length = object_propery_length
        self.object_inverse_length = object_inverse_length
        self.data_propery_length = data_propery_length
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
            object_propery_length=1,
            object_inverse_length=2,
            data_propery_length=1,
            datatype_length=1,
            data_one_of_length=1,
            data_complement_length=1,
            data_intersection_length=1,
            data_union_length=1,
        )

    @singledispatchmethod
    def length(self, o: OWLObject) -> int:
        raise NotImplementedError

    @length.register
    def _(self, o: OWLClass) -> int:
        return self.class_length

    @length.register
    def _(self, p: OWLObjectProperty) -> int:
        return self.object_propery_length

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
    def _(self, v: OWLObjectHasValue):
        return self.object_has_value_length + self.length(v.get_property())

    @length.register
    def _(self, o: OWLObjectOneOf):
        return self.object_one_of_length

    # TODO
    # @length.register
    # def _(self, n: OWLDatatypeRestriction):
    #     return iter_count(n.facet_restrictions())

    # TODO
    # @length.register
    # def _(self, t: OWLDatatype):
    #     return self.datatype_length


@total_ordering
class OrderedOWLObject:
    __slots__ = 'o'

    # we are limited by https://github.com/python/typing/issues/213 # o: Intersection[OWLObject, HasIndex]
    def __init__(self, o: HasIndex):
        self.o = o

    def __lt__(self, other):
        cs = [self.o.type_index]
        co = [other.o.type_index]

        if cs == co:
            if isinstance(self.o, OWLObjectRestriction):
                cs.append(OrderedOWLObject(as_index(self.o.get_property())))
                co.append(OrderedOWLObject(other.o.get_property()))
            if isinstance(self.o, HasFiller):
                cs.append(OrderedOWLObject(self.o.get_filler()))
                co.append(OrderedOWLObject(other.o.get_filler()))
            if isinstance(self.o, HasCardinality):
                cs.append(self.o.get_cardinality())
                co.append(other.o.get_cardinality())
            if isinstance(self.o, HasOperands):
                cs.append(tuple(map(OrderedOWLObject, self.o.operands())))
                co.append(tuple(map(OrderedOWLObject, other.o.operands())))
            if isinstance(self.o, HasIRI):
                cs.append(self.o.get_iri().as_str())
                co.append(other.o.get_iri().as_str())

        return tuple(cs) < tuple(co)

    def __eq__(self, other):
        cs = [self.o.type_index]
        co = [other.o.type_index]

        if cs == co:
            if isinstance(self.o, OWLObjectRestriction):
                cs.append(OrderedOWLObject(as_index(self.o.get_property())))
                co.append(OrderedOWLObject(other.o.get_property()))
            if isinstance(self.o, HasFiller):
                cs.append(OrderedOWLObject(self.o.get_filler()))
                co.append(OrderedOWLObject(other.o.get_filler()))
            if isinstance(self.o, HasCardinality):
                cs.append(self.o.get_cardinality())
                co.append(other.o.get_cardinality())
            if isinstance(self.o, HasOperands):
                cs.append(tuple(map(OrderedOWLObject, self.o.operands())))
                co.append(tuple(map(OrderedOWLObject, other.o.operands())))
            if isinstance(self.o, HasIRI):
                cs.append(self.o.get_iri().as_str())
                co.append(other.o.get_iri().as_str())

        return tuple(cs) == tuple(co)


class BitSet:
    """
    set() -> new empty set object
    set(iterable) -> new set object

    Build an unordered collection of unique elements.
    """
    __slots__ = 'v'

    def difference(self, b: 'BitSet') -> 'BitSet':
        """
        Return the difference of two sets as a new set.

        (i.e. all elements that are in this set but not the others.)
        """
        return BitSet(self.v & ~b.v)

    def intersection(self, b: 'BitSet') -> 'BitSet':
        """
        Return the intersection of two sets as a new set.

        (i.e. all elements that are in both sets.)
        """
        return BitSet(self.v & b.v)

    def isdisjoint(self, b: 'BitSet') -> bool:
        """ Return True if two sets have a null intersection. """
        return not self.v & b.v

    def issubset(self, b: 'BitSet') -> bool:
        """ Report whether another set contains this set. """
        return (self.v & b.v) == self.v

    def issuperset(self, b: 'BitSet') -> bool:
        """ Report whether this set contains another set. """
        return (self.v & b.v) == b.v

    def symmetric_difference(self, b: 'BitSet'):
        """
        Return the symmetric difference of two sets as a new set.

        (i.e. all elements that are in exactly one of the sets.)
        """
        return BitSet(self.v ^ b.v)

    def union(self, b: 'BitSet'):
        """
        Return the union of two sets as a new set.

        (i.e. all elements that are in either set.)
        """
        return BitSet(self.v | b.v)

    def __and__(self, b: 'BitSet') -> 'BitSet':
        """ Return self&value. """
        return self.intersection(b)

    def __contains__(self, y) -> bool:
        """ x.__contains__(y) <==> y in x. """
        if isinstance(y, BitSet):
            return self.issuperset(y)
        elif self.v & y:
            return True
        else:
            return False

    def __eq__(self, b: 'BitSet') -> bool:
        """ Return self==value. """
        return self.v == b.v

    def __ge__(self, b: 'BitSet') -> bool:
        """ Return self>=value. """
        return self.issuperset(b)

    def __gt__(self, b: 'BitSet') -> bool:
        """ Return self>value. """
        return self.v != b.v and self.issuperset(b)

    def __init__(self, v: int = 0):
        """
        BitSet() -> new empty BitSet object
        BitSet(value) -> new BitSet object with bits in value
        """
        self.v = v

    def __iter__(self) -> Iterable[int]:
        """ Implement iter(self). """
        yield from iter_bits(self.v)

    def __len__(self):
        """ Return len(self). """
        return popcount(self.v)

    def __le__(self, b: 'BitSet') -> bool:
        """ Return self<=value. """
        return self.issubset(b)

    def __lt__(self, b: 'BitSet') -> bool:
        """ Return self<value. """
        return self.v != b.v and self.issubset(b)

    def __ne__(self, b: 'BitSet') -> bool:
        """ Return self!=value. """
        return self.v != b.v

    def __or__(self, b: 'BitSet') -> 'BitSet':
        """ Return self|value. """
        return self.union(b)

    def __rand__(self, b: 'BitSet') -> 'BitSet':  # real signature unknown
        """ Return value&self. """
        return b.intersection(self)

    def __repr__(self) -> str:
        """ Return repr(self). """
        return f'BitSet({bin(self.v)})'

    def __ror__(self, b: 'BitSet') -> 'BitSet':
        """ Return value|self. """
        return b.union(self)

    def __rsub__(self, b: 'BitSet') -> 'BitSet':
        """ Return value-self. """
        return b.difference(self)

    def __rxor__(self, b: 'BitSet') -> 'BitSet':
        """ Return value^self. """
        return b.symmetric_difference(self)

    def __sub__(self, b: 'BitSet') -> 'BitSet':
        """ Return self-value. """
        return self.difference(b)

    def __xor__(self, b: 'BitSet') -> 'BitSet':
        """ Return self^value. """
        return self.symmetric_difference(b)

    def __hash__(self):
        return self.v
