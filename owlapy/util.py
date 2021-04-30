from functools import singledispatchmethod
from typing import Iterable, overload, TypeVar, Generic, Type, Tuple, Dict, List, cast

from owlapy import IRI
from owlapy.model import OWLObject, HasIndex, HasIRI, OWLClassExpression, OWLClass, OWLObjectIntersectionOf, \
    OWLObjectUnionOf, OWLObjectComplementOf, OWLNothing, OWLThing, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectHasValue, OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality, OWLObjectHasSelf, \
    OWLObjectOneOf, OWLDataMaxCardinality, OWLDataMinCardinality, OWLDataExactCardinality, OWLDataHasValue, \
    OWLDataAllValuesFrom, OWLDataSomeValuesFrom

_HasIRI = TypeVar('_HasIRI', bound=HasIRI)  #:

class NNF:
    @singledispatchmethod
    def get_class_nnf(self, ce: OWLClassExpression, negated: bool = False):
        raise NotImplementedError

    @get_class_nnf.register
    def _(self, ce: OWLClass, negated: bool = False):
        if negated:
            if ce.is_owl_thing():
                return OWLNothing
            if ce.is_owl_nothing():
                return OWLThing
            return OWLObjectComplementOf(ce)
        return ce

    @get_class_nnf.register
    def _(self, ce: OWLObjectIntersectionOf, negated: bool = False):
        ops = map(lambda _: self.get_class_nnf(_, negated), ce.operands())
        if negated:
            return OWLObjectUnionOf(ops)
        return OWLObjectIntersectionOf(ops)

    @get_class_nnf.register
    def _(self, ce: OWLObjectUnionOf, negated: bool = False):
        ops = map(lambda _: self.get_class_nnf(_, negated), ce.operands())
        if negated:
            return OWLObjectIntersectionOf(ce)
        return OWLObjectUnionOf(ops)

    @get_class_nnf.register
    def _(self, ce: OWLObjectComplementOf, negated: bool = False):
        return self.get_class_nnf(ce.get_operand(), not negated)
    
    @get_class_nnf.register
    def _(self, ce: OWLObjectSomeValuesFrom, negated: bool = False):
        filler = self.get_class_nnf(ce.get_filler(), negated)
        if negated:
            return OWLObjectAllValuesFrom(ce.get_property(), filler)
        return OWLObjectSomeValuesFrom(ce.get_property(), filler)
    
    @get_class_nnf.register
    def _(self, ce: OWLObjectAllValuesFrom, negated: bool = False):
        filler = self.get_class_nnf(ce.get_filler(), negated)
        if negated:
            return OWLObjectSomeValuesFrom(ce.get_property(), filler)
        return OWLObjectAllValuesFrom(ce.get_property(), filler)
    
    @get_class_nnf.register
    def _(self, ce: OWLObjectHasValue, negated: bool = False):
        return self.get_class_nnf(ce.as_some_values_from(), negated)
    
    @get_class_nnf.register
    def _(self, ce: OWLObjectMinCardinality, negated: bool = False):
        card = ce.get_cardinality()
        if negated:
            card = max(0, card - 1)
        filler = self.get_class_nnf(ce.get_filler(), negated=False)
        if negated:
            return OWLObjectMaxCardinality(card, ce.get_property(), filler)
        return OWLObjectMinCardinality(card, ce.get_property(), filler)
    
    @get_class_nnf.register
    def _(self, ce: OWLObjectExactCardinality, negated: bool = False):
        return self.get_class_nnf(ce.as_intersection_of_min_max(), negated)

    @get_class_nnf.register
    def _(self, ce: OWLObjectMaxCardinality, negated: bool = False):
        card = ce.get_cardinality()
        if negated:
            card = card + 1
        filler = self.get_class_nnf(ce.get_filler(), negated=False)
        if negated:
            return OWLObjectMinCardinality(card, ce.get_property(), filler)
        return OWLObjectMaxCardinality(card, ce.get_property(), filler)
    
    @get_class_nnf.register
    def _(self, ce: OWLObjectHasSelf, negated: bool = False):
        if negated:
            return ce.get_object_complement_of()
        return ce

    @get_class_nnf.register
    def _(self, ce: OWLObjectOneOf, negated: bool = False):
        union = ce.as_object_union_of()
        if isinstance(union, OWLObjectOneOf):
            if negated:
                return ce.get_object_complement_of()
            return ce
        return self.get_class_nnf(union, negated)

    @get_class_nnf.register
    def _(self, ce: OWLDataSomeValuesFrom, negated: bool = False):
        ...

    @get_class_nnf.register
    def _(self, ce: OWLDataAllValuesFrom, negated: bool = False):
        ...

    @get_class_nnf.register
    def _(self, ce: OWLDataHasValue, negated: bool = False):
        ...

    @get_class_nnf.register
    def _(self, ce: OWLDataExactCardinality, negated: bool = False):
        ...

    @get_class_nnf.register
    def _(self, ce: OWLDataMinCardinality, negated: bool = False):
        ...

    @get_class_nnf.register
    def _(self, ce: OWLDataMaxCardinality, negated: bool = False):
        ...

# OWL-APy custom util start

class IRIFixedSet:
    """A set of IRIs

    Call it with a list of IRIs to encode them to a number
    Call it with a number to get a list of IRIs back"""
    __slots__ = '_iri_idx', '_idx_iri'

    _iri_idx: Dict[IRI, int]
    _idx_iri: List[IRI]  # it works as Dict[int, IRI]

    def __init__(self, iri_set: Iterable[IRI]):
        """Create a new fixed set of IRIs

        Args:
            iri_set: IRIs in the set
        """
        fs = frozenset(iri_set)
        self._idx_iri = list(fs)
        self._iri_idx = dict(map(reversed, enumerate(self._idx_iri)))

    @overload
    def __call__(self, arg: int, /) -> Iterable[IRI]:
        ...

    @overload
    def __call__(self, arg: IRI, *, ignore_missing=False) -> int:
        ...

    @overload
    def __call__(self, arg: Iterable[IRI], *, ignore_missing=False) -> int:
        ...

    def __call__(self, arg, *, ignore_missing=False):
        """Encode or decode an IRI

        Args:
            arg: IRI or iterable of IRIs to encode
            ignore_missing: if missing(unrepresentable) objects should be silently ignored
                always True on decoding

        Returns:
            encoded or decoded representation of IRI(s)
        """
        if isinstance(arg, int):
            return self._decode(arg)
        elif isinstance(arg, IRI):
            return self._encode(arg, ignore_missing=ignore_missing)
        else:
            assert isinstance(arg, Iterable)
            r: int = 0
            for i in arg:
                r |= self._encode(i, ignore_missing=ignore_missing)
            return r

    def _decode(self, v: int) -> Iterable[IRI]:
        i: int = 0
        for i in iter_bits(v):
            yield self._idx_iri[i.bit_length() - 1]

    def _encode(self, i: IRI, ignore_missing: bool) -> int:
        if i in self._iri_idx:
            return 1 << self._iri_idx[i]
        elif ignore_missing:
            return 0
        else:
            raise KeyError(i)

    def items(self) -> Iterable[Tuple[int, IRI]]:
        """Return key-value pairs of bit => IRI"""
        for idx, i in enumerate(self._idx_iri):
            yield 1 << idx, i

    def __len__(self) -> int:
        return len(self._idx_iri)

    def __contains__(self, item: IRI) -> bool:
        return item in self._idx_iri


class NamedFixedSet(Generic[_HasIRI]):
    """Fixed set of objects that implement HasIRI
    """
    __slots__ = '_iri_set', '_Type'

    _iri_set: IRIFixedSet
    _Type: Type[_HasIRI]

    def __init__(self, factory: Type[_HasIRI], member_set: Iterable[_HasIRI]):
        """Create fixed set of same-class objects

        Args:
            factory: Type class to reconstruct an object
            member_set: members of the fixed set
        """
        self._Type = factory
        self._iri_set = IRIFixedSet(map(self._Type.get_iri, member_set))

    @overload
    def __call__(self, arg: Iterable[_HasIRI], *, ignore_missing=False) -> int:
        ...

    @overload
    def __call__(self, arg: _HasIRI, *, ignore_missing=False) -> int:
        ...

    @overload
    def __call__(self, arg: int, /) -> Iterable[_HasIRI]:
        ...

    def __call__(self, arg, *, ignore_missing=False):
        """Encode or decode an object

        Args:
            arg: object or iterable of objects to encode
            ignore_missing: if missing(unrepresentable) objects should be silently ignored
                always True on decoding

        Returns:
            encoded or decoded representation of object(s)
        """
        if isinstance(arg, int):
            return map(self._Type, self._iri_set(arg))
        else:
            try:
                if isinstance(arg, self._Type):
                    return self._iri_set(arg.get_iri(), ignore_missing=ignore_missing)
                else:
                    return self._iri_set(map(self._Type.get_iri, arg), ignore_missing=ignore_missing)
            except KeyError as ke:
                raise NameError(f"{self._Type(*ke.args)} not found in {type(self).__name__}") from ke

    def items(self) -> Iterable[Tuple[int, _HasIRI]]:
        """Return key-value pairs of bit => _HasIRI"""
        t = self._Type
        for e, i in self._iri_set.items():
            yield e, t(i)

    def __len__(self) -> int:
        return len(self._iri_set)

    def __contains__(self, item: _HasIRI) -> bool:
        return isinstance(item, self._Type) and item.get_iri() in self._iri_set


def popcount(v: int) -> int:
    """Count the active bits in a number"""
    return bin(v).count("1")


def iter_bits(v: int) -> Iterable[int]:
    """Iterate over individual bits in a number

    Args:
          v: input number

    Returns:
        numbers with only one bit set
    """
    while v:
        b = v & (~v + 1)
        yield b
        v ^= b


def iter_count(i: Iterable) -> int:
    """Count the number of elements in an iterable"""
    return sum(1 for _ in i)


def as_index(o: OWLObject) -> HasIndex:
    i = cast(HasIndex, o)
    assert type(i).type_index
    return i
