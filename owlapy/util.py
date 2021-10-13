from functools import singledispatchmethod, total_ordering
from typing import Iterable, overload, TypeVar, Generic, Type, Tuple, Dict, List, cast, Optional

from owlapy.model import OWLObject, HasIndex, HasIRI, OWLClassExpression, OWLClass, OWLObjectIntersectionOf, \
    OWLObjectUnionOf, OWLObjectComplementOf, OWLNothing, OWLThing, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectHasValue, OWLObjectMinCardinality, OWLObjectMaxCardinality, OWLObjectExactCardinality, OWLObjectHasSelf, \
    OWLObjectOneOf, OWLDataMaxCardinality, OWLDataMinCardinality, OWLDataExactCardinality, OWLDataHasValue, \
    OWLDataAllValuesFrom, OWLDataSomeValuesFrom, OWLObjectRestriction, HasFiller, HasCardinality, HasOperands, IRI, \
    OWLObjectInverseOf, OWLDatatypeRestriction, OWLDataComplementOf, OWLDatatype, OWLDataUnionOf, \
    OWLDataIntersectionOf, OWLDataOneOf

_HasIRI = TypeVar('_HasIRI', bound=HasIRI)  #:
_HasIndex = TypeVar('_HasIndex', bound=HasIndex)  #:
_O = TypeVar('_O')  #:
_K = TypeVar('_K')
_V = TypeVar('_V')


@total_ordering
class OrderedOWLObject:
    """Holder of OWL Objects that can be used for Python sorted

    The Ordering is dependent on the type_index of the impl. classes recursively followed by all components of the
    OWL Object.
    """
    __slots__ = 'o', '_chain'

    o: _HasIndex  # o: Intersection[OWLObject, HasIndex]
    _chain: Optional[Tuple]

    # we are limited by https://github.com/python/typing/issues/213 # o: Intersection[OWLObject, HasIndex]
    def __init__(self, o: _HasIndex):
        """OWL Object holder with a defined sort order

        Args:
            o: OWL Object
        """
        self.o = o
        self._chain = None

    def _comparison_chain(self):
        if self._chain is None:
            c = [self.o.type_index]

            if isinstance(self.o, OWLObjectRestriction):
                c.append(OrderedOWLObject(as_index(self.o.get_property())))
            if isinstance(self.o, OWLObjectInverseOf):
                c.append(self.o.get_named_property().get_iri().as_str())
            if isinstance(self.o, HasFiller):
                c.append(OrderedOWLObject(self.o.get_filler()))
            if isinstance(self.o, HasCardinality):
                c.append(self.o.get_cardinality())
            if isinstance(self.o, HasOperands):
                c.append(tuple(map(OrderedOWLObject, self.o.operands())))
            if isinstance(self.o, HasIRI):
                c.append(self.o.get_iri().as_str())
            if len(c) == 1:
                raise NotImplementedError(type(self.o))

            self._chain = tuple(c)

        return self._chain

    def __lt__(self, other):
        if self.o.type_index < other.o.type_index:
            return True
        elif self.o.type_index > other.o.type_index:
            return False
        else:
            return self._comparison_chain() < other._comparison_chain()

    def __eq__(self, other):
        return self.o == other.o


def _sort_by_ordered_owl_object(i: Iterable[_O]) -> Iterable[_O]:
    return sorted(i, key=OrderedOWLObject)


class NNF:
    """This class contains functions to transform a Class Expression into Negation Normal Form"""
    @singledispatchmethod
    def get_class_nnf(self, ce: OWLClassExpression, negated: bool = False) -> OWLClassExpression:
        """Convert a Class Expression to Negation Normal Form. Operands will be sorted.

        Args:
            ce: Class Expression
            negated: whether the result should be negated

        Returns:
            Class Expression in Negation Normal Form
            """
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
        ops = map(lambda _: self.get_class_nnf(_, negated),
                  _sort_by_ordered_owl_object(ce.operands()))
        if negated:
            return OWLObjectUnionOf(ops)
        return OWLObjectIntersectionOf(ops)

    @get_class_nnf.register
    def _(self, ce: OWLObjectUnionOf, negated: bool = False):
        ops = map(lambda _: self.get_class_nnf(_, negated),
                  _sort_by_ordered_owl_object(ce.operands()))
        if negated:
            return OWLObjectIntersectionOf(ops)
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
        filler = self.get_class_nnf(ce.get_filler(), negated)
        if negated:
            return OWLDataAllValuesFrom(ce.get_property(), filler)
        return OWLDataSomeValuesFrom(ce.get_property(), filler)

    @get_class_nnf.register
    def _(self, ce: OWLDataAllValuesFrom, negated: bool = False):
        filler = self.get_class_nnf(ce.get_filler(), negated)
        if negated:
            return OWLDataSomeValuesFrom(ce.get_property(), filler)
        return OWLDataAllValuesFrom(ce.get_property(), filler)

    @get_class_nnf.register
    def _(self, ce: OWLDatatypeRestriction, negated: bool = False):
        if negated:
            return OWLDataComplementOf(ce)
        return ce

    @get_class_nnf.register
    def _(self, ce: OWLDatatype, negated: bool = False):
        if negated:
            return OWLDataComplementOf(ce)
        return ce

    @get_class_nnf.register
    def _(self, ce: OWLDataComplementOf, negated: bool = False):
        return self.get_class_nnf(ce.get_data_range(), not negated)

    @get_class_nnf.register
    def _(self, ce: OWLDataHasValue, negated: bool = False):
        return self.get_class_nnf(ce.as_some_values_from(), negated)

    @get_class_nnf.register
    def _(self, ce: OWLDataOneOf, negated: bool = False):
        if len(list(ce.values())) == 1:
            if negated:
                return OWLDataComplementOf(ce)
            return ce
        union = OWLDataUnionOf([OWLDataOneOf(v) for v in ce.values()])
        return self.get_class_nnf(union, negated)

    @get_class_nnf.register
    def _(self, ce: OWLDataIntersectionOf, negated: bool = False):
        ops = map(lambda _: self.get_class_nnf(_, negated),
                  _sort_by_ordered_owl_object(ce.operands()))
        if negated:
            return OWLDataUnionOf(ops)
        return OWLDataIntersectionOf(ops)

    @get_class_nnf.register
    def _(self, ce: OWLDataUnionOf, negated: bool = False):
        ops = map(lambda _: self.get_class_nnf(_, negated),
                  _sort_by_ordered_owl_object(ce.operands()))
        if negated:
            return OWLDataIntersectionOf(ops)
        return OWLDataUnionOf(ops)

    @get_class_nnf.register
    def _(self, ce: OWLDataExactCardinality, negated: bool = False):
        return self.get_class_nnf(ce.as_intersection_of_min_max(), negated)

    @get_class_nnf.register
    def _(self, ce: OWLDataMinCardinality, negated: bool = False):
        card = ce.get_cardinality()
        if negated:
            card = max(0, card - 1)
        filler = self.get_class_nnf(ce.get_filler(), negated=False)
        if negated:
            return OWLDataMaxCardinality(card, ce.get_property(), filler)
        return OWLDataMinCardinality(card, ce.get_property(), filler)

    @get_class_nnf.register
    def _(self, ce: OWLDataMaxCardinality, negated: bool = False):
        card = ce.get_cardinality()
        if negated:
            card = card + 1
        filler = self.get_class_nnf(ce.get_filler(), negated=False)
        if negated:
            return OWLDataMinCardinality(card, ce.get_property(), filler)
        return OWLDataMaxCardinality(card, ce.get_property(), filler)


# OWL-APy custom util start

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
        for i in iter_bit_pos(v):
            yield self._idx_iri[i]

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
        b = 1 << v.bit_length() - 1
        yield b
        v -= b


def iter_bit_pos(v: int) -> Iterable[int]:
    """Iterate over individual bit positions in a number

    Args:
          v: input number

    Returns:
        current bit index
    """
    while v:
        b = v.bit_length() - 1
        yield b
        v -= 1 << b


def iter_count(i: Iterable) -> int:
    """Count the number of elements in an iterable"""
    return sum(1 for _ in i)


def as_index(o: OWLObject) -> HasIndex:
    """Cast OWL Object to HasIndex"""
    i = cast(HasIndex, o)
    assert type(i).type_index
    return i


# adapted from functools.lru_cache
class LRUCache(Generic[_K, _V]):
    # Constants shared by all lru cache instances:
    sentinel = object()  # unique object used to signal cache misses
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

    def __init__(self, maxsize: Optional[int] = None):
        from _thread import RLock

        self.cache = {}
        self.hits = self.misses = 0
        self.full = False
        self.cache_get = self.cache.get  # bound method to lookup a key or return None
        self.cache_len = self.cache.__len__  # get cache size without calling len()
        self.lock = RLock()  # because linkedlist updates aren't threadsafe
        self.root = []  # root of the circular doubly linked list
        self.root[:] = [self.root, self.root, None, None]  # initialize by pointing to self
        self.maxsize = maxsize

    def __contains__(self, item: _K) -> bool:
        with self.lock:
            link = self.cache_get(item)
            if link is not None:
                self.hits += 1
                return True
            self.misses += 1
            return False

    def __getitem__(self, item: _K) -> _V:
        with self.lock:
            link = self.cache_get(item)
            if link is not None:
                # Move the link to the front of the circular queue
                link_prev, link_next, _key, result = link
                link_prev[LRUCache.NEXT] = link_next
                link_next[LRUCache.PREV] = link_prev
                last = self.root[LRUCache.PREV]
                last[LRUCache.NEXT] = self.root[LRUCache.PREV] = link
                link[LRUCache.PREV] = last
                link[LRUCache.NEXT] = self.root
                return result

    def __setitem__(self, key: _K, value: _V):
        with self.lock:
            if key in self.cache:
                # Getting here means that this same key was added to the
                # cache while the lock was released.  Since the link
                # update is already done, we need only return the
                # computed result and update the count of misses.
                pass
            elif self.full:
                # Use the old root to store the new key and result.
                oldroot = self.root
                oldroot[LRUCache.KEY] = key
                oldroot[LRUCache.RESULT] = value
                # Empty the oldest link and make it the new root.
                # Keep a reference to the old key and old result to
                # prevent their ref counts from going to zero during the
                # update. That will prevent potentially arbitrary object
                # clean-up code (i.e. __del__) from running while we're
                # still adjusting the links.
                self.root = oldroot[LRUCache.NEXT]
                oldkey = self.root[LRUCache.KEY]
                _oldresult = self.root[LRUCache.RESULT]
                self.root[LRUCache.KEY] = self.root[LRUCache.RESULT] = None
                # Now update the cache dictionary.
                del self.cache[oldkey]
                # Save the potentially reentrant cache[key] assignment
                # for last, after the root and links have been put in
                # a consistent state.
                self.cache[key] = oldroot
            else:
                # Put result in a new link at the front of the queue.
                last = self.root[LRUCache.PREV]
                link = [last, self.root, key, value]
                last[LRUCache.NEXT] = self.root[LRUCache.PREV] = self.cache[key] = link
                # Use the cache_len bound method instead of the len() function
                # which could potentially be wrapped in an lru_cache itself.
                if self.maxsize is not None:
                    self.full = (self.cache_len() >= self.maxsize)

    def cache_info(self):
        """Report cache statistics"""
        with self.lock:
            from collections import namedtuple
            return namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])(
                self.hits, self.misses, self.maxsize, self.cache_len())

    def cache_clear(self):
        """Clear the cache and cache statistics"""
        with self.lock:
            self.cache.clear()
            self.root[:] = [self.root, self.root, None, None]
            self.hits = self.misses = 0
            self.full = False
