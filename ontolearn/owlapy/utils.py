from functools import singledispatchmethod
from typing import Iterable, overload, TypeVar, Generic, Type, Tuple, Dict, List

from ontolearn.owlapy import IRI, HasIRI

_HasIRI = TypeVar('_HasIRI', bound=HasIRI)


class IRIFixedSet:
    """A set of IRIs

    Call it with a list of IRIs to encode them to a number
    Call it with a number to get a list of IRIs back"""
    __slots__ = '_iri_idx', '_idx_iri'

    _iri_idx: Dict[IRI, int]
    _idx_iri: List[IRI] # it works as Dict[int, IRI]

    def __init__(self, iri_set: Iterable[IRI]):
        fs = frozenset(iri_set)
        self._idx_iri = list(fs)
        self._iri_idx = dict(map(reversed, enumerate(self._idx_iri)))

    @overload
    def __call__(self, arg: int, /) -> Iterable[IRI]: ...

    @overload
    def __call__(self, arg: IRI, *, ignore_missing=False) -> int: ...

    @overload
    def __call__(self, arg: Iterable[IRI], *, ignore_missing=False) -> int: ...

    def __call__(self, arg, *, ignore_missing=False):
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
        while v > 0:
            if v & 1:
                yield self._idx_iri[i]
            v >>= 1
            i += 1

    def _encode(self, i: IRI, ignore_missing: bool) -> int:
        if i in self._iri_idx:
            return 2 ** self._iri_idx[i]
        elif ignore_missing:
            return 0
        else:
            raise KeyError(i)

    def items(self) -> Iterable[Tuple[int, IRI]]:
        for idx, i in enumerate(self._idx_iri):
            yield 2 ** idx, i


class NamedFixedSet(Generic[_HasIRI]):
    __slots__ = '_iri_set', '_Type'

    _iri_set: IRIFixedSet
    _Type: Type[_HasIRI]

    def __init__(self, factory: Type[_HasIRI], member_set: Iterable[_HasIRI]):
        self._Type = factory
        self._iri_set = IRIFixedSet(map(self._Type.get_iri, member_set))

    @overload
    def __call__(self, arg: Iterable[_HasIRI], *, ignore_missing=False) -> int: ...

    @overload
    def __call__(self, arg: _HasIRI, *, ignore_missing=False) -> int: ...

    @overload
    def __call__(self, arg: int, /) -> Iterable[_HasIRI]: ...

    def __call__(self, arg, *, ignore_missing=False):
        if isinstance(arg, int):
            return map(self._Type, self._iri_set(arg))
        elif isinstance(arg, self._Type):
            return self._iri_set(arg.get_iri(), ignore_missing=ignore_missing)
        else:
            return self._iri_set(map(self._Type.get_iri, arg), ignore_missing=ignore_missing)

    def items(self) -> Iterable[Tuple[int, _HasIRI]]:
        t = self._Type
        for e, i in self._iri_set.items():
            yield e, t(i)


def popcount(v: int):
    """Count the active bits in a number"""
    return bin(v).count("1")
