from typing import Iterable

from owlapy.util import iter_bits, popcount


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
