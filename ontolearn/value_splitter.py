from abc import ABCMeta, abstractmethod
from typing import Dict, List, Set

from owlapy.model import TIME_DATATYPES, OWLDataProperty, OWLLiteral, OWLReasoner

import math


class AbstractValueSplitter(metaclass=ABCMeta):
    """Abstract base class for split calculation of data properties. """

    __slots__ = 'max_nr_splits'

    max_nr_splits: int

    @abstractmethod
    def __init__(self, max_nr_splits: int):
        self.max_nr_splits = max_nr_splits

    @abstractmethod
    def compute_splits_properties(self, reasoner: OWLReasoner, properties: List[OWLDataProperty]) \
            -> Dict[OWLDataProperty, List[OWLLiteral]]:
        pass

    def _combine_values(self, a: OWLLiteral, b: OWLLiteral) -> OWLLiteral:
        assert a.get_datatype() == b.get_datatype()

        if a.is_integer():
            return OWLLiteral((a.parse_integer() + b.parse_integer()) // 2)
        elif a.is_double():
            return OWLLiteral(round((a.parse_double() + b.parse_double()) / 2, 3))
        elif a.get_datatype() in TIME_DATATYPES:
            return a
        else:
            raise ValueError(a)


class BinningValueSplitter(AbstractValueSplitter):
    """Calculate a number of bins of equal size as splits."""

    __slots__ = ()

    def __init__(self, max_nr_splits: int = 12):
        super().__init__(max_nr_splits)

    def compute_splits_properties(self, reasoner: OWLReasoner, properties: List[OWLDataProperty]) \
            -> Dict[OWLDataProperty, List[OWLLiteral]]:
        return {p: self._compute_splits(set(reasoner.all_data_property_values(p))) for p in properties}

    def _compute_splits(self, dp_values: Set[OWLLiteral]) -> List[OWLLiteral]:
        values = sorted(list(dp_values))
        nr_splits = min(self.max_nr_splits, len(values) + 1)

        splits = set()
        if len(values) > 0:
            splits.add(values[0])
        for i in range(1, nr_splits):
            index = max(math.floor(i * len(values) / nr_splits),
                        math.floor(i * len(values) / (nr_splits - 1) - 1))
            splits.add(self._combine_values(values[index], values[min(index + 1, len(values)-1)]))

        return sorted(list(splits))
