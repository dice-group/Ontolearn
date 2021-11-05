from abc import ABCMeta, abstractmethod
from typing import Dict, List, Set, Union

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.model import NUMERIC_DATATYPES, OWLDataProperty, OWLLiteral

import math

Split_Types = Union[float, int]


class AbstractValueSplitter(metaclass=ABCMeta):
    """Abstract base class for split calculation of data properties. """

    __slots__ = 'max_nr_splits'

    max_nr_splits: int

    @abstractmethod
    def __init__(self, max_nr_splits: int):
        self.max_nr_splits = max_nr_splits

    @abstractmethod
    def compute_splits_properties(self, kb: KnowledgeBase, properties: List[OWLDataProperty] = None) \
            -> Dict[OWLDataProperty, List[OWLLiteral]]:
        pass

    def _get_all_properties(self, kb: KnowledgeBase) -> List[OWLDataProperty]:
        properties = []
        for p in kb.ontology().data_properties_in_signature():
            ranges = set(kb.reasoner().data_property_ranges(p))
            if NUMERIC_DATATYPES & ranges:
                properties.append(p)
        return properties


class BinningValueSplitter(AbstractValueSplitter):
    """Calculate a number of bins as splits.

    """
    __slots__ = ()

    def __init__(self, max_nr_splits: int = 10):
        super().__init__(max_nr_splits)

    def compute_splits_properties(self, kb: KnowledgeBase, properties: List[OWLDataProperty] = None) \
            -> Dict[OWLDataProperty, List[OWLLiteral]]:
        if properties is None:
            properties = self._get_all_properties(kb)
        dp_splits = dict()
        for p in properties:
            values = list({lit.to_python() for lit in kb.reasoner().all_data_property_values(p)})
            dp_splits[p] = self._compute_splits_values(values)
        return dp_splits

    def _compute_splits_values(self, values: List[Split_Types]) -> List[OWLLiteral]:
        values = sorted(list(values))
        nr_splits = min(self.max_nr_splits, len(values))

        splits = set()
        if len(values) > 0:
            splits.add(values[0])
        if len(values) > 1:
            splits.add(values[len(values)-1])

        for i in range(1, nr_splits):
            index = max(math.floor(i * len(values) / nr_splits),
                        math.floor(i * len(values) / (nr_splits - 1) - 1))
            splits.add(self._combine_values(values[index], values[min(index + 1, len(values)-1)]))

        return sorted(list(map(OWLLiteral, splits)))

    def _combine_values(self, a: Split_Types, b: Split_Types) -> Split_Types:
        assert type(a) == type(b)

        if isinstance(a, int):
            return (a + b) // 2
        elif isinstance(a, float):
            return round((a + b) / 2, 3)
        else:
            raise ValueError(a)
