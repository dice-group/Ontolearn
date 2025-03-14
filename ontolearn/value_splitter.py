"""Value splitters."""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from functools import total_ordering
from itertools import chain

from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import OWLReasoner
from pandas import Timedelta
from scipy.stats import entropy
from sortedcontainers import SortedDict
from typing import Dict, List, Optional, Set, Tuple, Union

import math


Values = Union[OWLLiteral, int, float, bool, Timedelta, datetime, date]  #:


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

    def _combine_values(self, a: Values, b: Values) -> Values:
        if isinstance(a, int) and isinstance(b, int):
            return (a+b) // 2
        elif isinstance(a, float) and isinstance(b, float):
            return round((a+b)/2, 3)
        else:
            return a

    def reset(self):
        pass


class BinningValueSplitter(AbstractValueSplitter):
    """Calculate a number of bins of equal size as splits."""

    __slots__ = ()

    def __init__(self, max_nr_splits: int = 12):
        super().__init__(max_nr_splits)

    def compute_splits_properties(self, reasoner: OWLReasoner, properties: List[OWLDataProperty]) \
            -> Dict[OWLDataProperty, List[OWLLiteral]]:
        return {p: self._compute_splits(set(reasoner.all_data_property_values(p))) for p in properties}

    def _compute_splits(self, dp_values: Set[OWLLiteral]) -> List[OWLLiteral]:
        values = sorted([val.to_python() for val in dp_values])
        nr_splits = min(self.max_nr_splits, len(values) + 1)

        splits = set()
        if len(values) > 0:
            splits.add(values[0])
        for i in range(1, nr_splits):
            index = max(math.floor(i * len(values) / nr_splits),
                        math.floor(i * len(values) / (nr_splits - 1) - 1))
            splits.add(self._combine_values(values[index], values[min(index + 1, len(values)-1)]))
        return sorted(list(map(OWLLiteral, splits)))


@total_ordering
@dataclass
class Split:
    pos: List[str]
    neg: List[str]
    entropy: float
    used_properties: Set[str]

    def __eq__(self, other):
        if type(self) == type(other):
            return math.isclose(self.entropy, other.entropy)
        return NotImplemented

    def __lt__(self, other):
        if type(self) == type(other):
            return self.entropy < other.entropy
        return NotImplemented


@dataclass
class IndividualValues:
    pos_map: Dict[str, Values]
    neg_map: Dict[str, Values]

    def get_pos_values(self) -> List[Values]:
        return list(self.pos_map.values())

    def get_neg_values(self) -> List[Values]:
        return list(self.neg_map.values())

    def get_overlapping_with_split(self, split: Split) -> 'IndividualValues':
        return IndividualValues({ind: v for ind, v in self.pos_map.items() if ind in split.pos},
                                {ind: v for ind, v in self.neg_map.items() if ind in split.neg})


class EntropyValueSplitter(AbstractValueSplitter):
    """Calculate the splits depending on the entropy of the resulting sets."""

    __slots__ = '_prop_to_values'

    _prop_to_values: Dict[OWLDataProperty, IndividualValues]

    def __init__(self, max_nr_splits: int = 2):
        super().__init__(max_nr_splits)
        self._prop_to_values = {}

    def compute_splits_properties(self, reasoner: OWLReasoner, properties: List[OWLDataProperty],
                                  pos: Set[OWLNamedIndividual] = None, neg: Set[OWLNamedIndividual] = None) \
            -> Dict[OWLDataProperty, List[OWLLiteral]]:
        assert pos is not None
        assert neg is not None

        self.reset()
        properties = properties.copy()

        dp_splits: Dict[OWLDataProperty, List[OWLLiteral]] = {}
        for property_ in properties:
            dp_splits[property_] = []
            self._prop_to_values[property_] = IndividualValues(self._get_values_for_inds(reasoner, property_, pos),
                                                               self._get_values_for_inds(reasoner, property_, neg))

        pos_str = [p.iri.get_remainder() for p in pos]
        neg_str = [n.iri.get_remainder() for n in neg]
        current_splits = [Split(pos_str, neg_str, 0, set())]
        while len(properties) > 0 and len(current_splits) > 0:
            next_level_splits = []
            for property_ in properties[:]:
                for split in current_splits:
                    if property_.iri.get_remainder() not in split.used_properties:
                        value, new_splits = self._compute_split_value(property_, split)

                        if value is not None:
                            value = OWLLiteral(value)
                            if value not in dp_splits[property_]:
                                dp_splits[property_].append(value)
                            next_level_splits.extend(new_splits)

                        if len(dp_splits[property_]) >= self.max_nr_splits:
                            properties.remove(property_)
                            break
            current_splits = sorted(next_level_splits, reverse=True)

        return dp_splits

    def _compute_split_value(self, property_: OWLDataProperty, split: Split) -> Tuple[Optional[Values], List[Split]]:
        current_values = self._prop_to_values[property_].get_overlapping_with_split(split)
        number_of_values = len(current_values.pos_map) + len(current_values.neg_map)

        if number_of_values == 0:
            return None, []
        current_entropy = entropy((len(current_values.pos_map) / number_of_values,
                                  len(current_values.neg_map) / number_of_values))

        best_gain = 0
        best_value = None
        best_splits = None
        pos_inv: 'SortedDict[Values, List[str]]' = SortedDict()
        for k, v in current_values.pos_map.items():
            pos_inv[v] = pos_inv.get(v, []) + [k]
        neg_inv: 'SortedDict[Values, List[str]]' = SortedDict()
        for k, v in current_values.neg_map.items():
            neg_inv[v] = neg_inv.get(v, []) + [k]

        values = sorted(list(pos_inv.keys()) + list(neg_inv.keys()))
        values = [self._combine_values(x, y) for x, y in zip(values, values[1:])]

        for value in values:
            pos_below, pos_above = self._get_inds_below_above(value, pos_inv)
            neg_below, neg_above = self._get_inds_below_above(value, neg_inv)
            num_below = len(pos_below) + len(neg_below)
            num_above = len(pos_above) + len(neg_above)

            entropy_below = 0
            if num_below > 0:
                entropy_below = entropy((len(pos_below) / num_below, len(neg_below) / num_below))
            entropy_above = 0
            if num_above > 0:
                entropy_above = entropy((len(pos_above) / num_above, len(neg_above) / num_above))
            cond_entropy = ((num_below / number_of_values) * entropy_below +
                            (num_above / number_of_values) * entropy_above)
            gain = current_entropy - cond_entropy

            if gain >= best_gain:
                best_gain = gain
                best_value = value
                best_splits = []
                if entropy_below > 0:
                    best_splits.append(self._make_split(pos_below, neg_below, entropy_below, split, property_))
                if entropy_above > 0:
                    best_splits.append(self._make_split(pos_above, neg_above, entropy_above, split, property_))

        return best_value, best_splits

    def _make_split(self, pos: List[str], neg: List[str], entropy: float,
                    split: Split, property_: OWLDataProperty) -> Split:
        used_properties = deepcopy(split.used_properties)
        used_properties.add(property_.iri.get_remainder())
        return Split(pos, neg, entropy, used_properties)

    def _get_inds_below_above(self, value: Values, ind_value_map: 'SortedDict[Values, List[str]]') \
            -> Tuple[List[str], List[str]]:
        idx = ind_value_map.bisect(value)
        inds_below = list(chain.from_iterable(ind_value_map.values()[:idx]))
        inds_above = list(chain.from_iterable(ind_value_map.values()[idx:]))
        return inds_below, inds_above

    def _get_values_for_inds(self, reasoner: OWLReasoner, property_: OWLDataProperty, inds: Set[OWLNamedIndividual]) \
            -> Dict[str, Values]:
        inds_to_value = dict()
        for ind in inds:
            try:
                val = next(iter(reasoner.data_property_values(ind, property_)))
                inds_to_value[ind.iri.get_remainder()] = val.to_python()
            except StopIteration:
                pass
        return inds_to_value

    def reset(self):
        self._prop_to_values = {}
