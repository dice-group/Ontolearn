import unittest
from datetime import datetime

from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLDataProperty

from ontolearn.value_splitter import (BinningValueSplitter, EntropyValueSplitter, IndividualValues, Split)

NS = "http://example.com/splitter#"


def ind(name):
    return OWLNamedIndividual(IRI.create(NS + name))


def dp(name):
    return OWLDataProperty(IRI.create(NS + name))


class FakeReasoner:
    """Minimal stand-in for the reasoner API used by the value splitters."""

    def __init__(self, values):
        # {(individual_remainder, property_remainder): [python values]}
        self.values = values

    def data_property_values(self, individual, prop, direct=True):
        key = (individual.iri.get_remainder(), prop.iri.get_remainder())
        return iter([OWLLiteral(v) for v in self.values.get(key, [])])

    def all_data_property_values(self, prop, direct=True):
        return iter([OWLLiteral(v) for (_, p), vs in self.values.items() if p == prop.iri.get_remainder()
                     for v in vs])


class TestCombineValues(unittest.TestCase):

    def setUp(self):
        self.splitter = BinningValueSplitter()

    def test_int_floor_mean(self):
        self.assertEqual(self.splitter._combine_values(20, 30), 25)
        self.assertEqual(self.splitter._combine_values(1, 2), 1)

    def test_float_rounded_mean(self):
        self.assertEqual(self.splitter._combine_values(1.0, 2.0), 1.5)
        self.assertEqual(self.splitter._combine_values(0.1111, 0.2222), 0.167)

    def test_other_types_return_first(self):
        d1, d2 = datetime(2000, 1, 1), datetime(2001, 1, 1)
        self.assertEqual(self.splitter._combine_values(d1, d2), d1)
        self.assertEqual(self.splitter._combine_values(1, 2.0), 1)

    def test_reset_is_noop_for_binning(self):
        self.assertIsNone(self.splitter.reset())


class TestBinningEdgeCases(unittest.TestCase):

    def test_no_values_gives_no_splits(self):
        splitter = BinningValueSplitter(max_nr_splits=4)
        self.assertEqual(splitter.compute_splits_properties(FakeReasoner({}), [dp("p")]), {dp("p"): []})

    def test_single_value(self):
        splitter = BinningValueSplitter(max_nr_splits=4)
        reasoner = FakeReasoner({("a", "p"): [7]})
        self.assertEqual(splitter.compute_splits_properties(reasoner, [dp("p")]), {dp("p"): [OWLLiteral(7)]})


class TestSplitAndIndividualValues(unittest.TestCase):

    def test_split_ordering_uses_entropy(self):
        low, high = Split(["a"], ["b"], 0.1, set()), Split(["c"], ["d"], 0.9, set())
        self.assertTrue(low < high)
        self.assertTrue(high > low)
        self.assertEqual(sorted([high, low], reverse=True), [high, low])
        self.assertEqual(Split([], [], 0.5, set()), Split(["x"], ["y"], 0.5, {"p"}))
        self.assertNotEqual(low, high)
        self.assertEqual(low.__eq__("not a split"), NotImplemented)
        self.assertEqual(low.__lt__("not a split"), NotImplemented)

    def test_individual_values_overlap(self):
        values = IndividualValues({"a": 1, "b": 2}, {"c": 3, "d": 4})
        self.assertEqual(sorted(values.get_pos_values()), [1, 2])
        self.assertEqual(sorted(values.get_neg_values()), [3, 4])
        sub = values.get_overlapping_with_split(Split(["a"], ["d"], 0.0, set()))
        self.assertEqual(sub.pos_map, {"a": 1})
        self.assertEqual(sub.neg_map, {"d": 4})


class TestEntropyValueSplitter(unittest.TestCase):

    def test_requires_pos_and_neg(self):
        splitter = EntropyValueSplitter()
        with self.assertRaises(AssertionError):
            splitter.compute_splits_properties(FakeReasoner({}), [dp("p")])
        with self.assertRaises(AssertionError):
            splitter.compute_splits_properties(FakeReasoner({}), [dp("p")], pos={ind("a")})

    def test_separable_property_gets_split_between_classes(self):
        pos = {ind("p1"), ind("p2")}
        neg = {ind("n1"), ind("n2")}
        reasoner = FakeReasoner({("p1", "age"): [10], ("p2", "age"): [20],
                                 ("n1", "age"): [30], ("n2", "age"): [40]})
        splitter = EntropyValueSplitter(max_nr_splits=2)
        splits = splitter.compute_splits_properties(reasoner, [dp("age")], pos=pos, neg=neg)
        self.assertEqual(splits, {dp("age"): [OWLLiteral(25)]})

    def test_float_values(self):
        pos, neg = {ind("p1")}, {ind("n1")}
        reasoner = FakeReasoner({("p1", "w"): [1.0], ("n1", "w"): [2.0]})
        splits = EntropyValueSplitter().compute_splits_properties(reasoner, [dp("w")], pos=pos, neg=neg)
        self.assertEqual(splits, {dp("w"): [OWLLiteral(1.5)]})

    def test_interleaved_values_respect_max_nr_splits(self):
        pos = {ind("p1"), ind("p2")}
        neg = {ind("n1"), ind("n2")}
        reasoner = FakeReasoner({("p1", "v"): [10], ("n1", "v"): [20], ("p2", "v"): [30], ("n2", "v"): [40]})
        splits = EntropyValueSplitter(max_nr_splits=2).compute_splits_properties(
            reasoner, [dp("v")], pos=pos, neg=neg)
        found = [lit.parse_integer() for lit in splits[dp("v")]]
        self.assertGreaterEqual(len(found), 1)
        self.assertLessEqual(len(found), 2)
        self.assertTrue(all(10 <= v <= 40 for v in found))

    def test_individuals_without_values_are_ignored(self):
        pos, neg = {ind("p1"), ind("p_missing")}, {ind("n1")}
        reasoner = FakeReasoner({("p1", "age"): [10], ("n1", "age"): [30]})
        splits = EntropyValueSplitter().compute_splits_properties(reasoner, [dp("age")], pos=pos, neg=neg)
        self.assertEqual(splits, {dp("age"): [OWLLiteral(20)]})

    def test_property_with_no_values_has_no_splits(self):
        pos, neg = {ind("p1")}, {ind("n1")}
        splits = EntropyValueSplitter().compute_splits_properties(FakeReasoner({}), [dp("age")], pos=pos, neg=neg)
        self.assertEqual(splits, {dp("age"): []})

    def test_multiple_properties_and_reset(self):
        pos, neg = {ind("p1"), ind("p2")}, {ind("n1"), ind("n2")}
        reasoner = FakeReasoner({("p1", "a"): [1], ("p2", "a"): [2], ("n1", "a"): [8], ("n2", "a"): [9],
                                 ("p1", "b"): [100], ("p2", "b"): [90], ("n1", "b"): [10], ("n2", "b"): [20]})
        splitter = EntropyValueSplitter(max_nr_splits=1)
        splits = splitter.compute_splits_properties(reasoner, [dp("a"), dp("b")], pos=pos, neg=neg)
        self.assertEqual(set(splits), {dp("a"), dp("b")})
        self.assertEqual(len(splits[dp("a")]), 1)
        self.assertEqual(len(splits[dp("b")]), 1)
        self.assertTrue(3 <= splits[dp("a")][0].parse_integer() <= 7)
        self.assertTrue(20 <= splits[dp("b")][0].parse_integer() <= 90)
        splitter.reset()
        self.assertEqual(splitter._prop_to_values, {})

    def test_get_inds_below_above(self):
        from sortedcontainers import SortedDict
        inv = SortedDict({1: ["a"], 5: ["b", "c"], 9: ["d"]})
        below, above = EntropyValueSplitter()._get_inds_below_above(5, inv)
        self.assertEqual(sorted(below), ["a", "b", "c"])
        self.assertEqual(above, ["d"])
        below, above = EntropyValueSplitter()._get_inds_below_above(6, inv)
        self.assertEqual(sorted(below), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
