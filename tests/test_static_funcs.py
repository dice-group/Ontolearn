import os
import tempfile

import pytest
from owlapy.class_expression import (OWLClass, OWLObjectIntersectionOf, OWLObjectMaxCardinality,
                                     OWLObjectMinCardinality, OWLObjectSomeValuesFrom)
from owlapy.iri import IRI
from owlapy.owl_property import OWLObjectProperty

from ontolearn.utils.static_funcs import (assert_class_expression_type, compute_f1_score,
                                          compute_f1_score_from_confusion_matrix, concept_len, concept_reducer,
                                          concept_reducer_properties, f1_set_similarity, get_file_base_name,
                                          init_length_metric, init_named_individuals, make_iterable_verbose,
                                          prepare_output_path)


class TestStaticFuncs:

    def test_f1(self):
        assert compute_f1_score(individuals={"A", "B", "C"}, pos={"A", "B", "C"}, neg={"D"}) == 1.0
        assert compute_f1_score(individuals={"A", "B", "C"}, pos={"A", "B", "C"}, neg={"D"}) == 1.0


NS = "http://example.com/static#"
A, B = OWLClass(IRI.create(NS + "A")), OWLClass(IRI.create(NS + "B"))
R = OWLObjectProperty(IRI.create(NS + "r"))


class TestStaticFuncsMore:

    def test_f1_set_similarity(self):
        assert f1_set_similarity(set(), set()) == 1.0
        assert f1_set_similarity({"a"}, set()) == 0.0
        assert f1_set_similarity(set(), {"a"}) == 0.0
        assert f1_set_similarity({"a"}, {"b"}) == 0.0
        assert f1_set_similarity({"a", "b"}, {"a", "b"}) == 1.0
        # tp=1, fp=1, fn=1 -> 0.5
        assert f1_set_similarity({"a", "b"}, {"a", "c"}) == 0.5

    def test_compute_f1_score_edge_cases(self):
        assert compute_f1_score({"A"}, {"A", "B"}, {"C"}) == (2 * 1 * 0.5) / 1.5
        # no positives retrieved
        assert compute_f1_score({"C"}, {"A"}, {"C"}) == 0.0
        # nothing retrieved at all
        assert compute_f1_score(set(), {"A"}, {"C"}) == 0.0
        # no positives in the learning problem
        assert compute_f1_score({"C"}, set(), {"C"}) == 0.0

    def test_compute_f1_score_type_mismatch(self):
        with pytest.raises(AssertionError):
            compute_f1_score({"A"}, frozenset({"A"}), {"B"})

    def test_f1_from_confusion_matrix(self):
        assert compute_f1_score_from_confusion_matrix({"tp": 2, "fn": 0, "fp": 0, "tn": 5}) == 1.0
        assert compute_f1_score_from_confusion_matrix({"tp": "1", "fn": "1", "fp": "1", "tn": "0"}) == 0.5
        assert compute_f1_score_from_confusion_matrix({"tp": 0, "fn": 0, "fp": 3, "tn": 1}) == 0.0
        assert compute_f1_score_from_confusion_matrix({"tp": 0, "fn": 3, "fp": 0, "tn": 1}) == 0.0
        assert compute_f1_score_from_confusion_matrix({"tp": 0, "fn": 3, "fp": 2, "tn": 1}) == 0.0

    def test_concept_reducer(self):
        assert concept_reducer({1, 2, 3}, lambda x: x[0] + x[1]) == {2, 3, 4, 5, 6}

    def test_concept_reducer_properties(self):
        res = concept_reducer_properties({A, B}, [R], cls=OWLObjectSomeValuesFrom)
        assert res == {OWLObjectSomeValuesFrom(R, A), OWLObjectSomeValuesFrom(R, B)}

        res = concept_reducer_properties({A}, [R], cls=OWLObjectMinCardinality, cardinality=3)
        assert res == {OWLObjectMinCardinality(cardinality=3, property=R, filler=A)}

        res = concept_reducer_properties({A}, [R], cls=OWLObjectMaxCardinality, cardinality=1)
        assert res == {OWLObjectMaxCardinality(cardinality=1, property=R, filler=A)}

    def test_concept_reducer_properties_validates_input(self):
        with pytest.raises(AssertionError):
            concept_reducer_properties({A}, [R], cls=None)
        with pytest.raises(AssertionError):
            concept_reducer_properties({A}, [R], cls=OWLObjectSomeValuesFrom, cardinality=0)

    def test_make_iterable_verbose(self):
        items = [1, 2, 3]
        assert make_iterable_verbose(items, verbose=0) is items
        assert list(make_iterable_verbose(items, verbose=1)) == items

    def test_init_length_metric_and_concept_len(self):
        explicit = init_length_metric()
        assert explicit is not None
        assert init_length_metric(explicit) is explicit
        sentinel = object()
        assert init_length_metric(length_metric_factory=lambda: sentinel) is sentinel
        assert concept_len(A) == 1
        assert concept_len(OWLObjectIntersectionOf([A, B])) == 3
        assert concept_len(OWLObjectSomeValuesFrom(R, A)) == 3

    def test_init_named_individuals(self):
        use, cache = init_named_individuals(0)
        assert use is False and cache is None
        use, cache = init_named_individuals(10)
        assert use is True and cache.maxsize == 10

    def test_file_name_helpers(self):
        assert get_file_base_name("/path/to/file.txt") == "file"
        assert get_file_base_name("a/b/archive.tar.gz") == "archive.tar"
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "dir")
            path = prepare_output_path("summary", output_dir=out, extension="json")
            assert path == os.path.join(out, "summary.json")
            assert os.path.isdir(out)

    def test_assert_class_expression_type(self):
        assert assert_class_expression_type(A)
        assert assert_class_expression_type(OWLObjectSomeValuesFrom(R, A))
        assert not assert_class_expression_type("A")
        assert not assert_class_expression_type(None)
        assert not assert_class_expression_type(R)
