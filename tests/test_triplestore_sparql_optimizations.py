"""Unit tests for the SPARQL COUNT/ASK based optimizations in `ontolearn.triple_store`
(see https://github.com/dice-group/Ontolearn/issues/613).

These tests mock the HTTP layer so they exercise the query-construction/result-parsing
logic without requiring a live triplestore, unlike `tests/test_triplestore.py`.
"""
import unittest
from unittest.mock import MagicMock, patch

from ontolearn.triple_store import TripleStore, TripleStoreOntology, TripleStoreReasoner
from owlapy.class_expression import OWLClass, OWLThing
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty

NS = "http://example.org/onto#"
URL = "http://localhost:3030/fake/sparql"


def _count_response(value: int):
    resp = MagicMock()
    resp.json.return_value = {"results": {"bindings": [{"cnt": {"type": "literal", "value": str(value)}}]}}
    return resp


def _ask_response(boolean: bool):
    resp = MagicMock()
    resp.json.return_value = {"boolean": boolean}
    return resp


def _empty_select_response():
    resp = MagicMock()
    resp.json.return_value = {"head": {"vars": ["x"]}, "results": {"bindings": []}}
    return resp


class TestDomainAxiomsWithNoExplicitDomain(unittest.TestCase):
    """Regression test: `most_general_object_properties` now actually drives
    `object_property_domain_axioms`/`data_property_domain_axioms`/`object_property_range_axioms`
    for every property in the signature (previously these were only exercised in tests when the
    domain was owl:Thing, since `func(prop)` used to be silently miscompared). For a property with
    no explicit rdfs:domain/rdfs:range triple, the underlying `peek()` helper returned a bare `None`
    on an empty generator instead of the `(None, generator)` tuple its callers unpack, crashing with
    `TypeError: cannot unpack non-iterable NoneType object`. Caught by CI running against the live
    Mutagenesis triplestore, where the `inStructure` object property has no declared domain."""

    def setUp(self):
        self.onto = TripleStoreOntology(URL)
        self.object_prop = OWLObjectProperty(IRI.create(NS, "noDomainProp"))
        self.data_prop = OWLDataProperty(IRI.create(NS, "noDomainDataProp"))

    def test_object_property_domain_axioms_falls_back_to_thing(self):
        with patch("ontolearn.triple_store.requests.post", return_value=_empty_select_response()):
            axioms = list(self.onto.object_property_domain_axioms(self.object_prop))
        self.assertEqual(len(axioms), 1)
        self.assertEqual(axioms[0].get_domain(), OWLThing)

    def test_object_property_range_axioms_falls_back_to_thing(self):
        with patch("ontolearn.triple_store.requests.post", return_value=_empty_select_response()):
            axioms = list(self.onto.object_property_range_axioms(self.object_prop))
        self.assertEqual(len(axioms), 1)
        self.assertEqual(axioms[0].get_range(), OWLThing)

    def test_data_property_domain_axioms_falls_back_to_thing(self):
        with patch("ontolearn.triple_store.requests.post", return_value=_empty_select_response()):
            axioms = list(self.onto.data_property_domain_axioms(self.data_prop))
        self.assertEqual(len(axioms), 1)
        self.assertEqual(axioms[0].get_domain(), OWLThing)


class TestIndividualsInSignatureCount(unittest.TestCase):
    def setUp(self):
        self.onto = TripleStoreOntology(URL)

    def test_issues_a_count_query_and_parses_the_binding(self):
        with patch("ontolearn.triple_store.requests.post", return_value=_count_response(7)) as mocked_post:
            count = self.onto.individuals_in_signature_count()
        self.assertEqual(count, 7)
        query = mocked_post.call_args.kwargs["data"]["query"]
        self.assertIn("COUNT(DISTINCT ?x)", query)
        self.assertNotIn("SELECT DISTINCT ?x", query)

    def test_no_bindings_returns_zero(self):
        empty = MagicMock()
        empty.json.return_value = {"results": {"bindings": []}}
        with patch("ontolearn.triple_store.requests.post", return_value=empty):
            self.assertEqual(self.onto.individuals_in_signature_count(), 0)

    def test_implicit_individuals_flag_changes_the_where_clause(self):
        with patch("ontolearn.triple_store.requests.post", return_value=_count_response(0)) as mocked_post:
            self.onto.individuals_in_signature_count(implicit_individuals=False)
        query = mocked_post.call_args.kwargs["data"]["query"]
        self.assertIn("owl:NamedIndividual", query)
        self.assertNotIn("FILTER NOT EXISTS", query)


class TestInstancesCount(unittest.TestCase):
    def setUp(self):
        self.onto = TripleStoreOntology(URL)
        self.reasoner = TripleStoreReasoner(self.onto)
        self.cls = OWLClass(IRI.create(NS, "Person"))

    def test_issues_a_count_query_instead_of_fetching_every_individual(self):
        with patch("ontolearn.triple_store.requests.Session") as mocked_session:
            mocked_session.return_value.post.return_value = _count_response(123)
            count = self.reasoner.instances_count(self.cls)
        self.assertEqual(count, 123)
        # A single COUNT round-trip, not one request per matching individual.
        self.assertEqual(mocked_session.return_value.post.call_count, 1)
        query = mocked_session.return_value.post.call_args.kwargs["data"]["query"]
        self.assertIn("COUNT ( DISTINCT ?x )", query)

    def test_no_bindings_returns_zero(self):
        empty = MagicMock()
        empty.json.return_value = {"results": {"bindings": []}}
        with patch("ontolearn.triple_store.requests.Session") as mocked_session:
            mocked_session.return_value.post.return_value = empty
            self.assertEqual(self.reasoner.instances_count(self.cls), 0)


class TestIsSubsetOfIndividuals(unittest.TestCase):
    def setUp(self):
        self.onto = TripleStoreOntology(URL)
        self.reasoner = TripleStoreReasoner(self.onto)
        self.cls = OWLClass(IRI.create(NS, "Person"))
        self.individuals = [OWLNamedIndividual(IRI.create(NS, "p1")), OWLNamedIndividual(IRI.create(NS, "p2"))]

    def test_empty_individuals_short_circuits_without_a_query(self):
        with patch("ontolearn.triple_store.requests.Session") as mocked_session:
            result = self.reasoner.is_subset_of_individuals([], self.cls)
        self.assertTrue(result)
        mocked_session.return_value.post.assert_not_called()

    def test_ask_query_embeds_values_and_filter_not_exists(self):
        with patch("ontolearn.triple_store.requests.Session") as mocked_session:
            mocked_session.return_value.post.return_value = _ask_response(False)
            self.reasoner.is_subset_of_individuals(self.individuals, self.cls)
        query = mocked_session.return_value.post.call_args.kwargs["data"]["query"]
        self.assertTrue(query.strip().startswith("ASK"))
        self.assertIn(f"VALUES ?x", query)
        self.assertIn(f"<{NS}p1>", query)
        self.assertIn(f"<{NS}p2>", query)
        self.assertIn("FILTER NOT EXISTS", query)

    def test_no_counterexample_means_subset_holds(self):
        with patch("ontolearn.triple_store.requests.Session") as mocked_session:
            mocked_session.return_value.post.return_value = _ask_response(False)
            self.assertTrue(self.reasoner.is_subset_of_individuals(self.individuals, self.cls))

    def test_counterexample_means_subset_does_not_hold(self):
        with patch("ontolearn.triple_store.requests.Session") as mocked_session:
            mocked_session.return_value.post.return_value = _ask_response(True)
            self.assertFalse(self.reasoner.is_subset_of_individuals(self.individuals, self.cls))


class TestTripleStoreIndividualsCount(unittest.TestCase):
    def setUp(self):
        self.kb = TripleStore(url=URL)

    def test_none_or_thing_dispatches_to_ontology_count(self):
        with patch.object(self.kb.ontology, "individuals_in_signature_count", return_value=5) as mocked, \
             patch.object(self.kb.reasoner, "instances_count") as mocked_reasoner:
            self.assertEqual(self.kb.individuals_count(), 5)
            self.assertEqual(self.kb.individuals_count(OWLThing), 5)
        mocked_reasoner.assert_not_called()
        self.assertEqual(mocked.call_count, 2)

    def test_named_class_dispatches_to_reasoner_count(self):
        cls = OWLClass(IRI.create(NS, "Person"))
        with patch.object(self.kb.reasoner, "instances_count", return_value=9) as mocked, \
             patch.object(self.kb.ontology, "individuals_in_signature_count") as mocked_onto:
            self.assertEqual(self.kb.individuals_count(cls), 9)
        mocked.assert_called_once_with(cls)
        mocked_onto.assert_not_called()


class TestMostGeneralObjectProperties(unittest.TestCase):
    """Regression tests for the bug fixed alongside the SPARQL optimization: previously
    `individuals_set(func(prop))` was passed the raw `func(prop)` generator object (not
    the individuals it would yield), so the subset check could only ever pass when
    `domain` was `owl:Thing`."""

    def setUp(self):
        self.kb = TripleStore(url=URL)
        self.person = OWLClass(IRI.create(NS, "Person"))
        self.company = OWLClass(IRI.create(NS, "Company"))
        self.works_for = OWLObjectProperty(IRI.create(NS, "worksFor"))
        self.lives_in = OWLObjectProperty(IRI.create(NS, "livesIn"))
        patcher = patch.object(self.kb.ontology, "object_properties_in_signature",
                                return_value=iter([self.works_for, self.lives_in]))
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_owl_thing_domain_returns_every_property_without_any_http_call(self):
        with patch("ontolearn.triple_store.requests.Session") as mocked_session, \
             patch("ontolearn.triple_store.requests.post") as mocked_post:
            props = list(self.kb.most_general_object_properties(domain=OWLThing))
        self.assertEqual(props, [self.works_for, self.lives_in])
        mocked_session.return_value.post.assert_not_called()
        mocked_post.assert_not_called()

    def test_named_domain_filters_by_declared_property_domain(self):
        def fake_get_object_property_domains(prop, direct=True):
            return iter([self.person] if prop == self.works_for else [self.company])

        def fake_is_subset_of_individuals(individuals, ce, direct=False):
            # worksFor's domain (Person) matches the query domain -> subset holds.
            # livesIn's domain (Company) does not -> subset fails.
            return ce == self.person

        with patch.object(self.kb, "get_object_property_domains", side_effect=fake_get_object_property_domains), \
             patch.object(self.kb, "individuals_set", return_value=frozenset({OWLNamedIndividual(IRI.create(NS, "p1"))})), \
             patch.object(self.kb.reasoner, "is_subset_of_individuals", side_effect=fake_is_subset_of_individuals):
            props = list(self.kb.most_general_object_properties(domain=self.person))
        self.assertEqual(props, [self.works_for])

    def test_property_without_any_declared_domain_is_skipped(self):
        with patch.object(self.kb, "get_object_property_domains", return_value=iter([])), \
             patch.object(self.kb, "individuals_set", return_value=frozenset({OWLNamedIndividual(IRI.create(NS, "p1"))})), \
             patch.object(self.kb.reasoner, "is_subset_of_individuals") as mocked_subset:
            props = list(self.kb.most_general_object_properties(domain=self.person))
        self.assertEqual(props, [])
        mocked_subset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
