"""Regression test for https://github.com/dice-group/Ontolearn/issues/575.

`TripleStoreReasoner.instances(ce, direct=False)` rewrites every `?x a ` occurrence in the
SPARQL generated for `ce` to introduce a `?some_cls` variable ranging over subclasses, so that
indirect (non-asserted) instances are also matched. When `ce` places more than one such
occurrence in different scopes -- e.g. one inside a `FILTER NOT EXISTS` block and one outside
it, as happens for `(NOT Daughter) AND Female` -- reusing the same variable name for every
occurrence lets them bind to each other across scopes instead of being scoped independently,
producing incorrect results.

Mocks the HTTP layer (like `tests/test_triplestore_sparql_optimizations.py`) so it exercises
the query-construction logic without requiring a live triplestore.
"""
import re
import unittest
from unittest.mock import MagicMock, patch

from ontolearn.triple_store import TripleStoreOntology, TripleStoreReasoner, with_subclass_closure
from ontolearn.triple_store import _PLAIN_SUBCLASS_PATH, _SUBCLASS_PATH
from owlapy.class_expression import (OWLClass, OWLObjectComplementOf, OWLObjectIntersectionOf,
                                     OWLObjectSomeValuesFrom)
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLObjectProperty

NS = "http://example.org/family#"
URL = "http://localhost:3030/fake/sparql"

SUBCLASS_BINDING_RE = re.compile(
    r"(\?some_cls\w*)\s+\(<http://www\.w3\.org/2000/01/rdf-schema#subClassOf>"
    r"\|<http://www\.w3\.org/2002/07/owl#equivalentClass>\|\^<http://www\.w3\.org/2002/07/owl#equivalentClass>\)\*"
    r"\s+<([^>]+)>"
)


def _empty_select_response():
    resp = MagicMock()
    resp.json.return_value = {"head": {"vars": ["x"]}, "results": {"bindings": []}}
    return resp


class TestIndirectInstancesSparqlScoping(unittest.TestCase):
    def setUp(self):
        self.daughter = OWLClass(IRI.create(NS, "Daughter"))
        self.female = OWLClass(IRI.create(NS, "Female"))
        self.onto = TripleStoreOntology(URL)
        self.reasoner = TripleStoreReasoner(self.onto)

    def _generated_query(self, ce, direct):
        # `instances()` may issue further HTTP calls afterwards (e.g. `equivalent_classes`
        # recursion for indirect retrieval on a named class), so always take the first call,
        # which is the query for `ce` itself.
        with patch("ontolearn.triple_store.requests.post", return_value=_empty_select_response()) as mocked_post:
            list(self.reasoner.instances(ce, direct=direct))
        return mocked_post.call_args_list[0].kwargs["data"]["query"]

    def test_each_subclass_occurrence_is_scoped_by_its_own_variable(self):
        # (NOT Daughter) AND Female: the converter places "?x a " for Daughter inside a
        # FILTER NOT EXISTS block, and "?x a " for Female outside it.
        ce = OWLObjectIntersectionOf([OWLObjectComplementOf(self.daughter), self.female])
        query = self._generated_query(ce, direct=False)

        bindings = SUBCLASS_BINDING_RE.findall(query)
        self.assertEqual(len(bindings), 2, f"expected exactly 2 subclass bindings, query was:\n{query}")

        var_by_class = {iri: var for var, iri in bindings}
        self.assertIn(self.daughter.str, var_by_class)
        self.assertIn(self.female.str, var_by_class)
        # The crux of #575: each occurrence must be scoped by its own variable, not
        # share one across the FILTER NOT EXISTS boundary.
        self.assertNotEqual(var_by_class[self.daughter.str], var_by_class[self.female.str])

        # No introduced variable name is reused across occurrences.
        variables = [var for var, _ in bindings]
        self.assertEqual(len(variables), len(set(variables)))

    def test_single_occurrence_still_rewritten_for_indirect_retrieval(self):
        query = self._generated_query(self.female, direct=False)
        bindings = SUBCLASS_BINDING_RE.findall(query)
        self.assertEqual(bindings, [("?some_cls_0", self.female.str)])

    def test_direct_true_does_not_rewrite_the_query(self):
        ce = OWLObjectIntersectionOf([OWLObjectComplementOf(self.daughter), self.female])
        query = self._generated_query(ce, direct=True)
        self.assertNotIn("some_cls", query)


if __name__ == "__main__":
    unittest.main()


class TestSubclassClosureOfFillers(unittest.TestCase):
    """https://github.com/dice-group/Ontolearn/issues/571: the subclass closure must cover every type
    triple of the converted query, not only the one on the root variable."""

    def setUp(self):
        self.grandfather = OWLClass(IRI.create(NS, "Grandfather"))
        self.grandparent = OWLClass(IRI.create(NS, "Grandparent"))
        self.has_child = OWLObjectProperty(IRI.create(NS, "hasChild"))
        self.ce = OWLObjectIntersectionOf(
            [self.grandfather, OWLObjectSomeValuesFrom(self.has_child, self.grandparent)])
        self.reasoner = TripleStoreReasoner(TripleStoreOntology(URL))

    def test_filler_type_triple_is_expanded(self):
        query = with_subclass_closure(self.reasoner._owl2sparql_converter.as_query("?x", self.ce))
        bindings = SUBCLASS_BINDING_RE.findall(query)
        self.assertEqual({iri for _, iri in bindings}, {self.grandfather.str, self.grandparent.str})
        self.assertEqual(len({var for var, _ in bindings}), 2)
        self.assertIn("?s_1 a ?some_cls_", query)

    def test_owl_named_individual_is_not_expanded(self):
        sparql = "SELECT ?x WHERE { ?x a <http://www.w3.org/2002/07/owl#NamedIndividual> . ?x a <http://e.org/A> . }"
        self.assertEqual(SUBCLASS_BINDING_RE.findall(with_subclass_closure(sparql)), [("?some_cls_0", "http://e.org/A")])

    def test_count_and_subset_queries_scope_each_occurrence(self):
        ce = OWLObjectIntersectionOf([OWLObjectComplementOf(OWLClass(IRI.create(NS, "Daughter"))),
                                      OWLClass(IRI.create(NS, "Female"))])
        resp = MagicMock()
        resp.json.return_value = {"head": {"vars": ["cnt"]}, "boolean": False,
                                  "results": {"bindings": [{"cnt": {"value": "0"}}]}}
        self.reasoner._subclass_path = _SUBCLASS_PATH  # skip the one-off owl:equivalentClass probe
        with patch.object(TripleStoreReasoner, "query", return_value=resp) as mocked_query:
            self.reasoner.instances_count(ce)
            self.reasoner.is_subset_of_individuals([OWLNamedIndividual(IRI.create(NS, "a"))], ce)
        self.assertEqual(mocked_query.call_count, 2)
        for call in mocked_query.call_args_list:
            variables = [var for var, _ in SUBCLASS_BINDING_RE.findall(call.args[0])]
            self.assertEqual(len(variables), 2)
            self.assertEqual(len(variables), len(set(variables)))

    def test_indirect_instances_found_through_subclass_fillers(self):
        """Run the rewritten query on a tiny graph where the only type assertions are for subclasses."""
        from rdflib import Graph
        g = Graph()
        g.parse(format="turtle", data=f"""
            @prefix : <{NS}> . @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            :Grandfather rdfs:subClassOf :Father .  :Grandparent rdfs:subClassOf :Parent .
            :GrandfatherOfTwo rdfs:subClassOf :Grandfather .  :GreatGrandparent rdfs:subClassOf :Grandparent .
            :a a :GrandfatherOfTwo ; :hasChild :b .   :b a :GreatGrandparent .
            :c a :GrandfatherOfTwo ; :hasChild :d .   :d a :Parent .
        """)
        base = self.reasoner._owl2sparql_converter.as_query("?x", self.ce)
        names = lambda q: {str(r[0]).split("#")[1] for r in g.query(q)}
        self.assertEqual(names(base), set())  # no asserted Grandfather/Grandparent types
        self.assertEqual(names(with_subclass_closure(base)), {"a"})  # c's child is only a Parent, not a Grandparent


class TestEquivalentClassesInSubclassClosure(unittest.TestCase):
    """The closure path also follows owl:equivalentClass, at every nesting level (#571)."""

    def setUp(self):
        self.reasoner = TripleStoreReasoner(TripleStoreOntology(URL))
        self.person = OWLClass(IRI.create(NS, "Person"))
        self.parent = OWLClass(IRI.create(NS, "Parent"))
        self.has_child = OWLObjectProperty(IRI.create(NS, "hasChild"))

    def test_filler_matches_individuals_typed_by_an_equivalent_class(self):
        from rdflib import Graph
        g = Graph()
        g.parse(format="turtle", data=f"""
            @prefix : <{NS}> . @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            :Mother rdfs:subClassOf :Parent .
            :Guardian owl:equivalentClass :Parent .          # declared on the other side: Guardian = Parent
            :Carer owl:equivalentClass :Mother .              # equivalent to a subclass of Parent
            :Unrelated rdfs:subClassOf :Person .
            :a a :Person ; :hasChild :b .  :b a :Mother .      # subclass
            :c a :Person ; :hasChild :d .  :d a :Guardian .    # equivalent class
            :e a :Person ; :hasChild :f .  :f a :Carer .       # equivalent class of a subclass
            :g a :Person ; :hasChild :h .  :h a :Unrelated .   # not a parent
        """)
        ce = OWLObjectIntersectionOf([self.person, OWLObjectSomeValuesFrom(self.has_child, self.parent)])
        query = with_subclass_closure(self.reasoner._owl2sparql_converter.as_query("?x", ce))
        self.assertEqual({str(r[0]).split("#")[1] for r in g.query(query)}, {"a", "c", "e"})

    def test_instances_issues_a_single_query_without_equivalent_class_recursion(self):
        self.reasoner._subclass_path = _SUBCLASS_PATH  # skip the one-off owl:equivalentClass probe
        with patch("ontolearn.triple_store.requests.post", return_value=_empty_select_response()) as mocked_post:
            list(self.reasoner.instances(self.parent, direct=False))
        self.assertEqual(mocked_post.call_count, 1)


class TestEquivalencePathSelection(unittest.TestCase):
    """The equivalence-aware path is only used for stores that have owl:equivalentClass triples."""

    def setUp(self):
        self.reasoner = TripleStoreReasoner(TripleStoreOntology(URL))
        self.sparql = "SELECT DISTINCT ?x WHERE { ?x a <http://example.org/f#A> . }"

    def _closure(self, ask_response=None, ask_error=None):
        resp = MagicMock()
        resp.json.return_value = {"boolean": ask_response}
        with patch.object(TripleStoreReasoner, "query", return_value=resp, side_effect=ask_error) as mocked_query:
            return self.reasoner._closure(self.sparql), mocked_query

    def test_store_without_equivalent_classes_uses_the_plain_path(self):
        query, _ = self._closure(ask_response=False)
        self.assertIn(_PLAIN_SUBCLASS_PATH, query)
        self.assertNotIn("equivalentClass", query)

    def test_store_with_equivalent_classes_uses_the_full_path(self):
        query, _ = self._closure(ask_response=True)
        self.assertIn(_SUBCLASS_PATH, query)

    def test_failing_probe_falls_back_to_the_full_path(self):
        query, _ = self._closure(ask_error=RuntimeError("endpoint does not support ASK"))
        self.assertIn(_SUBCLASS_PATH, query)

    def test_probe_runs_once_per_reasoner(self):
        resp = MagicMock()
        resp.json.return_value = {"boolean": False}
        with patch.object(TripleStoreReasoner, "query", return_value=resp) as mocked_query:
            self.reasoner._closure(self.sparql)
            self.reasoner._closure(self.sparql)
        self.assertEqual(mocked_query.call_count, 1)
