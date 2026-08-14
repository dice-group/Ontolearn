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

from ontolearn.triple_store import TripleStoreOntology, TripleStoreReasoner
from owlapy.class_expression import OWLClass, OWLObjectComplementOf, OWLObjectIntersectionOf
from owlapy.iri import IRI

NS = "http://example.org/family#"
URL = "http://localhost:3030/fake/sparql"

SUBCLASS_BINDING_RE = re.compile(
    r"(\?some_cls\w*)\s+<http://www\.w3\.org/2000/01/rdf-schema#subClassOf>\*\s+<([^>]+)>"
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
