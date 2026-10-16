"""Regression guard for #581.

owlapy <= 1.6.5 rendered a ``owl:Thing`` filler as an explicit ``?s a owl:Thing`` pattern. A triplestore without
reasoning holds no such triples, so e.g. ``∃r.⊤`` matched nothing there while ``KnowledgeBase`` (which treats ⊤
as "any value") returned the expected individuals. These tests need no triplestore: the generated SPARQL is run
on an in-memory rdflib graph of the raw (unmaterialized) ontology.
"""
import unittest

import rdflib
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectUnionOf
from owlapy.converter import Owl2SparqlConverter
from owlapy.iri import IRI
from owlapy.owl_property import OWLObjectProperty

from ontolearn.knowledge_base import KnowledgeBase

PATH_FAMILY = "KGs/Family/family-benchmark_rich_background.owl"
NS = "http://www.benchmark.org/family#"
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"

THING = OWLClass(IRI.create(OWL_THING))
HAS_CHILD = OWLObjectProperty(IRI.create(NS + "hasChild"))
MARRIED = OWLObjectProperty(IRI.create(NS + "married"))
FEMALE = OWLClass(IRI.create(NS + "Female"))


class TestSparqlOwlThingFiller(unittest.TestCase):

    def test_top_filler_emits_no_type_pattern(self):
        for ce in (OWLObjectSomeValuesFrom(HAS_CHILD, THING),
                   OWLObjectUnionOf([OWLObjectIntersectionOf([OWLObjectSomeValuesFrom(HAS_CHILD, THING), FEMALE]),
                                     OWLObjectSomeValuesFrom(MARRIED, THING)])):
            query = Owl2SparqlConverter().as_query("?x", ce)
            self.assertNotIn(f"<{OWL_THING}>", query, query)

    def test_top_filler_matches_knowledge_base_on_raw_triples(self):
        kb = KnowledgeBase(path=PATH_FAMILY)
        graph = rdflib.Graph()
        graph.parse(PATH_FAMILY)
        ce = OWLObjectUnionOf([OWLObjectIntersectionOf([OWLObjectSomeValuesFrom(HAS_CHILD, THING), FEMALE]),
                               OWLObjectSomeValuesFrom(MARRIED, THING)])
        expected = {i.str for i in kb.individuals_set(ce)}
        self.assertGreater(len(expected), 0)

        query = Owl2SparqlConverter().as_query("?x", ce)
        got = {str(row[0]) for row in graph.query(query)}
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
