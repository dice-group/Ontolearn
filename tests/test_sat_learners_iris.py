"""Regression tests for IRI preservation in SAT-based learners (ALCSAT and SPELL)."""

import os
import tempfile
import unittest

from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import ALCSAT, SPELL
from ontolearn.learning_problem import PosNegLPStandard


SLASH_AND_MULTI_NS_OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/ont/"
     xml:base="http://example.org/ont/"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:ext="http://other.example.org/vocab/">
    <owl:Ontology rdf:about="http://example.org/ont/"/>
    <owl:Class rdf:about="http://example.org/ont/Parent"/>
    <owl:Class rdf:about="http://other.example.org/vocab/Parent"/>
    <owl:Class rdf:about="http://other.example.org/vocab/Special"/>
    <owl:ObjectProperty rdf:about="http://example.org/ont/hasChild"/>
    <owl:NamedIndividual rdf:about="http://example.org/ont/alice">
        <rdf:type rdf:resource="http://example.org/ont/Parent"/>
        <hasChild rdf:resource="http://example.org/ont/charlie"/>
    </owl:NamedIndividual>
    <owl:NamedIndividual rdf:about="http://example.org/ont/diana">
        <rdf:type rdf:resource="http://example.org/ont/Parent"/>
    </owl:NamedIndividual>
    <owl:NamedIndividual rdf:about="http://example.org/ont/bob">
        <rdf:type rdf:resource="http://other.example.org/vocab/Parent"/>
    </owl:NamedIndividual>
    <owl:NamedIndividual rdf:about="http://example.org/ont/charlie">
        <rdf:type rdf:resource="http://other.example.org/vocab/Special"/>
    </owl:NamedIndividual>
</rdf:RDF>
"""

HASH_NS_OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/family#"
     xml:base="http://example.org/family"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.org/family"/>
    <owl:Class rdf:about="http://example.org/family#Father"/>
    <owl:NamedIndividual rdf:about="http://example.org/family#john">
        <rdf:type rdf:resource="http://example.org/family#Father"/>
    </owl:NamedIndividual>
    <owl:NamedIndividual rdf:about="http://example.org/family#mary"/>
</rdf:RDF>
"""


class TestSATLearnersPreserveIRIs(unittest.TestCase):
    """ALCSAT and SPELL preserve full class and object-property IRIs."""

    @classmethod
    def setUpClass(cls):
        fd1, cls.slash_path = tempfile.mkstemp(suffix=".owl")
        with os.fdopen(fd1, "w", encoding="utf-8") as f:
            f.write(SLASH_AND_MULTI_NS_OWL)
        cls.kb = KnowledgeBase(path=cls.slash_path)
        cls.alice = OWLNamedIndividual(IRI.create("http://example.org/ont/alice"))
        cls.diana = OWLNamedIndividual(IRI.create("http://example.org/ont/diana"))
        cls.bob = OWLNamedIndividual(IRI.create("http://example.org/ont/bob"))
        cls.charlie = OWLNamedIndividual(IRI.create("http://example.org/ont/charlie"))

        fd2, cls.hash_path = tempfile.mkstemp(suffix=".owl")
        with os.fdopen(fd2, "w", encoding="utf-8") as f:
            f.write(HASH_NS_OWL)
        cls.hash_kb = KnowledgeBase(path=cls.hash_path)
        cls.john = OWLNamedIndividual(IRI.create("http://example.org/family#john"))
        cls.mary = OWLNamedIndividual(IRI.create("http://example.org/family#mary"))

    @classmethod
    def tearDownClass(cls):
        for path in (cls.slash_path, cls.hash_path):
            if os.path.exists(path):
                os.unlink(path)

    def test_alcsat_distinguishes_same_remainder_across_namespaces(self):
        lp = PosNegLPStandard(pos={self.alice, self.diana}, neg={self.bob, self.charlie})
        learner = ALCSAT(knowledge_base=self.kb, max_concept_size=3, max_runtime=15)
        learner.fit(lp)
        hypothesis = learner.best_hypothesis()
        self.assertEqual(learner.best_hypothesis_accuracy(), 1.0)
        self.assertEqual(hypothesis, OWLClass(IRI.create("http://example.org/ont/Parent")))
        self.assertEqual(set(self.kb.individuals(hypothesis)), {self.alice, self.diana})

    def test_spell_distinguishes_same_remainder_across_namespaces(self):
        lp = PosNegLPStandard(pos={self.bob}, neg={self.alice, self.diana, self.charlie})
        learner = SPELL(knowledge_base=self.kb, max_query_size=3, search_mode="exact", max_runtime=15)
        learner.fit(lp)
        hypothesis = learner.best_hypothesis()
        self.assertEqual(learner.best_hypothesis_accuracy(), 1.0)
        self.assertEqual(hypothesis, OWLClass(IRI.create("http://other.example.org/vocab/Parent")))
        self.assertEqual(set(self.kb.individuals(hypothesis)), {self.bob})

    def test_alcsat_preserves_slash_role_iri(self):
        # alice and diana have identical class types, so only the hasChild relation separates alice.
        lp = PosNegLPStandard(pos={self.alice}, neg={self.diana, self.bob, self.charlie})
        learner = ALCSAT(knowledge_base=self.kb, max_concept_size=3, max_runtime=15)
        learner.fit(lp)
        hypothesis = learner.best_hypothesis()
        self.assertEqual(learner.best_hypothesis_accuracy(), 1.0)
        self.assertEqual(set(self.kb.individuals(hypothesis)), {self.alice})

    def test_spell_preserves_slash_role_iri(self):
        lp = PosNegLPStandard(pos={self.alice}, neg={self.diana, self.bob, self.charlie})
        learner = SPELL(knowledge_base=self.kb, max_query_size=3, search_mode="exact", max_runtime=15)
        learner.fit(lp)
        hypothesis = learner.best_hypothesis()
        self.assertEqual(learner.best_hypothesis_accuracy(), 1.0)
        self.assertEqual(set(self.kb.individuals(hypothesis)), {self.alice})

    def test_hash_namespace_ontology_still_works(self):
        lp = PosNegLPStandard(pos={self.john}, neg={self.mary})
        for learner_cls in (ALCSAT, SPELL):
            learner = learner_cls(knowledge_base=self.hash_kb, max_runtime=15)
            learner.fit(lp)
            hypothesis = learner.best_hypothesis()
            self.assertEqual(learner.best_hypothesis_accuracy(), 1.0)
            self.assertEqual(hypothesis, OWLClass(IRI.create("http://example.org/family#Father")))
            self.assertEqual(set(self.hash_kb.individuals(hypothesis)), {self.john})


if __name__ == "__main__":
    unittest.main()
