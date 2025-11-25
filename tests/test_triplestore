import unittest

from six import assertCountEqual

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL
from ontolearn.triple_store import TripleStore, TripleStoreReasoner, TripleStoreOntology
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.class_expression import OWLClass, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLThing, \
    OWLClassExpression, OWLObjectSomeValuesFrom, OWLObjectOneOf
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLDisjointClassesAxiom, OWLDeclarationAxiom, OWLClassAssertionAxiom
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLBottomObjectProperty, OWLTopObjectProperty, OWLBottomDataProperty, OWLTopDataProperty, \
    OWLLiteral
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty
from owlapy.owl_reasoner import SyncReasoner
from owlapy.providers import owl_datatype_min_inclusive_restriction
import json


class TestTriplestore(unittest.TestCase):
    ns = "http://dl-learner.org/mutagenesis#"
    ontology_path = "KGs/Mutagenesis/mutagenesis.owl"
    native_kb = KnowledgeBase(path=ontology_path)
    native_onto = native_kb.ontology
    nitrogen38 = OWLClass(IRI.create(ns, "Nitrogen-38"))
    compound = OWLClass(IRI.create(ns, "Compound"))
    atom = OWLClass(IRI.create(ns, "Atom"))
    charge = OWLDataProperty(IRI.create(ns, "charge"))
    hasAtom = OWLObjectProperty(IRI.create(ns, "hasAtom"))
    d100_25 = OWLNamedIndividual(IRI.create(ns, "d100_25"))
    has_charge_more_than_0_85 = OWLDataSomeValuesFrom(charge, owl_datatype_min_inclusive_restriction(0.85))
    ce = OWLObjectIntersectionOf([nitrogen38, has_charge_more_than_0_85])
    onto = TripleStoreOntology("http://localhost:3030/mutagenesis/sparql")
    reasoner = TripleStoreReasoner(onto)

    def test_triplestore_runs_error_free(self):
        kb = TripleStore(url="http://localhost:3030/mutagenesis/sparql")
        model = TDL(knowledge_base=kb)
        with open('LPs/Mutagenesis/lps.json') as json_file:
            settings = json.load(json_file)
        p = set(settings['problems']['NotKnown']['positive_examples'])
        n = set(settings['problems']['NotKnown']['negative_examples'])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        model.fit(lp)
        hypotheses = model.best_hypotheses(n=3)
        [print(_) for _ in hypotheses]

    def test_ontology_signature_methods(self):
        self.assertCountEqual(list(self.onto.classes_in_signature()), list(self.native_onto.classes_in_signature()))
        self.assertCountEqual(list(self.onto.individuals_in_signature()), list(self.native_onto.individuals_in_signature()))
        self.assertCountEqual(list(self.onto.object_properties_in_signature()), list(self.native_onto.object_properties_in_signature()))
        self.assertCountEqual(list(self.onto.data_properties_in_signature()), list(self.native_onto.data_properties_in_signature()))

    def test_instances_retrieval(self):
        instances = list(self.reasoner.instances(self.ce))
        expected = [OWLNamedIndividual(IRI('http://dl-learner.org/mutagenesis#', 'd141_10')),
                    OWLNamedIndividual(IRI('http://dl-learner.org/mutagenesis#', 'd195_12')),
                    OWLNamedIndividual(IRI('http://dl-learner.org/mutagenesis#', 'd144_10')),
                    OWLNamedIndividual(IRI('http://dl-learner.org/mutagenesis#', 'd147_11')),
                    OWLNamedIndividual(IRI('http://dl-learner.org/mutagenesis#', 'e18_9')),
                    OWLNamedIndividual(IRI('http://dl-learner.org/mutagenesis#', 'd175_17')),
                    OWLNamedIndividual(IRI('http://dl-learner.org/mutagenesis#', 'e16_9'))]
        # Assert equal without considering the order
        for instance in instances:
            self.assertIn(instance, expected)
        self.assertEqual(len(list(instances)), len(expected))

    def test_object_property_domains(self):
        self.assertCountEqual(list(self.reasoner.object_property_domains(self.hasAtom, False)),
                              [self.compound])
        self.assertCountEqual(list(self.reasoner.object_property_domains(self.hasAtom, True)), [self.compound])

    def test_data_property_values(self):
        self.assertCountEqual(list(self.reasoner.data_property_values(self.d100_25, self.charge)), [OWLLiteral(0.332)])


    def test_local_triplestore_family_tdl(self):
        # @TODO: CD: Removed because rdflib does not produce correct results
        """

        # (1) Load a knowledge graph.
        kb = TripleStore(path='KGs/Family/family-benchmark_rich_background.owl')
        # (2) Get learning problems.
        with open("LPs/Family/lps.json") as json_file:
            settings = json.load(json_file)
        # (3) Initialize learner.
        model = TDL(knowledge_base=kb, kwargs_classifier={"max_depth": 2})
        # (4) Fitting.
        for str_target_concept, examples in settings['problems'].items():
            p = set(examples['positive_examples'])
            n = set(examples['negative_examples'])
            print('Target concept: ', str_target_concept)
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            h = model.fit(learning_problem=lp).best_hypotheses()
            print(h)
            predicted_expression = frozenset({i for i in kb.individuals(h)})
            print("Number of individuals:", len(predicted_expression))
            q = compute_f1_score(individuals=predicted_expression, pos=lp.pos, neg=lp.neg)
            print(q)
            assert q>=0.80
            break
        """
    def test_remote_triplestore_dbpedia_tdl(self):
        """
        url = "http://dice-dbpedia.cs.upb.de:9080/sparql"
        kb = TripleStore(url=url)
        # Check whether there is a connection
        num_object_properties = len([i for i in kb.get_object_properties()])
        if num_object_properties > 0:
            examples = {"positive_examples": ["http://dbpedia.org/resource/Angela_Merkel"],
                        "negative_examples": ["http://dbpedia.org/resource/Barack_Obama"]}
            model = TDL(knowledge_base=kb, report_classification=True, kwargs_classifier={"random_state": 1})
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, examples["positive_examples"])))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, examples["negative_examples"])))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            h = model.fit(learning_problem=lp).best_hypotheses()
            assert h
            assert DLSyntaxObjectRenderer().render(h)
            save_owl_class_expressions(h)
            sparql = Owl2SparqlConverter().as_query("?x", h)
            assert sparql
        else:
            pass
        """

