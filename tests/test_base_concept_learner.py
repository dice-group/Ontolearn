import pandas as pd
import unittest
from ontolearn.concept_learner import EvoLearner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.search import EvoLearnerNode
from owlapy.model import OWLClass, OWLClassAssertionAxiom, OWLNamedIndividual, IRI, OWLObjectIntersectionOf, \
                         OWLObjectProperty, OWLObjectPropertyAssertionAxiom, OWLObjectSomeValuesFrom, OWLThing
from owlapy.render import DLSyntaxObjectRenderer


class BaseConceptLearnerTest(unittest.TestCase):

    def setUp(self):
        kb = KnowledgeBase(path='KGs/father.owl')
        self.model = EvoLearner(knowledge_base=kb)
        self.namespace = 'http://example.com/father#'

        self.male = OWLClass(IRI.create(self.namespace, 'male'))
        self.female = OWLClass(IRI.create(self.namespace, 'female'))
        self.has_child = OWLObjectProperty(IRI.create(self.namespace, 'hasChild'))

        self.renderer = DLSyntaxObjectRenderer()

    def test_predict(self):
        anna = OWLNamedIndividual(IRI.create(self.namespace, 'anna'))
        markus = OWLNamedIndividual(IRI.create(self.namespace, 'markus'))
        michelle = OWLNamedIndividual(IRI.create(self.namespace, 'michelle'))
        individuals = [anna, markus, michelle]

        hyp1 = OWLObjectIntersectionOf([self.male, OWLObjectSomeValuesFrom(self.has_child, OWLThing)])
        hyp2 = OWLObjectIntersectionOf([self.female, OWLObjectSomeValuesFrom(self.has_child, OWLThing)])
        hyp3 = self.female
        hypotheses = [hyp1, hyp2, hyp3]

        predictions = self.model.predict(individuals=individuals, hypotheses=hypotheses)
        labels = pd.DataFrame({self.renderer.render(hyp1): [0.0, 1.0, 0.0],
                               self.renderer.render(hyp2): [1.0, 0.0, 0.0],
                               self.renderer.render(hyp3): [1.0, 0.0, 1.0]},
                              index=["anna", "markus", "michelle"])
        self.assertTrue(predictions.equals(labels))

        # Same test with nodes instead of hypotheses, all statistics are set to 0 since they don't matter here
        node1 = EvoLearnerNode(hyp1, 0, 0, 0.0, 0, 0)
        node2 = EvoLearnerNode(hyp2, 0, 0, 0.0, 0, 0)
        node3 = EvoLearnerNode(hyp3, 0, 0, 0.0, 0, 0)
        hypotheses = [node1, node2, node3]
        predictions = self.model.predict(individuals=individuals, hypotheses=hypotheses)
        self.assertTrue(predictions.equals(labels))

    def test_predict_new_individuals(self):
        julia = OWLNamedIndividual(IRI.create(self.namespace, 'julia'))
        julian = OWLNamedIndividual(IRI.create(self.namespace, 'julian'))
        thomas = OWLNamedIndividual(IRI.create(self.namespace, 'thomas'))
        individuals = [julia, julian, thomas]

        axiom1 = OWLClassAssertionAxiom(individual=julia, class_expression=self.female)
        axiom2 = OWLClassAssertionAxiom(individual=julian, class_expression=self.male)
        axiom3 = OWLClassAssertionAxiom(individual=thomas, class_expression=self.male)
        axiom4 = OWLObjectPropertyAssertionAxiom(subject=thomas, property_=self.has_child, object_=julian)
        axiom5 = OWLObjectPropertyAssertionAxiom(subject=julia, property_=self.has_child, object_=julian)
        axioms = [axiom1, axiom2, axiom3, axiom4, axiom5]

        hyp1 = OWLObjectIntersectionOf([self.male, OWLObjectSomeValuesFrom(self.has_child, OWLThing)])
        hyp2 = OWLObjectIntersectionOf([self.female, OWLObjectSomeValuesFrom(self.has_child, OWLThing)])
        hyp3 = self.female
        hyp4 = self.male
        hypotheses = [hyp1, hyp2, hyp3, hyp4]

        predictions = self.model.predict_new_individuals(individuals=individuals, axioms=axioms, hypotheses=hypotheses)
        labels = pd.DataFrame({self.renderer.render(hyp1): [0.0, 0.0, 1.0],
                               self.renderer.render(hyp2): [1.0, 0.0, 0.0],
                               self.renderer.render(hyp3): [1.0, 0.0, 0.0],
                               self.renderer.render(hyp4): [0.0, 1.0, 1.0]},
                              index=["julia", "julian", "thomas"])
        self.assertTrue(predictions.equals(labels))

        # Check if axioms are removed afterward
        # father.owl should have 6 individuals: 2 females and 4 males
        kb = self.model.kb
        self.assertEqual(len(kb.individuals_set(OWLThing)), 6)
        self.assertEqual(len(kb.individuals_set(self.female)), 2)
        self.assertEqual(len(kb.individuals_set(self.male)), 4)
