import unittest
import tempfile
import pandas as pd
from owlapy.class_expression import OWLClass, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLThing
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLClassAssertionAxiom, OWLObjectPropertyAssertionAxiom
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLObjectProperty

from ontolearn.concept_learner import CELOE
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.search import EvoLearnerNode
from owlapy.render import DLSyntaxObjectRenderer


class TestBaseConceptLearner(unittest.TestCase):

    def setUp(self):
        kb = KnowledgeBase(path='KGs/Family/father.owl')
        self.model = CELOE(knowledge_base=kb)
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

        predictions = self.model.predict(individuals=individuals, axioms=axioms, hypotheses=hypotheses)
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

    def test_predict_new_individuals_without_axioms(self):
        julia = OWLNamedIndividual(IRI.create(self.namespace, 'julia'))
        anna = OWLNamedIndividual(IRI.create(self.namespace, 'anna'))
        individuals = [julia, anna]
        hypotheses = [self.male]
        # Should throw an error when no axioms are provided
        self.assertRaises(RuntimeError, self.model.predict, individuals, hypotheses)

    def test_learn_predict_workflow(self):
        examples = {
            'positive_examples': [
                OWLNamedIndividual(IRI.create("http://example.com/father#stefan")),
                OWLNamedIndividual(IRI.create("http://example.com/father#markus")),
                OWLNamedIndividual(IRI.create("http://example.com/father#martin"))],
            'negative_examples': [
                OWLNamedIndividual(IRI.create("http://example.com/father#heinz")),
                OWLNamedIndividual(IRI.create("http://example.com/father#anna")),
                OWLNamedIndividual(IRI.create("http://example.com/father#michelle"))]
        }

        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        lp = PosNegLPStandard(pos=p, neg=n)

        # 1. Learn class expressions
        model = self.model.fit(learning_problem=lp)

        # 2., 3. Save and load class expressions
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = tmpdirname + '/Predictions'
            model.save_best_hypothesis(n=2, path=file_path)
            hypotheses = list(model.load_hypotheses(file_path + '.owl'))

        # New Individuals
        julia = OWLNamedIndividual(IRI.create(self.namespace, 'julia'))
        julian = OWLNamedIndividual(IRI.create(self.namespace, 'julian'))
        thomas = OWLNamedIndividual(IRI.create(self.namespace, 'thomas'))
        # Existing Individuals
        anna = OWLNamedIndividual(IRI.create(self.namespace, 'anna'))
        markus = OWLNamedIndividual(IRI.create(self.namespace, 'markus'))
        michelle = OWLNamedIndividual(IRI.create(self.namespace, 'michelle'))
        individuals = [julia, julian, thomas, anna, markus, michelle]

        axiom1 = OWLClassAssertionAxiom(individual=julia, class_expression=self.female)
        axiom2 = OWLClassAssertionAxiom(individual=julian, class_expression=self.male)
        axiom3 = OWLClassAssertionAxiom(individual=thomas, class_expression=self.male)
        axiom4 = OWLObjectPropertyAssertionAxiom(subject=thomas, property_=self.has_child, object_=julian)
        axiom5 = OWLObjectPropertyAssertionAxiom(subject=julia, property_=self.has_child, object_=julian)
        axioms = [axiom1, axiom2, axiom3, axiom4, axiom5]

        # 4. Use loaded class expressions for predictions
        predictions = self.model.predict(individuals=individuals, axioms=axioms, hypotheses=hypotheses)
        # Assuming predictions of CELOE are (¬female) ⊓ (∃ hasChild.⊤) and male
        labels = pd.DataFrame({self.renderer.render(hypotheses[0]): [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                               self.renderer.render(hypotheses[1]): [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]},
                              index=["julia", "julian", "thomas", "anna", "markus", "michelle"])
        self.assertTrue(predictions.equals(labels))
