"""
Unit tests for the Tree-based Description Logic (TDL) learner.

This test suite covers:
- Feature extraction methods
- Configuration flags (use_nominals, use_inverse, use_data_properties, use_card_restrictions)
- Filter logic for expressions
- Training and prediction functionality
- Integration with different knowledge bases
"""

import unittest
from ontolearn.learners import TDL
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.class_expression import (
    OWLClass,
    OWLObjectOneOf,
    OWLObjectHasValue,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLDataSomeValuesFrom,
    OWLObjectIntersectionOf,
    OWLObjectUnionOf
)
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty
from ontolearn.utils.static_funcs import compute_f1_score
import json
import numpy as np
import pandas as pd


class TestTDLBasics(unittest.TestCase):
    """Test basic TDL functionality and initialization."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        cls.path_family = "KGs/Family/family-benchmark_rich_background.owl"
        cls.kb_family = KnowledgeBase(path=cls.path_family)
        
        # Load test learning problems
        with open("LPs/Family/lps.json") as json_file:
            cls.family_lps = json.load(json_file)

    def test_initialization_custom(self):
        """Test TDL initialization with custom parameters."""
        model = TDL(
            knowledge_base=self.kb_family,
            use_inverse=True,
            use_data_properties=True,
            use_nominals=False,
            use_card_restrictions=True,
            verbose=0,
            kwargs_classifier={"max_depth": 5, "random_state": 42}
        )
        
        self.assertEqual(model.use_inverse, True)
        self.assertEqual(model.use_data_properties, True)
        self.assertEqual(model.use_nominals, False)
        self.assertEqual(model.use_card_restrictions, True)
        self.assertEqual(model.verbose, 0)
        self.assertEqual(model.kwargs_classifier["max_depth"], 5)
        self.assertEqual(model.kwargs_classifier["random_state"], 42)


class TestTDLFilterLogic(unittest.TestCase):
    """Test the filtering logic for OWL expressions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        cls.NS = "http://www.benchmark.org/family#"

    def test_should_include_expression_nominals_enabled(self):
        """Test expression filtering when nominals are enabled."""
        model = TDL(knowledge_base=self.kb, use_nominals=True, verbose=0)
        
        # Create a nominal expression
        ind = OWLNamedIndividual(IRI.create(self.NS + "markus"))
        nominal_expr = OWLObjectOneOf([ind])
        
        # Should be included when use_nominals=True
        self.assertTrue(model._should_include_expression(nominal_expr))

    def test_should_include_expression_nominals_disabled(self):
        """Test expression filtering when nominals are disabled."""
        model = TDL(knowledge_base=self.kb, use_nominals=False, verbose=0)
        
        # Create a nominal expression
        ind = OWLNamedIndividual(IRI.create(self.NS + "markus"))
        nominal_expr = OWLObjectOneOf([ind])
        
        # Should be excluded when use_nominals=False
        self.assertFalse(model._should_include_expression(nominal_expr))

    def test_should_include_expression_cardinality_enabled(self):
        """Test expression filtering when cardinality restrictions are enabled."""
        model = TDL(knowledge_base=self.kb, use_card_restrictions=True, verbose=0)
        
        # Create a cardinality expression
        prop = OWLObjectProperty(IRI.create(self.NS + "hasChild"))
        filler = OWLClass(IRI.create(self.NS + "Person"))
        card_expr = OWLObjectMinCardinality(cardinality=2, property=prop, filler=filler)
        
        # Should be included when use_card_restrictions=True
        self.assertTrue(model._should_include_expression(card_expr))

    def test_should_include_expression_cardinality_disabled(self):
        """Test expression filtering when cardinality restrictions are disabled."""
        model = TDL(knowledge_base=self.kb, use_card_restrictions=False, verbose=0)
        
        # Create a cardinality expression
        prop = OWLObjectProperty(IRI.create(self.NS + "hasChild"))
        filler = OWLClass(IRI.create(self.NS + "Person"))
        card_expr = OWLObjectMinCardinality(cardinality=2, property=prop, filler=filler)
        
        # Should be excluded when use_card_restrictions=False
        self.assertFalse(model._should_include_expression(card_expr))

    def test_should_include_expression_data_properties_enabled(self):
        """Test expression filtering when data properties are enabled."""
        model = TDL(knowledge_base=self.kb, use_data_properties=True, verbose=0)
        
        # Create a data property expression (if available in the KB)
        # For this test, we'll create a mock expression
        # In a real scenario, this would be extracted from the KB
        from owlapy.owl_datatype import OWLDatatype
        
        
        data_prop = OWLDataProperty(IRI.create(self.NS + "hasAge"))
        data_range = OWLDatatype(IRI("http://www.w3.org/2001/XMLSchema#", "integer"))
        data_expr = OWLDataSomeValuesFrom(property=data_prop, filler=data_range)
        
        # Should be included when use_data_properties=True
        self.assertTrue(model._should_include_expression(data_expr))

    def test_should_include_expression_data_properties_disabled(self):
        """Test expression filtering when data properties are disabled."""
        model = TDL(knowledge_base=self.kb, use_data_properties=False, verbose=0)
        
        # Create a data property expression
        from owlapy.owl_datatype import OWLDatatype
        from owlapy.iri import IRI as OWL_IRI
        
        data_prop = OWLDataProperty(IRI.create(self.NS + "hasAge"))
        data_range = OWLDatatype(OWL_IRI("http://www.w3.org/2001/XMLSchema#", "integer"))
        data_expr = OWLDataSomeValuesFrom(property=data_prop, filler=data_range)
        
        # Should be excluded when use_data_properties=False
        self.assertFalse(model._should_include_expression(data_expr))

    def test_should_include_regular_class(self):
        """Test that regular OWL classes are always included."""
        model = TDL(knowledge_base=self.kb, use_nominals=False, verbose=0)
        
        # Regular OWL class should always be included
        regular_class = OWLClass(IRI.create(self.NS + "Person"))
        self.assertTrue(model._should_include_expression(regular_class))


class TestTDLRecursiveChecking(unittest.TestCase):
    """Test recursive checking of forbidden constructs in nested expressions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.NS = "http://www.benchmark.org/family#"

    def test_contains_nominal_in_intersection(self):
        """Test that nominals are detected when nested in intersections."""
        from ontolearn.learners.tree_learner import contains_nominal
        
        ind = OWLNamedIndividual(IRI.create(self.NS + "markus"))
        nominal = OWLObjectOneOf([ind])
        cls = OWLClass(IRI.create(self.NS + "Person"))
        
        # Create intersection containing nominal
        intersection = OWLObjectIntersectionOf([cls, nominal])
        
        self.assertTrue(contains_nominal(intersection))

    def test_contains_nominal_in_union(self):
        """Test that nominals are detected when nested in unions."""
        from ontolearn.learners.tree_learner import contains_nominal
        
        ind = OWLNamedIndividual(IRI.create(self.NS + "markus"))
        nominal = OWLObjectOneOf([ind])
        cls = OWLClass(IRI.create(self.NS + "Person"))
        
        # Create union containing nominal
        union = OWLObjectUnionOf([cls, nominal])
        
        self.assertTrue(contains_nominal(union))

    def test_contains_nominal_in_existential_filler(self):
        """Test that nominals are detected in existential restriction fillers."""
        from ontolearn.learners.tree_learner import contains_nominal
        
        ind = OWLNamedIndividual(IRI.create(self.NS + "markus"))
        nominal = OWLObjectOneOf([ind])
        prop = OWLObjectProperty(IRI.create(self.NS + "hasParent"))
        
        # Create existential restriction with nominal as filler
        exists = OWLObjectSomeValuesFrom(prop, nominal)
        
        self.assertTrue(contains_nominal(exists))

    def test_contains_cardinality_in_intersection(self):
        """Test that cardinality restrictions are detected when nested in intersections."""
        from ontolearn.learners.tree_learner import contains_cardinality
        
        prop = OWLObjectProperty(IRI.create(self.NS + "hasChild"))
        cls = OWLClass(IRI.create(self.NS + "Person"))
        filler = OWLClass(IRI.create(self.NS + "Person"))
        
        # Create cardinality restriction
        card = OWLObjectMinCardinality(cardinality=2, property=prop, filler=filler)
        
        # Create intersection containing cardinality
        intersection = OWLObjectIntersectionOf([cls, card])
        
        self.assertTrue(contains_cardinality(intersection))

    def test_contains_cardinality_in_union(self):
        """Test that cardinality restrictions are detected when nested in unions."""
        from ontolearn.learners.tree_learner import contains_cardinality
        
        prop = OWLObjectProperty(IRI.create(self.NS + "hasChild"))
        cls = OWLClass(IRI.create(self.NS + "Person"))
        filler = OWLClass(IRI.create(self.NS + "Person"))
        
        # Create cardinality restriction
        card = OWLObjectMinCardinality(cardinality=2, property=prop, filler=filler)
        
        # Create union containing cardinality
        union = OWLObjectUnionOf([cls, card])
        
        self.assertTrue(contains_cardinality(union))

    def test_contains_data_property_in_intersection(self):
        """Test that data properties are detected when nested in intersections."""
        from ontolearn.learners.tree_learner import contains_data_property
        from owlapy.owl_datatype import OWLDatatype
        
        data_prop = OWLDataProperty(IRI.create(self.NS + "hasAge"))
        data_range = OWLDatatype(IRI("http://www.w3.org/2001/XMLSchema#", "integer"))
        cls = OWLClass(IRI.create(self.NS + "Person"))
        
        # Create data property expression
        data_expr = OWLDataSomeValuesFrom(property=data_prop, filler=data_range)
        
        # Create intersection containing data property
        intersection = OWLObjectIntersectionOf([cls, data_expr])
        
        self.assertTrue(contains_data_property(intersection))

    def test_no_false_positives_for_clean_alc(self):
        """Test that clean ALC expressions don't trigger false positives."""
        from ontolearn.learners.tree_learner import contains_nominal, contains_cardinality, contains_data_property
        
        cls = OWLClass(IRI.create(self.NS + "Person"))
        prop = OWLObjectProperty(IRI.create(self.NS + "hasParent"))
        
        # Create clean ALC expression: Person ⊓ ∃hasParent.Person
        exists = OWLObjectSomeValuesFrom(prop, cls)
        intersection = OWLObjectIntersectionOf([cls, exists])
        
        # Should not contain any forbidden constructs
        self.assertFalse(contains_nominal(intersection))
        self.assertFalse(contains_cardinality(intersection))
        self.assertFalse(contains_data_property(intersection))

    def test_deeply_nested_nominal(self):
        """Test detection of deeply nested nominals."""
        from ontolearn.learners.tree_learner import contains_nominal
        
        ind = OWLNamedIndividual(IRI.create(self.NS + "markus"))
        nominal = OWLObjectOneOf([ind])
        cls = OWLClass(IRI.create(self.NS + "Person"))
        prop = OWLObjectProperty(IRI.create(self.NS + "hasParent"))
        
        # Create deeply nested structure:
        # Person ⊓ (∃hasParent.Person ⊔ {markus})
        exists = OWLObjectSomeValuesFrom(prop, cls)
        union = OWLObjectUnionOf([exists, nominal])
        intersection = OWLObjectIntersectionOf([cls, union])
        
        self.assertTrue(contains_nominal(intersection))

    def test_should_include_with_nested_nominal(self):
        """Test that _should_include_expression rejects expressions with nested nominals."""
        # Create a minimal mock KB for this test
        try:
            kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        except FileNotFoundError:
            # Skip test if KB file is not available
            self.skipTest("Knowledge base file not available")
            return
        
        model = TDL(knowledge_base=kb, use_nominals=False, verbose=0)
        
        ind = OWLNamedIndividual(IRI.create(self.NS + "markus"))
        nominal = OWLObjectOneOf([ind])
        cls = OWLClass(IRI.create(self.NS + "Person"))
        
        # Create intersection with nested nominal
        intersection = OWLObjectIntersectionOf([cls, nominal])
        
        # Should be excluded when use_nominals=False
        self.assertFalse(model._should_include_expression(intersection))

    def test_should_include_with_nested_cardinality(self):
        """Test that _should_include_expression rejects expressions with nested cardinality."""
        # Create a minimal mock KB for this test
        try:
            kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        except FileNotFoundError:
            # Skip test if KB file is not available
            self.skipTest("Knowledge base file not available")
            return
            
        model = TDL(knowledge_base=kb, use_card_restrictions=False, verbose=0)
        
        prop = OWLObjectProperty(IRI.create(self.NS + "hasChild"))
        cls = OWLClass(IRI.create(self.NS + "Person"))
        filler = OWLClass(IRI.create(self.NS + "Person"))
        
        # Create cardinality restriction
        card = OWLObjectMinCardinality(cardinality=2, property=prop, filler=filler)
        
        # Create intersection with nested cardinality
        intersection = OWLObjectIntersectionOf([cls, card])
        
        # Should be excluded when use_card_restrictions=False
        self.assertFalse(model._should_include_expression(intersection))


class TestTDLFeatureExtraction(unittest.TestCase):
    """Test feature extraction methods."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        with open("LPs/Family/lps.json") as json_file:
            cls.family_lps = json.load(json_file)

    def test_extract_expressions_basic(self):
        """Test basic expression extraction from individuals."""
        model = TDL(knowledge_base=self.kb, verbose=0)
        
        # Get a small learning problem
        examples = self.family_lps['problems']['Brother']
        p = set(examples['positive_examples'][:3])  # Use only 3 examples for speed
        typed_pos = list(map(OWLNamedIndividual, map(IRI.create, p)))
        
        X, features = model.extract_expressions_from_owl_individuals(typed_pos)
        
        # Check that we extracted features
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        self.assertEqual(X.shape[0], len(typed_pos))
        self.assertEqual(X.shape[1], len(features))
        
        # Check that X is binary
        self.assertTrue(np.all(np.isin(X, [0.0, 1.0])))

    def test_add_feature_method(self):
        """Test the _add_feature helper method."""
        model = TDL(knowledge_base=self.kb, verbose=0)
        
        features = {}
        individuals_to_feature_mapping = {}
        
        # Create a test expression
        NS = "http://www.benchmark.org/family#"
        test_class = OWLClass(IRI.create(NS + "Male"))
        test_individual = OWLNamedIndividual(IRI.create(NS + "markus"))
        
        # Add the feature
        model._add_feature(test_class, test_individual, features, individuals_to_feature_mapping)
        
        # Check that the feature was added
        self.assertGreater(len(features), 0)
        self.assertIn(test_individual.str, individuals_to_feature_mapping)
        self.assertGreater(len(individuals_to_feature_mapping[test_individual.str]), 0)

    def test_extract_expressions_with_inverse(self):
        """Test expression extraction with inverse properties enabled."""
        model = TDL(knowledge_base=self.kb, use_inverse=True, verbose=0)
        
        examples = self.family_lps['problems']['Brother']
        p = set(examples['positive_examples'][:2])  # Use only 2 examples for speed
        typed_pos = list(map(OWLNamedIndividual, map(IRI.create, p)))
        
        X, features = model.extract_expressions_from_owl_individuals(typed_pos)
        
        # With inverse properties, we should have more features
        self.assertGreater(len(features), 0)
        self.assertEqual(X.shape[0], len(typed_pos))


class TestTDLTraining(unittest.TestCase):
    """Test TDL training and prediction."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        with open("LPs/Family/lps.json") as json_file:
            cls.family_lps = json.load(json_file)

    def test_create_training_data(self):
        """Test training data creation."""
        model = TDL(knowledge_base=self.kb, verbose=0, kwargs_classifier={"random_state": 42})
        
        # Create a learning problem
        examples = self.family_lps['problems']['Brother']
        p = set(examples['positive_examples'][:5])  # Reduced to 5 for speed
        n = set(examples['negative_examples'][:5])  # Reduced to 5 for speed
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        X, y = model.create_training_data(lp)
        
        # Check the structure of training data
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.DataFrame)
        self.assertEqual(len(X), len(p) + len(n))
        self.assertEqual(len(y), len(p) + len(n))
        self.assertEqual(y.columns[0], "label")
        
        # Check labels
        self.assertEqual(y['label'].sum(), len(p))  # Sum of 1s should equal positive examples

    def test_fit_basic(self):
        """Test basic fitting of TDL model."""
        model = TDL(knowledge_base=self.kb, verbose=0, kwargs_classifier={"random_state": 42})
        
        # Create a simple learning problem with reduced examples for speed
        examples = self.family_lps['problems']['Brother']
        p = set(examples['positive_examples'][:8])  # Reduced for speed
        n = set(examples['negative_examples'][:8])  # Reduced for speed
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        # Fit the model
        model.fit(lp)
        
        # Check that model was trained
        self.assertIsNotNone(model.clf)
        self.assertIsNotNone(model.features)
        self.assertIsNotNone(model.conjunctive_concepts)
        self.assertIsNotNone(model.disjunction_of_conjunctive_concepts)
        self.assertIsNotNone(model.X)
        self.assertIsNotNone(model.y)

    def test_best_hypotheses(self):
        """Test retrieval of best hypotheses."""
        model = TDL(knowledge_base=self.kb, verbose=0, kwargs_classifier={"random_state": 42})
        
        examples = self.family_lps['problems']['Sister']
        p = set(examples['positive_examples'][:8])  # Reduced for speed
        n = set(examples['negative_examples'][:8])  # Reduced for speed
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        model.fit(lp)
        
        # Get best hypothesis
        h = model.best_hypotheses(n=1)
        self.assertIsNotNone(h)
        self.assertIsInstance(h, (OWLObjectUnionOf, OWLObjectIntersectionOf, OWLClass, 
                                   OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom))

    def test_fit_with_different_configurations(self):
        """Test fitting with different feature configurations."""
        examples = self.family_lps['problems']['Daughter']
        p = set(examples['positive_examples'][:6])  # Reduced for speed
        n = set(examples['negative_examples'][:6])  # Reduced for speed
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        # Test with nominals disabled
        model1 = TDL(knowledge_base=self.kb, use_nominals=False, verbose=0, 
                     kwargs_classifier={"random_state": 42})
        model1.fit(lp)
        self.assertIsNotNone(model1.clf)
        
        # Test with cardinality restrictions enabled
        model2 = TDL(knowledge_base=self.kb, use_card_restrictions=True, verbose=0,
                     kwargs_classifier={"random_state": 42})
        model2.fit(lp)
        self.assertIsNotNone(model2.clf)

    def test_classification_report(self):
        """Test that classification report is generated."""
        model = TDL(knowledge_base=self.kb, report_classification=True, verbose=0,
                    kwargs_classifier={"random_state": 42})
        
        examples = self.family_lps['problems']['Brother']
        p = set(examples['positive_examples'][:5])  # Reduced for speed
        n = set(examples['negative_examples'][:5])  # Reduced for speed
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        model.fit(lp)
        
        # Check that report was generated
        report = model.classification_report
        self.assertIsNotNone(report)
        self.assertIsInstance(report, str)
        self.assertIn("Classification Report", report)


class TestTDLPerformance(unittest.TestCase):
    """Test TDL performance on known problems."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        with open("LPs/Family/lps.json") as json_file:
            cls.family_lps = json.load(json_file)

    def test_performance_brother(self):
        """Test TDL performance on Brother concept."""
        model = TDL(knowledge_base=self.kb, verbose=0, kwargs_classifier={"random_state": 1})
        
        examples = self.family_lps['problems']['Brother']
        # Use subset for faster testing while maintaining concept learning ability
        p = set(examples['positive_examples'][:12])
        n = set(examples['negative_examples'][:12])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        model.fit(lp)
        h = model.best_hypotheses()
        
        # Compute F1 score
        f1 = compute_f1_score(
            individuals=frozenset({i for i in self.kb.individuals(h)}),
            pos=lp.pos,
            neg=lp.neg
        )
        
        # Brother should be learned well even with subset
        self.assertGreaterEqual(f1, 0.85)

    def test_performance_sister(self):
        """Test TDL performance on Sister concept."""
        model = TDL(knowledge_base=self.kb, verbose=0, kwargs_classifier={"random_state": 1})
        
        examples = self.family_lps['problems']['Sister']
        # Use subset for faster testing while maintaining concept learning ability
        p = set(examples['positive_examples'][:12])
        n = set(examples['negative_examples'][:12])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        model.fit(lp)
        h = model.best_hypotheses()
        
        # Compute F1 score
        f1 = compute_f1_score(
            individuals=frozenset({i for i in self.kb.individuals(h)}),
            pos=lp.pos,
            neg=lp.neg
        )
        
        # Sister should be learned well even with subset
        self.assertGreaterEqual(f1, 0.85)


if __name__ == '__main__':
    unittest.main()
