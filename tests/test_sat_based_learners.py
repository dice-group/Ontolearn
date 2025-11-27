"""
Regression tests for SAT-based learners (ALCSAT and SPELL).

This module contains tests to ensure that ALCSAT and SPELL learners
work correctly on standard benchmark problems.
"""

import unittest
import json
from ontolearn.learners import ALCSAT, SPELL
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.utils.static_funcs import compute_f1_score


class TestSATBasedLearners(unittest.TestCase):
    """Test suite for SAT-based concept learners."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by multiple tests."""
        # Load the Family knowledge base once for all tests
        cls.kb_path = "KGs/Family/family-benchmark_rich_background.owl"
        cls.kb = KnowledgeBase(path=cls.kb_path)
        
        # Load learning problems
        with open("LPs/Family/lps.json") as json_file:
            cls.lps_data = json.load(json_file)

    def _get_learning_problem(self, problem_name):
        """Helper method to create a learning problem from test data."""
        examples = self.lps_data['problems'][problem_name]
        p = set(examples['positive_examples'])
        n = set(examples['negative_examples'])
        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
        return PosNegLPStandard(pos=typed_pos, neg=typed_neg)

    def test_alcsat_basic_functionality(self):
        """Test that ALCSAT can learn a basic concept."""
        # Test on a simpler problem - Sister
        lp = self._get_learning_problem("Sister")
        
        model = ALCSAT(
            knowledge_base=self.kb,
            max_concept_size=5,
            start_concept_size=1,
            max_runtime=30
        )
        
        # Fit the model
        model.fit(lp)
        
        # Check that a hypothesis was found
        hypothesis = model.best_hypothesis()
        self.assertIsNotNone(hypothesis, "ALCSAT should find a hypothesis")
        
        # Check that accuracy is reasonable
        accuracy = model.best_hypothesis_accuracy()
        self.assertIsNotNone(accuracy, "ALCSAT should return accuracy")
        self.assertGreater(accuracy, 0.5, "ALCSAT accuracy should be > 0.5")

    def test_alcsat_on_aunt_problem(self):
        """Test ALCSAT on the Aunt learning problem."""
        lp = self._get_learning_problem("Aunt")
        
        model = ALCSAT(
            knowledge_base=self.kb,
            max_concept_size=8,
            start_concept_size=1,
            max_runtime=60
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis, "ALCSAT should find a hypothesis for Aunt")
        
        # Compute F1 score
        individuals = frozenset({i for i in self.kb.individuals(hypothesis)})
        f1 = compute_f1_score(individuals=individuals, pos=lp.pos, neg=lp.neg)
        
        # ALCSAT should achieve reasonable performance
        self.assertGreater(f1, 0.6, f"ALCSAT F1 score should be > 0.6, got {f1:.3f}")

    def test_alcsat_returns_owl_expression(self):
        """Test that ALCSAT returns a valid OWL expression."""
        lp = self._get_learning_problem("Brother")
        
        model = ALCSAT(
            knowledge_base=self.kb,
            max_concept_size=5,
            max_runtime=30
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis)
        # Check that we can retrieve individuals using the hypothesis
        individuals = list(self.kb.individuals(hypothesis))
        self.assertIsInstance(individuals, list)

    def test_spell_exact_mode(self):
        """Test SPELL learner with exact search mode."""
        lp = self._get_learning_problem("Brother")
        
        model = SPELL(
            knowledge_base=self.kb,
            max_query_size=6,
            starting_query_size=1,
            search_mode='exact',
            max_runtime=30
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis, "SPELL should find a hypothesis in exact mode")
        
        accuracy = model.best_hypothesis_accuracy()
        self.assertIsNotNone(accuracy)
        self.assertGreater(accuracy, 0.5, "SPELL accuracy should be > 0.5")

    def test_spell_full_approx_mode(self):
        """Test SPELL learner with full approximation mode."""
        lp = self._get_learning_problem("Sister")
        
        model = SPELL(
            knowledge_base=self.kb,
            max_query_size=8,
            starting_query_size=1,
            search_mode='full_approx',
            max_runtime=30
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis, "SPELL should find a hypothesis in full_approx mode")
        
        # Compute F1 score
        individuals = frozenset({i for i in self.kb.individuals(hypothesis)})
        f1 = compute_f1_score(individuals=individuals, pos=lp.pos, neg=lp.neg)
        
        self.assertGreater(f1, 0.5, f"SPELL F1 score should be > 0.5, got {f1:.3f}")

    def test_spell_neg_approx_mode(self):
        """Test SPELL learner with negative approximation mode."""
        lp = self._get_learning_problem("Daughter")
        
        model = SPELL(
            knowledge_base=self.kb,
            max_query_size=6,
            starting_query_size=1,
            search_mode='neg_approx',
            max_runtime=30
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis, "SPELL should find a hypothesis in neg_approx mode")
        
        accuracy = model.best_hypothesis_accuracy()
        self.assertIsNotNone(accuracy)

    def test_spell_on_aunt_problem(self):
        """Test SPELL on the Aunt learning problem."""
        lp = self._get_learning_problem("Aunt")
        
        model = SPELL(
            knowledge_base=self.kb,
            max_query_size=10,
            starting_query_size=1,
            search_mode='full_approx',
            max_runtime=60
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis, "SPELL should find a hypothesis for Aunt")
        
        # Compute F1 score
        individuals = frozenset({i for i in self.kb.individuals(hypothesis)})
        f1 = compute_f1_score(individuals=individuals, pos=lp.pos, neg=lp.neg)
        
        # SPELL should achieve reasonable performance
        self.assertGreater(f1, 0.5, f"SPELL F1 score should be > 0.5, got {f1:.3f}")

    def test_alcsat_with_timeout(self):
        """Test that ALCSAT respects timeout constraints."""
        lp = self._get_learning_problem("Aunt")
        
        # Set a very short timeout
        model = ALCSAT(
            knowledge_base=self.kb,
            max_concept_size=20,  # Large size that might timeout
            start_concept_size=1,
            max_runtime=5  # Short timeout
        )
        
        import time
        start = time.time()
        model.fit(lp)
        elapsed = time.time() - start
        
        # Should complete within a reasonable time beyond the timeout
        self.assertLess(elapsed, 15, "ALCSAT should respect timeout (with some overhead)")

    def test_spell_with_timeout(self):
        """Test that SPELL respects timeout constraints."""
        lp = self._get_learning_problem("Aunt")
        
        # Set a very short timeout
        model = SPELL(
            knowledge_base=self.kb,
            max_query_size=20,  # Large size that might timeout
            starting_query_size=1,
            search_mode='full_approx',
            max_runtime=5  # Short timeout
        )
        
        import time
        start = time.time()
        model.fit(lp)
        elapsed = time.time() - start
        
        # Should complete within a reasonable time beyond the timeout
        self.assertLess(elapsed, 25, "SPELL should respect timeout (with some overhead)")

    def test_alcsat_incremental_search(self):
        """Test that ALCSAT performs incremental search correctly."""
        lp = self._get_learning_problem("Brother")
        
        # Start from size 3 instead of 1
        model = ALCSAT(
            knowledge_base=self.kb,
            max_concept_size=8,
            start_concept_size=3,
            max_runtime=30
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis)

    def test_spell_incremental_search(self):
        """Test that SPELL performs incremental search correctly."""
        lp = self._get_learning_problem("Sister")
        
        # Start from size 2 instead of 1
        model = SPELL(
            knowledge_base=self.kb,
            max_query_size=8,
            starting_query_size=2,
            search_mode='full_approx',
            max_runtime=30
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis)

    def test_alcsat_small_problem(self):
        """Test ALCSAT on a small, well-defined problem."""
        lp = self._get_learning_problem("Granddaughter")
        
        model = ALCSAT(
            knowledge_base=self.kb,
            max_concept_size=6,
            start_concept_size=1,
            max_runtime=30
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis)
        
        # Check F1 score
        individuals = frozenset({i for i in self.kb.individuals(hypothesis)})
        f1 = compute_f1_score(individuals=individuals, pos=lp.pos, neg=lp.neg)
        
        self.assertGreater(f1, 0.5)

    def test_spell_small_problem(self):
        """Test SPELL on a small, well-defined problem."""
        lp = self._get_learning_problem("Granddaughter")
        
        model = SPELL(
            knowledge_base=self.kb,
            max_query_size=8,
            starting_query_size=1,
            search_mode='full_approx',
            max_runtime=30
        )
        
        model.fit(lp)
        hypothesis = model.best_hypothesis()
        
        self.assertIsNotNone(hypothesis)
        
        # Check F1 score
        individuals = frozenset({i for i in self.kb.individuals(hypothesis)})
        f1 = compute_f1_score(individuals=individuals, pos=lp.pos, neg=lp.neg)
        
        self.assertGreater(f1, 0.5)


if __name__ == '__main__':
    unittest.main()
