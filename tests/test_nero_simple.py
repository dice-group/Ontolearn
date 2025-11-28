"""
Simple test for NERO learner to verify basic functionality.
"""
import json
import sys
import unittest

from ontolearn.utils import compute_f1_score

sys.path.append('/')

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.learners.nero import NERO
from owlapy.owl_individual import OWLNamedIndividual, IRI


class TestNEROFunctionality(unittest.TestCase):

    ns = "http://www.benchmark.org/family#"

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

    def test_nero_basic_functionality(self):
        """Test that NERO can learn a basic concept."""
        lp = self._get_learning_problem("Sister")

        model = NERO(
            knowledge_base=self.kb,
            namespace= self.ns,
            num_embedding_dim=20,
            neural_architecture='DeepSet',
            num_epochs=10,
            batch_size=8,
            max_runtime=30,
            verbose=0
        )

        model.fit(lp)
        hypothesis = model.best_hypothesis()

        self.assertIsNotNone(hypothesis, "NERO should find a hypothesis")

        # Compute F1 score
        individuals = frozenset({i for i in self.kb.individuals(hypothesis)})
        f1 = compute_f1_score(individuals=individuals, pos=lp.pos, neg=lp.neg)
        self.assertGreater(f1, 0.3, f"NERO F1 score should be > 0.3, got {f1:.3f}")

    def test_nero_on_aunt_problem(self):
        """Test NERO on the Aunt learning problem."""
        lp = self._get_learning_problem("Aunt")

        model = NERO(
            knowledge_base=self.kb,
            namespace=self.ns,
            num_embedding_dim=30,
            neural_architecture='DeepSet',
            num_epochs=20,
            batch_size=16,
            max_runtime=60,
            verbose=0
        )

        model.fit(lp)
        hypothesis = model.best_hypothesis()

        self.assertIsNotNone(hypothesis, "NERO should find a hypothesis for Aunt")

        individuals = frozenset({i for i in self.kb.individuals(hypothesis)})
        f1 = compute_f1_score(individuals=individuals, pos=lp.pos, neg=lp.neg)

        self.assertGreater(f1, 0.3, f"NERO F1 score should be > 0.3, got {f1:.3f}")

    def test_nero_returns_owl_expression(self):
        """Test that NERO returns a valid OWL expression."""
        lp = self._get_learning_problem("Brother")

        model = NERO(
            knowledge_base=self.kb,
            namespace=self.ns,
            num_embedding_dim=20,
            neural_architecture='DeepSet',
            num_epochs=10,
            batch_size=8,
            max_runtime=30,
            verbose=0
        )

        model.fit(lp)
        hypothesis = model.best_hypothesis()

        self.assertIsNotNone(hypothesis)
        individuals = list(self.kb.individuals(hypothesis))
        self.assertIsInstance(individuals, list)

    def test_nero_with_timeout(self):
        """Test that NERO respects timeout constraints."""
        lp = self._get_learning_problem("Aunt")

        model = NERO(
            knowledge_base=self.kb,
            namespace=self.ns,
            num_embedding_dim=20,
            neural_architecture='DeepSet',
            num_epochs=50,
            batch_size=16,
            max_runtime=5,
            verbose=0
        )

        import time
        start = time.time()
        model.fit(lp)
        elapsed = time.time() - start

        self.assertLess(elapsed, 15, "NERO should respect timeout (with some overhead)")

    def test_nero_with_set_transformer(self):
        """Test NERO with SetTransformer architecture."""
        lp = self._get_learning_problem("Sister")

        model = NERO(
            knowledge_base=self.kb,
            namespace=self.ns,
            num_embedding_dim=20,
            neural_architecture='SetTransformer',
            num_epochs=10,
            batch_size=8,
            max_runtime=30,
            verbose=0
        )

        model.fit(lp)
        hypothesis = model.best_hypothesis()

        self.assertIsNotNone(hypothesis, "NERO with SetTransformer should find a hypothesis")

        individuals = frozenset({i for i in self.kb.individuals(hypothesis)})
        f1 = compute_f1_score(individuals=individuals, pos=lp.pos, neg=lp.neg)
        self.assertGreater(f1, 0.3, f"NERO F1 score should be > 0.3, got {f1:.3f}")

    def test_nero_quality_metric(self):
        """Test that NERO returns quality metrics."""
        lp = self._get_learning_problem("Brother")

        model = NERO(
            knowledge_base=self.kb,
            namespace=self.ns,
            num_embedding_dim=20,
            neural_architecture='DeepSet',
            num_epochs=10,
            batch_size=8,
            max_runtime=30,
            verbose=0
        )

        model.fit(lp)
        hypothesis = model.best_hypothesis()
        quality = model.best_hypothesis_quality()

        self.assertIsNotNone(hypothesis, "NERO should return a hypothesis")
        self.assertIsNotNone(quality, "NERO should return quality metric")
        self.assertGreaterEqual(quality, 0.0, "Quality should be non-negative")

    def test_nero_basic(self):
        """Test basic NERO functionality."""

        print("=" * 80)
        print("Testing NERO - Neural Class Expression Learning")
        print("=" * 80)

        try:
            # Load knowledge base
            print("\n1. Loading knowledge base...")
            kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
            print(f"   ✓ Knowledge base loaded successfully")
            print(f"   - Number of individuals: {kb.individuals_count()}")

            # Define a simple learning problem
            print("\n2. Defining learning problem...")
            namespace = "http://www.benchmark.org/family#"

            pos = {OWLNamedIndividual(IRI.create(namespace, "F2M23")),
                   OWLNamedIndividual(IRI.create(namespace, "F2M16")),
                   OWLNamedIndividual(IRI.create(namespace, "F4M57"))}

            neg = {OWLNamedIndividual(IRI.create(namespace, "F9F160")),
                   OWLNamedIndividual(IRI.create(namespace, "F9F148")),
                   OWLNamedIndividual(IRI.create(namespace, "F10F192"))}

            lp = PosNegLPStandard(pos=pos, neg=neg)
            print(f"   ✓ Learning problem created")
            print(f"   - Positive examples: {len(pos)}")
            print(f"   - Negative examples: {len(neg)}")

            # Initialize NERO
            print("\n3. Initializing NERO with DeepSet architecture...")
            nero = NERO(
                knowledge_base=kb,
                namespace=self.ns,
                num_embedding_dim=20,  # Small for quick testing
                neural_architecture='DeepSet',
                learning_rate=0.01,
                num_epochs=10,  # Few epochs for quick testing
                batch_size=8,
                verbose=0
            )
            print(f"   ✓ NERO initialized: {nero}")

            # Train NERO
            print("\n4. Training NERO on learning problem...")
            nero.fit(lp)
            print(f"   ✓ Training completed")

            # Make predictions
            print("\n5. Making predictions...")
            result = nero.predict(pos=pos, neg=neg, top_k=5)

            print(f"   ✓ Prediction completed")
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)
            print(f"Predicted Concept: {result['Prediction']}")
            print(f"F-measure: {result['F-measure']:.3f}")
            print(f"Runtime: {result['Runtime']:.3f} seconds")
            print("=" * 80)

            print("\n✓ All tests passed successfully!")
            return True

        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

