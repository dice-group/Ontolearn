"""
Simple test for NERO learner to verify basic functionality.
"""

import sys
sys.path.append('/')

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.learners.nero import NERO
from owlapy.owl_individual import OWLNamedIndividual, IRI


def test_nero_basic():
    """Test basic NERO functionality."""
    
    print("=" * 80)
    print("Testing NERO - Neural Class Expression Learning")
    print("=" * 80)
    
    try:
        # Load knowledge base
        print("\n1. Loading knowledge base...")
        kb = KnowledgeBase(path="../KGs/Family/family-benchmark_rich_background.owl")
        print(f"   ✓ Knowledge base loaded successfully")
        print(f"   - Number of individuals: {kb.individuals_count()}")
        
        # Define a simple learning problem
        print("\n2. Defining learning problem...")
        namespace = "http://www.benchmark.org/family#"
        
        pos = {OWLNamedIndividual(IRI.create(namespace, "F2F13")),
               OWLNamedIndividual(IRI.create(namespace, "F2M7")),
               OWLNamedIndividual(IRI.create(namespace, "F2M8"))}
        
        neg = {OWLNamedIndividual(IRI.create(namespace, "F1M1")),
               OWLNamedIndividual(IRI.create(namespace, "F1F2")),
               OWLNamedIndividual(IRI.create(namespace, "F2M5"))}
        
        lp = PosNegLPStandard(pos=pos, neg=neg)
        print(f"   ✓ Learning problem created")
        print(f"   - Positive examples: {len(pos)}")
        print(f"   - Negative examples: {len(neg)}")
        
        # Initialize NERO
        print("\n3. Initializing NERO with DeepSet architecture...")
        nero = NERO(
            knowledge_base=kb,
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


if __name__ == '__main__':
    success = test_nero_basic()
    sys.exit(0 if success else 1)

