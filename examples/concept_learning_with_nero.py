"""
Example: Concept Learning with NERO

This example demonstrates how to use NERO (Neural Class Expression Learning)
for learning OWL class expressions from positive and negative examples.
"""
import json

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.learners.nero import NERO
from owlapy.owl_individual import OWLNamedIndividual, IRI


def example_nero_family():
    """Example using NERO on the Family benchmark."""
    
    # Load knowledge base
    kb = KnowledgeBase(path="../KGs/Family/family-benchmark_rich_background.owl")

    # Get positive and negative examples from JSON
    with open('../LPs/Family/lps.json') as json_file:
        settings = json.load(json_file)

    # Create learning problem
    p = set(settings['problems']['Aunt']['positive_examples'])
    n = set(settings['problems']['Aunt']['negative_examples'])
    pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=pos, neg=neg)
    
    print("=" * 80)
    print("NERO - Neural Class Expression Learning")
    print("=" * 80)
    print(f"Knowledge Base: {kb}")
    print(f"Positive examples: {len(pos)}")
    print(f"Negative examples: {len(neg)}")
    print()
    
    # Initialize NERO with DeepSet architecture
    print("Initializing NERO with DeepSet architecture...")
    nero_deepset = NERO(
        knowledge_base=kb,
        num_embedding_dim=50,
        neural_architecture='DeepSet',
        learning_rate=0.001,
        num_epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Train NERO on the learning problem
    print("\nTraining NERO...")
    nero_deepset.fit(lp)
    
    # Make predictions
    print("\nMaking predictions...")
    
    result = nero_deepset.predict(pos=pos, neg=neg, top_k=10)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Predicted Concept: {result['Prediction']}")
    print(f"F-measure: {result['F-measure']:.3f}")
    print(f"Runtime: {result['Runtime']:.3f} seconds")
    print("=" * 80)
    

def example_nero_set_transformer():
    """Example using NERO with SetTransformer architecture."""
    
    # Load knowledge base
    kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
    
    # Define learning problem
    namespace = "http://www.benchmark.org/family#"
    
    pos = {OWLNamedIndividual(IRI.create(namespace, "F2F13")),
           OWLNamedIndividual(IRI.create(namespace, "F2M7")),
           OWLNamedIndividual(IRI.create(namespace, "F2M8"))}
    
    neg = {OWLNamedIndividual(IRI.create(namespace, "F1M1")),
           OWLNamedIndividual(IRI.create(namespace, "F1F2")),
           OWLNamedIndividual(IRI.create(namespace, "F2M5"))}
    
    lp = PosNegLPStandard(pos=pos, neg=neg)
    
    print("\n" + "=" * 80)
    print("NERO - Neural Class Expression Learning (SetTransformer)")
    print("=" * 80)
    
    # Initialize NERO with SetTransformer architecture
    print("Initializing NERO with SetTransformer architecture...")
    nero_st = NERO(
        knowledge_base=kb,
        num_embedding_dim=50,
        neural_architecture='SetTransformer',
        learning_rate=0.001,
        num_epochs=30,
        batch_size=16,
        verbose=1
    )
    
    # Train and predict
    print("\nTraining NERO...")
    nero_st.fit(lp)

    result = nero_st.predict(pos=pos, neg=neg, top_k=5)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Predicted Concept: {result['Prediction']}")
    print(f"F-measure: {result['F-measure']:.3f}")
    print(f"Runtime: {result['Runtime']:.3f} seconds")
    print("=" * 80)


if __name__ == '__main__':
    # Run DeepSet example
    example_nero_family()
    
    # Run SetTransformer example
    # example_nero_set_transformer()

