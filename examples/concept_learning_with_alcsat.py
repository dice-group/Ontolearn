"""
Example script demonstrating the use of the ALCSAT learner.

This example shows how to use the ALCSAT SAT-based learner to find
ALC concept expressions that fit positive and negative examples.
"""
import json
import os

from ontolearn.learners import ALCSAT
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer


def alcsat_example():
    """
    Run ALCSAT learner example on the family ontology.
    """
    print("=" * 60)
    print("ALCSAT Learner Example")
    print("=" * 60)

    # Load knowledge base
    kb = KnowledgeBase(path=os.path.join(os.path.dirname(__file__), '..', 'KGs', 'Family', 'family-benchmark_rich_background.owl'))

    # Get positive and negative examples from JSON
    with open(os.path.join(os.path.dirname(__file__), '..', 'LPs', 'Family', 'lps.json')) as json_file:
        settings = json.load(json_file)

    # Create learning problem
    p = set(settings['problems']['Aunt']['positive_examples'])
    n = set(settings['problems']['Aunt']['negative_examples'])
    pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=pos, neg=neg)

    # Initialize ALCSAT learner
    print("\nInitializing ALCSAT learner...")
    model = ALCSAT(
        knowledge_base=kb,
        max_concept_size=10,      # Maximum concept tree depth
        start_concept_size=1,     # Start with small concepts
        max_runtime=60,           # 60 second timeout
    )

    # Run the learner
    print("Running ALCSAT to find concept expressions...")
    print(f"Positive examples: {len(pos)}")
    print(f"Negative examples: {len(neg)}")
    print("-" * 60)

    model.fit(lp)

    # Get and display results
    print("\nResults:")
    print("=" * 60)

    renderer = DLSyntaxObjectRenderer()
    hypothesis = model.best_hypothesis()

    if hypothesis:
        print(f"\nLearned concept:")
        print(f"   {renderer.render(hypothesis)}")

        quality_results = model.best_hypothesis_accuracy()
        if quality_results is not None:
            print(f"   Accuracy: {quality_results:.3f}")
    else:
        print("No hypothesis found.")

    print("\n" + "=" * 60)
    if model.start_time:
        import time
        print(f"Total runtime: {time.time() - model.start_time:.2f} seconds")

    return model


if __name__ == "__main__":
    try:
        alcsat_example()
    except Exception as e:
        print(f"\nALCSAT example failed: {e}")
        import traceback
        traceback.print_exc()
