"""
Example script demonstrating the use of the ALCSAT learner.

This example shows how to use the ALCSAT SAT-based learner to find
ALC concept expressions that fit positive and negative examples.
"""
import json

from ontolearn.learners import ALCSAT
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer


def alcsat_example():
    """
    Run a simple example using the ALCSAT learner on the family ontology.
    """
    
    # () Load knowledge base
    kb = KnowledgeBase(path="../KGs/Family/family-benchmark_rich_background.owl")


    # () Get positive and negative examples
    namespace = "http://www.benchmark.org/family#"
    with open('../LPs/Family/lps.json') as json_file:
        settings = json.load(json_file)

    # () Create learning problem
    p = set(settings['problems']['Uncle']['positive_examples'])
    n = set(settings['problems']['Uncle']['negative_examples'])
    pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
    neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
    lp = PosNegLPStandard(pos=pos, neg=neg)
    
    # () Initialize ALCSAT learner
    print("Initializing ALCSAT learner...")
    model = ALCSAT(
        knowledge_base=kb,
        max_concept_size=10,      # Maximum concept tree depth
        start_concept_size=1,     # Start with small concepts
        max_runtime=60,               # 60 second timeout
    )
    
    # () Run the learner
    print("Running ALCSAT to find concept expressions...")
    print(f"Positive examples: {len(pos)}")
    print(f"Negative examples: {len(neg)}")
    print("-" * 60)
    
    model.fit(lp)
    
    # () Get and display results
    print("\nResults:")
    print("=" * 60)
    
    renderer = DLSyntaxObjectRenderer()
    hypothesis = model.best_hypothesis()
    
    if hypothesis:
        print(f"\nLearned concept:")
        print(f"   {renderer.render(hypothesis)}")

        quality_results = model.best_hypothesis_accuracy()
        print(f"   Accuracy: {quality_results:.3f}")
    else:
        print("No hypothesis found.")
    
    print("\n" + "=" * 60)
    print(f"Total runtime: {model.start_time and __import__('time').time() - model.start_time:.2f} seconds")
    
    return model


if __name__ == "__main__":
    print("ALCSAT Learner Example")
    print("=" * 60)
    print()
    alcsat_example()
