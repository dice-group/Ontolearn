"""
Example script demonstrating the use of ALCSAT and SPELL learners.

This example shows how to use both SAT-based learners to find
concept expressions that fit positive and negative examples.
"""
import json

from ontolearn.learners import ALCSAT, SPELL
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer


def spell_example():
    """
    Run SPELL learner example on the family ontology.
    """
    print("\n\n" + "=" * 89)
    print("SPELL Learner Example")
    print("=" * 89)

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

    print("Running SPELL for each searching mode to find class expressions with the best accuracy...")
    print(f"Positive examples: {len(pos)}")
    print(f"Negative examples: {len(neg)}")

    # Test different search modes
    for search_mode in ['exact', 'neg_approx', 'full_approx']:
        print("\n"+ "-" * 30 + f" Testing mode: {search_mode} " + "-" * 30)
        print()

        # Initialize the learner
        model = SPELL(
            knowledge_base=kb,
            max_query_size=10,        # Maximum query size
            starting_query_size=1,    # Start with small queries
            search_mode=search_mode,  # Search mode
            max_runtime=60,           # 60 second timeout
        )

        # Run the learner
        model.fit(lp)

        # Get and display results
        print("\nResults:")

        renderer = DLSyntaxObjectRenderer()
        hypothesis = model.best_hypothesis()

        if hypothesis:
            print(f"Learned concept using search_mode = '{search_mode}':")
            print(f"   {renderer.render(hypothesis)}")

            quality_results = model.best_hypothesis_accuracy()
            if quality_results is not None:
                print(f"   Accuracy: {quality_results:.3f}")
        else:
            print("No hypothesis found.")

        if model.start_time:
            import time
            print(f"Runtime: {time.time() - model.start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        spell_example()
    except Exception as e:
        print(f"\nSPELL example failed: {e}")
        import traceback
        traceback.print_exc()
