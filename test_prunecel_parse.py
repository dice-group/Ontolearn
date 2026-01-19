"""Quick test to see PruneCEL CSV output."""
from ontolearn.knowledge_base import KnowledgeBase
from prunecel_wrapper import PruneCELWrapper
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
import json
from owlapy import owl_expression_to_dl
from ontolearn.utils.static_funcs import compute_f1_score


# Load KB
kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")

# Load one learning problem
with open("LPs/Family/lps_difficult.json") as f:
    lps_data = json.load(f)
    lps = lps_data.get('problems', lps_data)

# Take first problem
lp_name, lp_data = list(lps.items())[6]
print(f"Testing with: {lp_name}")

pos = set([OWLNamedIndividual(IRI.create(x)) for x in lp_data['positive_examples']])
neg = set([OWLNamedIndividual(IRI.create(x)) for x in lp_data['negative_examples']])
lp = PosNegLPStandard(pos=pos, neg=neg)

# Initialize PruneCEL
prunecel = PruneCELWrapper(
    knowledge_base=kb,
    jar_path="PruneCEL/target/prune-cel-0.0.1-SNAPSHOT.jar",
    sparql_url="http://localhost:3030/family/sparql",
    max_runtime=10 # 10 seconds
)

# Fit
print("\nFitting PruneCEL...")
prunecel.fit(lp)
print("Done.")

# Get best hypothesis
best = prunecel.best_hypothesis()
# print(f"\nBest hypothesis type: {type(best)}")
owl_desbest = owl_expression_to_dl(best)
print(f"Best hypothesis: {owl_desbest}")

f1_prunecel = compute_f1_score(individuals=frozenset({i for i in kb.individuals(best)}),
                                          pos=lp.pos,
                                          neg=lp.neg)

print(f"F1 Score of PruneCEL hypothesis: {f1_prunecel:.3f}")

# Access the number of tested concepts
num_concepts = prunecel.number_of_tested_concepts
print(f"PruneCEL tested {num_concepts} concepts")

# Access the last runtime
last_runtime = prunecel.last_runtime
print(f"PruneCEL last runtime: {last_runtime:.2f} seconds")