import asyncio
import aiohttp
import time
import pandas as pd
from tqdm import tqdm
import random
import itertools
from argparse import ArgumentParser
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.utils import concept_reducer, concept_reducer_properties
from itertools import chain
from owlapy.class_expression import (
	OWLObjectUnionOf,
	OWLObjectIntersectionOf,
	OWLObjectSomeValuesFrom,
	OWLObjectAllValuesFrom,
	OWLObjectMinCardinality,
	OWLObjectMaxCardinality,
	OWLObjectOneOf,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_dl
from owlapy.iri import IRI

# Asynchronous function to query the neural reasoner API
async def async_neural_retrieval(session, url, expression, namespace):
	"""
	Asynchronously queries the neural reasoner API to retrieve individuals satisfying the given OWL expression.	
	"""
	payload = {'expression': owl_expression_to_dl(expression), 'namespace': namespace}
	async with session.post(url + '/predict', json=payload) as response:
		result = await response.json()
		return set(result['retrieval_result'])

def generate_concepts_from_kb(symbolic_kb: KnowledgeBase):
	"""
	Generates concepts from the knowledge base, including named concepts, negations, unions, intersections,
	existential restrictions, and other combinations as described.
	"""
	object_properties = {i for i in symbolic_kb.get_object_properties()}
	
	
	# (4) R⁻: Inverse of object properties.
	object_properties_inverse = {i.get_inverse_property() for i in object_properties}

	# (5) R*: R UNION R⁻.
	object_properties_and_inverse = object_properties.union(object_properties_inverse)
	# (6) NC: Named owl concepts.
	nc = {i for i in symbolic_kb.get_concepts()}
	# (7) NC⁻: Complement of NC.
	nnc = {i.get_object_complement_of() for i in nc}
	# (8) NC*: NC UNION NC⁻.
	nc_star = nc.union(nnc)
	# (9) Retrieve 10 random Nominals.
	nominals = symbolic_kb.all_individuals_set()
	# (10) All combinations of 3 for Nominals, e.g. {martin, heinz, markus}
	nominal_combinations = set( OWLObjectOneOf(combination)for combination in itertools.combinations(nominals, 3))
	# (13) NC* UNION NC*.
	unions_nc_star = concept_reducer(nc_star, opt=OWLObjectUnionOf)
	# (14) NC* INTERACTION NC*.
	intersections_nc_star = concept_reducer(nc_star, opt=OWLObjectIntersectionOf)
	# (15) \exist r. C s.t. C \in NC* and r \in R* .
	exist_nc_star = concept_reducer_properties(
		concepts=nc_star,
		properties=object_properties_and_inverse,
		cls=OWLObjectSomeValuesFrom,
	)
	# (16) \forall r. C s.t. C \in NC* and r \in R* .
	for_all_nc_star = concept_reducer_properties(
		concepts=nc_star,
		properties=object_properties_and_inverse,
		cls=OWLObjectAllValuesFrom,
	)
	# (17) >= n r. C  and =< n r. C, s.t. C \in NC* and r \in R* .
	min_cardinality_nc_star_1, min_cardinality_nc_star_2, min_cardinality_nc_star_3 = (
		concept_reducer_properties(
			concepts=nc_star,
			properties=object_properties_and_inverse,
			cls=OWLObjectMinCardinality,
			cardinality=i,
		)
		for i in [1, 2, 3]
	)
	max_cardinality_nc_star_1, max_cardinality_nc_star_2, max_cardinality_nc_star_3 = (
		concept_reducer_properties(
			concepts=nc_star,
			properties=object_properties_and_inverse,
			cls=OWLObjectMaxCardinality,
			cardinality=i,
		)
		for i in [1, 2, 3]
	)
	# (18) \exist r. Nominal s.t. Nominal \in Nominals and r \in R* .
	exist_nominals = concept_reducer_properties(
		concepts=nominal_combinations,
		properties=object_properties_and_inverse,
		cls=OWLObjectSomeValuesFrom,
	)

	concepts = list(
		chain(
			nc,           # named concepts          (C)
			nnc,                   # negated named concepts  (\neg C)
			unions_nc_star,        # A set of Union of named concepts and negat
			intersections_nc_star, #
			exist_nc_star,
			for_all_nc_star,
			min_cardinality_nc_star_1, min_cardinality_nc_star_1, min_cardinality_nc_star_3,
			max_cardinality_nc_star_1, max_cardinality_nc_star_2, max_cardinality_nc_star_3,
			exist_nominals))
			
	random.shuffle(concepts)  
	return concepts

def execute(args):
	"""
	Executes the retrieval runtime evaluation for the neural reasoner API.
	"""
	# Initialize the normal reasoner using OwlReady2
	kb = KnowledgeBase(path=args.path_kg)
	local_reasoner = TripleStoreNeuralReasoner(path_of_kb=args.path_kg)

	# Generate concepts from the KB
	concepts = generate_concepts_from_kb(kb) * 10

	# Prepare data storage
	data = []
	total_local_runtime = 0
	total_litserve_runtime = 0

	# Start asynchronous session for neural reasoner API
	async def main():
		nonlocal total_local_runtime, total_litserve_runtime
		async with aiohttp.ClientSession() as session:
			tasks = []
			for expression in tqdm(concepts, desc="Preparing tasks"):
				namespace = args.namespace  # Assume a fixed namespace for all expressions
				tasks.append(
					asyncio.ensure_future(
						async_neural_retrieval(
							session, args.neural_reasoner_url, expression, namespace
						)
					)
				)

			# Measure retrieval times
			results = []
			for idx, expression in enumerate(tqdm(concepts, desc="Evaluating expressions")):
				dl_expression = owl_expression_to_dl(expression)
				
				# Normal reasoner retrieval
				start_time = time.time()
				individuals_local_gen = local_reasoner.individuals(expression)
				individuals_local = set(individuals_local_gen)
				time_local = time.time() - start_time
				total_local_runtime += time_local

				# Neural reasoner retrieval (already scheduled)
				start_time = time.time()
				individuals_litserve = await tasks[idx]
				time_litserve = time.time() - start_time  
				total_litserve_runtime += time_litserve
				individuals_litserve = {OWLNamedIndividual(IRI.create(namespace, ind)) for ind in individuals_litserve}

				if individuals_local != individuals_litserve:
					print(f"Error in retrieval for expression: {dl_expression}")
					print(f"Local: {individuals_local}")
					print(f"Litserve: {individuals_litserve}")
					print()
				# Store data
				data.append({
					"Expression": dl_expression,
					"Type": type(expression).__name__,
					"Runtime Local": time_local,
					"Runtime litserve": time_litserve,
					"Runtime Difference": time_local - time_litserve,
					"Individuals Normal": individuals_local,
					"Individuals Neural": individuals_litserve,
				})

	# Run the asynchronous main function
	asyncio.run(main())

	if total_litserve_runtime > 0:
		speedup_factor = total_local_runtime / total_litserve_runtime
		print(f"Litserve is, on average, {speedup_factor:.2f} times faster than the local retrieval.")
	# Save results to DataFrame
	df = pd.DataFrame(data)
	df.to_csv(args.path_report, index=False)

	# Display summary
	df_summary = df.groupby("Type").agg({
		"Runtime Local": "mean",
		"Runtime litserve": "mean",
		"Runtime Difference": "mean"
	})
	print(df_summary)

def get_default_arguments():
	parser = ArgumentParser()
	parser.add_argument("--path_kg", type=str, default="KGs/Family/father.owl", help="Path to the ontology file")
	parser.add_argument("--neural_reasoner_url", type=str, default="http://localhost:8000", help="URL of the neural reasoner API")
	parser.add_argument("--namespace", type=str, default="http://example.com/father#", help="Default namespace")
	parser.add_argument("--path_report", type=str, default="retrieval_runtime_results.csv", help="Path to save the report")
	return parser.parse_args()

if __name__ == "__main__":
	execute(get_default_arguments())
