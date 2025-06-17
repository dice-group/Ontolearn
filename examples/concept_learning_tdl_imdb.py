
import json
import os
import re
from tqdm import tqdm
import random
import itertools
import ast
import json
import time
import requests
from requests.exceptions import RequestException, JSONDecodeError
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.learners import TDL, Drill, CELOE
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.triple_store import TripleStore
from ontolearn.utils.static_funcs import save_owl_class_expressions
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils import f1_set_similarity, compute_f1_score
from typing import Tuple, Set
import pandas as pd
from owlapy.parser import DLSyntaxParser
from owlapy import owl_expression_to_dl
from owlapy.converter import owl_expression_to_sparql




# Define a query function to retrieve instances of class expressions
def query_func(query):
    endpoint = "http://localhost:3030/imdb/sparql"
    try:
        response = requests.post(endpoint, data={"query": query}, timeout=300)
        response.raise_for_status()
    except RequestException as e:
        print(query)
        raise RequestException(
            f"Make sure the server is running at '{endpoint}'.\n"
            f"Check the error below:\n  --> Error: {e}"
        )

    try:
        json_results = response.json()
    except ValueError as e:
        
        raise ValueError(f"Invalid JSON response from SPARQL endpoint:\n  --> Error: {e}")
        


    json_results = response.json()
    vars_ = list(json_results["head"]["vars"])
    inds = []
    for b in json_results["results"]["bindings"]:
        val = []
        for v in vars_:
            if b[v]["type"] == "uri":
                val.append(b[v]["value"])
        inds.extend(val)

    if inds:
        yield from inds
    else:
        yield None


# (1) Initialize Triplestore- Make sure that UPB VPN is on
kb = TripleStore(url="http://localhost:3030/imdb/sparql")

kb_namespace = "https://www.imdb.com/"
dl_parser = DLSyntaxParser(kb_namespace)
# (2) Initialize a DL renderer.
render = DLSyntaxObjectRenderer()
# (3) Initialize a learner.
model = TDL(knowledge_base=kb,
                  max_runtime=10)
# Load the JSON file
with open('LPs/IMDB/learning_problems.json', 'r') as f:
    problems_data = json.load(f)


results = []

# Extract and iterate over the learning problems
for idx, problem in enumerate(problems_data["problems"]):
    print(f"\nSolving LP {idx + 1} with threshold {problem['threshold']}")
    
    positives_uris = problem["positives"]
    negatives_uris = problem["negatives"]

    positives = {OWLNamedIndividual(IRI.create(uri)) for uri in positives_uris}
    negatives = {OWLNamedIndividual(IRI.create(uri)) for uri in negatives_uris}

    # Create learning problem
    lp = PosNegLPStandard(pos=positives, neg=negatives)

    # Learn the concept
    t0 = time.time()
    hypotheses = model.fit(learning_problem=lp).best_hypotheses()
    t1 = time.time()

    h = hypotheses  # assuming best_hypotheses() returns a single expression
    str_concept = render.render(h)
    print("Concept:", str_concept)

    print("\nEvaluating learned concept over known examples...\n")

    # === BUILD QUERY ONLY OVER KNOWN INSTANCES ===
    all_uris = set(positives_uris + negatives_uris)
    concept_sparql = owl_expression_to_sparql(h)

    # Extract body inside WHERE { ... }
    body_match = re.search(r'\bWHERE\s*{(.*)}\s*$', concept_sparql, re.DOTALL | re.IGNORECASE)
    if body_match:
        concept_body = body_match.group(1).strip()
    else:
        concept_body = concept_sparql.strip()
        if concept_body.startswith("{") and concept_body.endswith("}"):
            concept_body = concept_body[1:-1].strip()

    if "GROUP BY" in concept_body or "HAVING" in concept_body or "UNION" in concept_body:
        concept_body = f"{{ {concept_body} }}"

    # Build VALUES clause
    values_clause = "\n".join([f"<{uri}>" for uri in all_uris])

    # Final SPARQL query
    filtered_query = f"""
    SELECT ?x WHERE {{
        VALUES ?x {{
            {values_clause}
        }}
        {concept_body}
    }}
    """

    # Run query and get retrieved instances
    retrieved_instances = set(query_func(filtered_query))

    # Compute F1-score
    f1 = compute_f1_score(
        retrieved_instances,
        set(positives_uris),
        set(negatives_uris)
    )

    # Store results
    results.append({
        "Threshold": problem["threshold"],
        "Expression": owl_expression_to_dl(dl_parser.parse(str_concept)),
        "Type": type(dl_parser.parse(str_concept)).__name__,
        "F1": f1,
        "Runtime": t1 - t0
    })

    print(f"Threshold {problem['threshold']}, F1 Score: {f1}")

# Convert results to DataFrame and save
df = pd.DataFrame(results)
df.to_csv("learning_results.csv", index=False)
print("\nSaved results to learning_results.csv")
