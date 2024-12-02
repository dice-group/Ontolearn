import json, os, sys
import numpy as np
import time
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.learners import Drill, TDL
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.triple_store import TripleStore
from ontolearn.utils.static_funcs import save_owl_class_expressions
from ontolearn.metrics import F1
from owlapy.converter import owl_expression_to_sparql
from owlapy.render import DLSyntaxObjectRenderer
import requests
from requests import Response
from requests.exceptions import RequestException, JSONDecodeError

if len(sys.argv) < 2:
    print("You need to provide the model name; either tdl or drill")
    sys.exit(1)

model_name = sys.argv[1]
assert model_name.lower() in ["drill", "tdl"], "Currently, only Drill and TDL are supported"

sparql_endpoint = "https://dbpedia.data.dice-research.org/sparql"

# Helper functions
def f1_score(actual: set, retrieved: set):
    tp = len(actual.intersection(retrieved))
    fn = len(actual.difference(retrieved))
    fp = len(retrieved.difference(actual))
    tn = 0 # Not used
    return F1().score2(tp, fn, fp, tn)[-1]

def query_func(query):
    try:
        response = requests.post(sparql_endpoint, data={"query": query}, timeout=300)
    except RequestException as e:
        raise RequestException(
            f"Make sure the server is running on the `triplestore_address` = '{sparql_endpoint}'"
            f". Check the error below:"
            f"\n  -->Error: {e}"
        )
        
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

# (1) Initialize knowledge source with TripleStore
kb = TripleStore(url=sparql_endpoint)
# (2) Initialize a DL renderer.
renderer = DLSyntaxObjectRenderer()
# (3) Initialize a learner.
model = Drill(knowledge_base=kb, max_runtime=240) if model_name.lower() == "drill" else TDL(knowledge_base=kb)
# (4) Solve learning problems
with open("./LPs/DBpedia2022-12/lps.json") as f:
    lps = json.load(f)
    
results = {"algo": f"{model_name}", "runtime": {"values": [], "mean": None}, "f1": {"values": [], "mean": None}}
for i, item in enumerate(lps):
    print("\nTarget expression: ", item["target expression"], "\n")
    lp = PosNegLPStandard(pos=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["positive examples"])))),
                          neg=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["negative examples"])))))
    # (5) Learn description logic concepts best fitting
    t0 = time.time()
    h = model.fit(learning_problem=lp).best_hypotheses()
    t1 = time.time()
    print("Starting query after solution is computed!")
    concept_to_sparql_query = owl_expression_to_sparql(h) + "\nLIMIT 100"
    actual_instances = set(item["examples"]["positive examples"])
    retrieved_instances = set(query_func(concept_to_sparql_query)) #set(kb.individuals(h))
    f1 = f1_score(actual_instances, retrieved_instances)
    str_concept = renderer.render(h)
    print("Concept:", str_concept)
    print("Quality (F1): ", f1)
    results["runtime"]["values"].append(t1-t0)
    results["f1"]["values"].append(f1)
    # (6) Save e.g., ∃ predecessor.WikicatPeopleFromBerlin into disk
    if not os.path.exists(f"./learned_owl_expressions_{model_name}"):
        os.mkdir(f"./learned_owl_expressions_{model_name}")
    save_owl_class_expressions(expressions=h, path=f"./learned_owl_expressions_{model_name}/owl_prediction_{i}")
    
results["f1"]["mean"] = np.mean(results["f1"]["values"])
results["runtime"]["mean"] = np.mean(results["runtime"]["values"])

with open(f"./large_scale_cel_results_{model_name}.json", "w") as f:
    json.dump(results, f)
