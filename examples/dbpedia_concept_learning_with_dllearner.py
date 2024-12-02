import numpy as np
import time
from ontolearn.binders import DLLearnerBinder
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.metrics import F1
from owlapy.converter import owl_expression_to_sparql
from owlapy.render import DLSyntaxObjectRenderer
import requests
from requests import Response
from requests.exceptions import RequestException, JSONDecodeError
import json

from ontolearn.learning_problem import PosNegLPStandard


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

# (1) SPARQL endpoint as knowledge source: supported only by DL-Learner-1.4.0
kb_path = sparql_endpoint

# To download DL-learner,  https://github.com/SmartDataAnalytics/DL-Learner/releases.
dl_learner_binary_path = "./dllearner-1.4.0/bin/cli"

results = {"algo": "dllearner_celoe", "runtime": {"values": [], "mean": None}, "f1": {"values": [], "mean": None}}

# (2) Read learning problem file
with open("./LPs/DBpedia2022-12/lps.json") as f:
    lps = json.load(f)
# (3) Start class expression learning
for i, item in enumerate(lps):
    print(f"\nLP {i+1}/{len(lps)} ==> Target expression: ", item["target expression"], "\n")
    lp = PosNegLPStandard(pos=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["positive examples"])))),
                          neg=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["negative examples"])))))
    
    celoe = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='celoe')
    print("\nStarting class expression learning with DL-Learner")
    try:
        t0 = time.time()
        h = celoe.fit(lp, use_sparql=True).best_hypothesis()
        t1 = time.time()
        print("\nLearned expression: ", h)
        print("Duration: ", t1-t0)
        concept_to_sparql_query = owl_expression_to_sparql(h) + "\nLIMIT 100"
        actual_instances = set(item["examples"]["positive examples"])
        retrieved_instances = set(query_func(concept_to_sparql_query)) #set(kb.individuals(h))
        print("Retrieved instances: ", retrieved_instances)
        f1 = f1_score(actual_instances, retrieved_instances)
        str_concept = renderer.render(h)
        print("Concept:", str_concept)
        print("Quality (F1): ", f1)
        results["runtime"]["values"].append(t1-t0)
        results["f1"]["values"].append(f1)
    except Exception as e:
        print(e)
        pass

if results["f1"]["values"]:
    results["f1"]["mean"] = np.mean(results["f1"]["values"])
    results["runtime"]["mean"] = np.mean(results["runtime"]["values"])

with open(f"./large_scale_cel_results_dllearner_celoe.json", "w") as f:
    json.dump(results, f)