import requests
from requests import Response
from requests.exceptions import RequestException, JSONDecodeError
from owlapy.converter import owl_expression_to_sparql
from owlapy.parser import DLSyntaxParser
from ontolearn.triple_store import TripleStore
import random
import numpy as np
import json

random.seed(42)
np.random.seed(42)

sparql_endpoint = "https://dbpedia.data.dice-research.org/sparql"

rdfs_prefix = "PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n "

namespace = "http://dbpedia.org/ontology/"

dls_parser = DLSyntaxParser(namespace=namespace)

kb = TripleStore(url=sparql_endpoint)


selected_concepts_str = ['http://dbpedia.org/ontology/Journalist', 'http://dbpedia.org/ontology/HistoricPlace', 'http://dbpedia.org/ontology/Lipid', 'http://dbpedia.org/ontology/Profession', 'http://dbpedia.org/ontology/Model', 'http://dbpedia.org/ontology/President', 'http://dbpedia.org/ontology/Academic', 'http://dbpedia.org/ontology/Actor', 'http://dbpedia.org/ontology/Place', 'http://dbpedia.org/ontology/FootballMatch']

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

    
def generate_lps():
    pass

if __name__ == "__main__":
    all_obj_props = list(kb.ontology.object_properties_in_signature())
    
    all_lps = []
    
    for i in range(200):
        connectors = ['⊔', '⊓']
        neg = "¬"
        quantifiers = ['∃', '∀']

        expression = f"<{random.choice(selected_concepts_str)}> {random.choice(connectors)} <{random.choice(selected_concepts_str)}>"

        if random.random() > 0.9:
            expression = f"{neg}{expression}"

        if random.random() > 0.8:
            expression = f"{random.choice(quantifiers)} <{random.choice(all_obj_props).str}>.({expression})"

        neg_expression = neg + f"({expression})"
        concept = dls_parser.parse(expression)
        concept_neg = dls_parser.parse(neg_expression)

        sparql_query = owl_expression_to_sparql(concept) + "\nLIMIT 100"
        sparql_query_neg = owl_expression_to_sparql(concept_neg) + "\nLIMIT 100"

        print(sparql_query)
        print("\nNeg query")
        print(sparql_query_neg)
        
        pos_inds = list(query_func(sparql_query))
        neg_inds = list(query_func(sparql_query_neg))
        
        if len(pos_inds) <= 1 or len(neg_inds) <= 1:
            continue
        
        if pos_inds and neg_inds:
            lp = {"target expression": expression,
                   "examples": {"positive examples": pos_inds,
                                "negative examples": neg_inds}
              }
            
            all_lps.append(lp)
    
    with open("Large_scale_lps.json", "w") as f:
        json.dump(all_lps, f)
    
    