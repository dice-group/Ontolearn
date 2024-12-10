"""$ python examples/retrieval_eval.py --path_kg "https://dbpedia.data.dice-research.org/sparql"

"""
import os
from tqdm import tqdm
import random
import itertools
import ast
import json
import time
import requests
from requests import Response
from requests.exceptions import RequestException, JSONDecodeError
from ontolearn.learners import Drill, TDL
from ontolearn.triple_store import TripleStore
from owlapy.owl_individual import OWLNamedIndividual, IRI
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.utils import f1_set_similarity, compute_f1_score
from typing import Tuple, Set
import pandas as pd
from owlapy.parser import DLSyntaxParser
from owlapy import owl_expression_to_dl
from owlapy.converter import owl_expression_to_sparql
from argparse import ArgumentParser
# Set pandas options to ensure full output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.expand_frame_repr', False)

def execute(args):
    # (1) Initialize knowledge base.
    assert args.endpoint_triple_store, 'A SPARQL endpoint of DBpedia must be provided via `--endpoint_triple_store "url"`'
    try:
        kb = TripleStore(url=args.endpoint_triple_store)
        kb_namespace = list(kb.ontology.classes_in_signature())[0].iri.get_namespace()
        dl_parser = DLSyntaxParser(kb_namespace)
    except:
        raise ValueError("You must provide a valid SPARQL endpoint!")
    # Fix the random seed.
    random.seed(args.seed)

    
    ###################################################################

    print("\n")
    print("#"*50)
    print("Starting class expression learning on DBpedia...")
    print("#" * 50,end="\n\n")
    
    def query_func(query):
        try:
            response = requests.post(args.endpoint_triple_store, data={"query": query}, timeout=300)
        except RequestException as e:
            raise RequestException(
                f"Make sure the server is running on the `triplestore_address` = '{args.endpoint_triple_store}'"
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

    model = Drill(knowledge_base=kb, max_runtime=240) if args.model.lower() == "drill" else TDL(knowledge_base=kb)
    with open("./LPs/DBpedia2022-12/lps.json") as f:
        lps = json.load(f)

    if os.path.exists(args.path_report):
        os.remove(args.path_report)
    file_exists = False
    
    for item in (tqdm_bar := tqdm(lps, position=0, leave=True)):
        
        retrieval_y: Set[str]
        runtime_y: float
            
        lp = PosNegLPStandard(pos=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["positive examples"])))),
                      neg=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["negative examples"])))))
        # (5) Learn description logic concepts best fitting
        t0 = time.time()
        h = model.fit(learning_problem=lp).best_hypotheses()
        t1 = time.time()
        print("\nStarting query after solution is computed!\n")
        concept_to_sparql_query = owl_expression_to_sparql(h) + "\nLIMIT 100" # Due to the size of DBpedia learning problems contain at most 100 pos and 100 neg examples
        actual_instances = set(item["examples"]["positive examples"])
        retrieved_instances = set(query_func(concept_to_sparql_query))
        f1 = compute_f1_score(retrieved_instances, set(item["examples"]["positive examples"]), set(item["examples"]["negative examples"]))
        print(f"Computed solution: {h}")
        
        df_row = pd.DataFrame(
            [{
                "Expression": owl_expression_to_dl(dl_parser.parse(item["target expression"])),
                "Type": type(dl_parser.parse(item["target expression"])).__name__,
                "F1": f1,
                "Runtime": t1 - t0,
                #"Retrieved_Instances": retrieved_instances,
            }])
        
        # Append the row to the CSV file
        df_row.to_csv(args.path_report, mode='a', header=not file_exists, index=False)
        file_exists = True
        # () Update the progress bar.
        tqdm_bar.set_description_str(
            f"Expression: {owl_expression_to_dl(dl_parser.parse(item['target expression']))} | F1 :{f1:.4f} | Runtime:{t1 - t0:.3f}"
        )
    # () Read the data into pandas dataframe
    df = pd.read_csv(args.path_report, index_col=0)
    # () Assert that the mean Jaccard Similarity meets the threshold
    assert df["F1"].mean() >= args.min_f1_score

    # () Extract numerical features
    numerical_df = df.select_dtypes(include=["number"])

    # () Group by the type of OWL concepts
    df_g = df.groupby(by="Type")
    print(df_g["Type"].count())

    # () Compute mean of numerical columns per group
    mean_df = df_g[numerical_df.columns].mean()
    print(mean_df)
    return f1

def get_default_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Drill")
    parser.add_argument("--path_kge_model", type=str, default=None)
    parser.add_argument("--endpoint_triple_store", type=str, default="https://dbpedia.data.dice-research.org/sparql")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min_f1_score", type=float, default=0.0, help="Minimum f1 score of computed solutions")

    # H is obtained if the forward chain is applied on KG.
    parser.add_argument("--path_report", type=str, default=f"CEL_on_DBpedia.csv")
    return parser.parse_args()

if __name__ == "__main__":
    execute(get_default_arguments())
