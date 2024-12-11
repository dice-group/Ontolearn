"""$ python examples/owl_class_expresion_learning_dbpedia.py --endpoint_triple_store "https://dbpedia.data.dice-research.org/sparql" --model "TDL"
Computing conjunctive_concepts...

Constructing Description Logic Concepts:   0%|                                  Constructing Description Logic Concepts: 100%|██████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 870187.55it/s]
Computing disjunction_of_conjunctive_concepts...

Starting query after solution is computed!

Computed solution: OWLClass(IRI('http://dbpedia.org/ontology/', 'President'))
Expression: President ⊔ Actor | F1 :1.0000 | Runtime:171.275:  99%|██████████████████████████████████████████████████████████████▍| 105/106 [3:49:14<01:31, 91.8Expression: President ⊔ Actor | F1 :1.0000 | Runtime:171.275: 100%|██████████████████████████████████████████████████████████████| 106/106 [3:49:14<00:00, 115.6Expression: President ⊔ Actor | F1 :1.0000 | Runtime:171.275: 100%|██████████████████████████████████████████████████████████████| 106/106 [3:49:14<00:00, 129.76s/it]
Type
OWLObjectAllValuesFrom      7
OWLObjectIntersectionOf    14
OWLObjectUnionOf           85
Name: Type, dtype: int64
                         F1        Runtime   
Type                                         
OWLObjectAllValuesFrom   1.000000  206.287996
OWLObjectIntersectionOf  0.717172   91.663047
OWLObjectUnionOf         0.966652  129.699940
"""
# Make imports
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
    # Initialize knowledge base.
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
    
    # Define a query function to retrieve instances of class expressions
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
            
    # Initialize the model
    model = Drill(knowledge_base=kb, max_runtime=240) if args.model.lower() == "drill" else TDL(knowledge_base=kb)
    # Read learning problems from file
    with open("./LPs/DBpedia2022-12/lps.json") as f:
        lps = json.load(f)
        
    # Check if csv arleady exists and delete it cause we want to override it
    if os.path.exists(args.path_report):
        os.remove(args.path_report)
        
    file_exists = False
    # Iterate over all problems and solve
    for item in (tqdm_bar := tqdm(lps, position=0, leave=True)):
        # Create a learning problem object
        lp = PosNegLPStandard(pos=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["positive examples"])))),
                      neg=set(list(map(OWLNamedIndividual,map(IRI.create, item["examples"]["negative examples"])))))
        # Learn description logic concepts best fitting
        t0 = time.time()
        h = model.fit(learning_problem=lp).best_hypotheses()
        t1 = time.time()
        print("\nStarting query after solution is computed!\n")
        # Convert the learned expression into a sparql query
        concept_to_sparql_query = owl_expression_to_sparql(h) + "\nLIMIT 100" # Due to the size of DBpedia learning problems contain at most 100 pos and 100 neg examples
        # Load actual instances of the target expression
        actual_instances = set(item["examples"]["positive examples"])
        # Compute instances of the learned expression
        retrieved_instances = set(query_func(concept_to_sparql_query))
        # Compute the quality of the learned expression
        f1 = compute_f1_score(retrieved_instances, set(item["examples"]["positive examples"]), set(item["examples"]["negative examples"]))
        print(f"Computed solution: {h}")
        # Write results in a dictionary and create a dataframe
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
        # Update the progress bar.
        tqdm_bar.set_description_str(
            f"Expression: {owl_expression_to_dl(dl_parser.parse(item['target expression']))} | F1 :{f1:.4f} | Runtime:{t1 - t0:.3f}"
        )
    # Read the data into pandas dataframe
    df = pd.read_csv(args.path_report, index_col=0)
    # Assert that the mean f1 score meets the threshold
    assert df["F1"].mean() >= args.min_f1_score

    # Extract numerical features
    numerical_df = df.select_dtypes(include=["number"])

    # Group by the type of OWL concepts
    df_g = df.groupby(by="Type")
    print(df_g["Type"].count())

    # Compute mean of numerical columns per group
    mean_df = df_g[numerical_df.columns].mean()
    print(mean_df)
    return f1

def get_default_arguments():
    # Define an argument parser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Drill")
    parser.add_argument("--path_kge_model", type=str, default=None)
    parser.add_argument("--endpoint_triple_store", type=str, default="https://dbpedia.data.dice-research.org/sparql")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min_f1_score", type=float, default=0.0, help="Minimum f1 score of computed solutions")

    parser.add_argument("--path_report", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    # Get default or input values of arguments
    args = get_default_arguments()
    if not args.path_report:
        args.path_report = f"CEL_on_DBpedia_{args.model.upper()}.csv"
    execute(args)
