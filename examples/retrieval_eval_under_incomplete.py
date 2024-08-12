"""
TODO: Write few lines of code to run this script and explanations
"""
from argparse import ArgumentParser
from ontolearn.knowledge_base import KnowledgeBase
import pandas as pd
from typing import Set
from incomplete_kb import *
import os
from ontolearn.utils import jaccard_similarity
import subprocess
from owlapy.class_expression import *
from owlapy.iri import IRI
from owlapy.parser import DLSyntaxParser
import ast


# [] Create sub/incomplete KGs
def generated_incomplete_kg(kb_path: str, directory: str, n: int, ratio: float) -> Set[str]:

    # (1)
    # TODO:CD: Ensure that randomness can be controlled via seed
    # TODO:CD: Return a set of strings where each item corresponds ot the local path of a sub kg.

    #e.g. of how the file can be save
    # kb_path = "KGs/Family/father.owl"  
    # output_path = f"incomplete_father_ratio_10_number_1.owl" 

    name = kb_path.split('/')[-1].split('.')[0]
    rate = int(ratio * 100)

    os.makedirs(directory, exist_ok=True)

    file_paths = set()

    for i in range(1, n + 1):

        # Construct the output path for the incomplete KG
        output_path = f'{directory}/incomplete_{name}_ratio_{rate}_number_{i}.owl'

        # function to generate the incomplete KG
        make_kb_incomplete(kb_path, output_path, rate, seed=i)

        # Add the output path to the set
        file_paths.add(output_path)

    # Return the set of file paths
    return file_paths

# kb_path = "KGs/Family/father.owl"  
# directory = "incomplete_father" 
# u,v = generated_incomplete_kg(kb_path , directory , 2, 0.1)
# print(u,v)

def execute(args):

    symbolic_kb = KnowledgeBase(path=args.path_kg)
    # print({i.str for i in symbolic_kb.individuals(OWLClass(IRI('http://example.com/fatherr#','female')))})
    # exit(0)
    # TODO: Report the results in a CSV file as we have done it in retieval_eval.py
    # Load the full KG
     

    # TODO: What should be directory args.path_kg?
    name_KG = args.path_kg.split('/')[-1].split('.')[0]

    directory = f"incomplete_{name_KG}"
    paths_of_incomplete_kgs = generated_incomplete_kg(kb_path=args.path_kg, directory=directory, n=1, ratio=0.99)


    expressions = None

    

    for path_of_an_incomplete_kgs in paths_of_incomplete_kgs:

        symbolic_kb = KnowledgeBase(path=path_of_an_incomplete_kgs)
        # print({i.str for i in symbolic_kb.individuals(OWLClass(IRI('http://www.w3.org/2002/07/owl#','female')))})
        # exit(0)

        namespace = list(symbolic_kb.ontology.classes_in_signature())[0].iri.get_namespace()

        parser = DLSyntaxParser(namespace)

        list_jaccard_symbolic = []
        list_jaccard_neural = []

        # Train a KGE, retrieval eval vs KGE and Symbolic
        # args.ratio_sample_nc
        # args.ratio_sample_object_prob
        subprocess.run(['python', 'examples/retrieval_eval.py', "--path_kg", path_of_an_incomplete_kgs])
        # Load the results on the current view.
        df = pd.read_csv("ALCQHI_Retrieval_Results.csv", index_col=0)

        # Sanity checking
        if expressions is None:
            expressions = df["Expression"].values
        else:
            assert expressions == df["Expression"].values

        # Iterate
        for expression in expressions:

            # TODO: str -> owlapy.owl_classexpression object
            
            target_concept = parser.parse_expression(expression) #row["Expression"]
            # print(target_concept)
            
            
            goal_retrieval = {i.str for i in symbolic_kb.individuals(target_concept)}

            result_symbolic: Set[str]
            result_neural_symbolic: Set[str]

            result_symbolic = df[df["Expression"]==expression]["Symbolic_Retrieval"].apply(ast.literal_eval)
            result_symbolic = result_symbolic.iloc[0] 

            result_neural_symbolic = df[df["Expression"]==expression]["Symbolic_Retrieval_Neural"].apply(ast.literal_eval)
            result_neural_symbolic = result_neural_symbolic.iloc[0]
            

            # jaccard_sim_symbolic = jaccard_similarity(row["Symbolic_Retrieval_Neural"], goal_retrieval)
            jaccard_sim_symbolic = jaccard_similarity(result_symbolic, goal_retrieval)

            # jaccard_sim_neural = jaccard_similarity(row["Symbolic_Retrieval_Neural"], goal_retrieval)
            jaccard_sim_neural = jaccard_similarity(result_neural_symbolic, goal_retrieval)

            #Update the Average
            list_jaccard_neural.append(jaccard_sim_neural)
            list_jaccard_symbolic.append(jaccard_sim_symbolic)


            # Ideally
            # jaccard_sim_neural > jaccard_sim_symbolic
        avg_jaccard_sym = sum(list_jaccard_symbolic)/len(list_jaccard_symbolic)
        avg_jaccard_neural = sum(list_jaccard_neural)/len(list_jaccard_neural)

        print("Average jaccard symbolic", avg_jaccard_sym)
        print("Average Jaccard neural", avg_jaccard_neural)
        print(list_jaccard_neural)
        print(list_jaccard_symbolic)


def get_default_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path_kg", type=str, default="KGs/Family/family-benchmark_rich_background.owl")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ratio_sample_nc", type=float, default=None, help="To sample OWL Classes.")
    parser.add_argument("--ratio_sample_object_prob", type=float, default=None, help="To sample OWL Object Properties.")
    parser.add_argument("--path_report", type=str, default="ALCQHI_Retrieval_Incomplete_Results.csv")
    return parser.parse_args()


if __name__ == "__main__":
    execute(get_default_arguments())
