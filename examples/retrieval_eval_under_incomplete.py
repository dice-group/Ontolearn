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
from owlapy import owl_expression_to_dl
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owlapi_adaptor import OWLAPIAdaptor
import pandas as pd


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

        # output path for the incomplete KGs
        output_path = f'{directory}/incomplete_{name}_ratio_{rate}_number_{i}.owl'

        # Check if the file already exists
        if not os.path.exists(output_path):
            # If file does not exist, generate it
            make_kb_incomplete(kb_path, output_path, rate, seed=i)

        # Add the output path to the set
        file_paths.add(output_path)

    return file_paths


def execute(args):

    symbolic_kb = KnowledgeBase(path=args.path_kg)
         
    namespace = list(symbolic_kb.ontology.classes_in_signature())[0].iri.get_namespace()

    parser = DLSyntaxParser(namespace)


    # TODO: What should be directory args.path_kg?
    name_KG = args.path_kg.split('/')[-1].split('.')[0]

    directory = f"incomplete_{name_KG}"

    paths_of_incomplete_kgs = generated_incomplete_kg(kb_path=args.path_kg, directory=directory,\
                                n=args.number_of_incomplete_graphs, ratio=args.level_of_incompleteness)

    # TODO: make sure the number of triple match inside 
    # TODO: ensure all triples are subset of the original KG
    expressions = None

    for path_of_an_incomplete_kgs in paths_of_incomplete_kgs:

        data = []
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
            expressions = {i for i in df["Expression"].to_list()}
        else:
            assert expressions == {i for i in df["Expression"].to_list()}

#----------------------------------------------------------------------------------------------------------------

        ontology_path = path_of_an_incomplete_kgs
        # Available OWL Reasoners: 'HermiT', 'Pellet', 'JFact', 'Openllet'

        reasoners = ['HermiT', 'Pellet', 'JFact', 'Openllet']

        owlapi_adaptor = OWLAPIAdaptor(path=ontology_path, name_reasoner="JFact")
        # Iterate over defined owl Classes in the signature
        # for i in onto.classes_in_signature():
        #     print(i)
        #     exit(0)
        #     # Performing type inference with Pellet
        #     instances=owlapi_adaptor.instances(i,direct=False)
        #     print(f"Class:{i}\t Num instances:{len(instances)}")
        # owlapi_adaptor.stopJVM()

        # exit(0)

#------------------------------------------------------------------------------------------------------------------

        # Iterate
        for expression in expressions:

            # TODO: str -> owlapy.owl_classexpression object
            
            target_concept = parser.parse_expression(expression) 
    
            goal_retrieval = {i.str for i in symbolic_kb.individuals(target_concept)}

            result_symbolic: Set[str]
            result_neural_symbolic: Set[str]

            # result_symbolic = df[df["Expression"]==expression]["Symbolic_Retrieval"].apply(ast.literal_eval)
            # result_symbolic = result_symbolic.iloc[0] 

            result_symbolic = {i.str for i in (owlapi_adaptor.instances(target_concept,direct=False))}

            result_neural_symbolic = df[df["Expression"]==expression]["Symbolic_Retrieval_Neural"].apply(ast.literal_eval)
            result_neural_symbolic = result_neural_symbolic.iloc[0]
            

            jaccard_sim_symbolic = jaccard_similarity(result_symbolic, goal_retrieval)

            jaccard_sim_neural = jaccard_similarity(result_neural_symbolic, goal_retrieval)

            # Update for Averaging
            list_jaccard_neural.append(jaccard_sim_neural)
            list_jaccard_symbolic.append(jaccard_sim_symbolic)

            data.append(
            {
                "Expression": expression,
                "Type": type(expression).__name__,
                "Jaccard_sym": jaccard_sim_symbolic,
                "Jaccard_EBR": jaccard_sim_neural,
                # "Runtime Benefits": runtime_y - runtime_neural_y,
                # "Symbolic_Retrieval": retrieval_y,
                # "Symbolic_Retrieval_Neural": retrieval_neural_y,
            }
        )
            
        
        df = pd.DataFrame(data=data)

        print(df)

        avg_jaccard_sym = sum(list_jaccard_symbolic)/len(list_jaccard_symbolic)
        avg_jaccard_neural = sum(list_jaccard_neural)/len(list_jaccard_neural)

        print("Average jaccard symbolic", avg_jaccard_sym)
        print("Average Jaccard neural", avg_jaccard_neural)

    owlapi_adaptor.stopJVM() #stop the standard reasoner

def get_default_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path_kg", type=str, default="KGs/Family/family-benchmark_rich_background.owl")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ratio_sample_nc", type=float, default=None, help="To sample OWL Classes.")
    parser.add_argument("--ratio_sample_object_prob", type=float, default=None, help="To sample OWL Object Properties.")
    parser.add_argument("--path_report", type=str, default="ALCQHI_Retrieval_Incomplete_Results.csv")
    parser.add_argument("--number_of_incomplete_graphs", type = int, default=1)
    parser.add_argument("--level_of_incompleteness", type = float, default=0.1, \
                        help="Percentage of incompleteness from the original KGs between 0 and 1")
    return parser.parse_args()


if __name__ == "__main__":
    execute(get_default_arguments())
