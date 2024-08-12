"""
TODO: Write few lines of code to run this script and explanations
"""
from argparse import ArgumentParser
from ontolearn.knowledge_base import KnowledgeBase
import pandas as pd
from typing import Set

import subprocess


# [] Create sub/incomplete KGs
def generated_incomplete_kg(path: str, directory: str, n: int, ratio: float) -> Set[str]:
    # (1)
    # TODO:CD: Ensure that randomness can be controlled via seed
    # TODO:CD: Return a set of strings where each item corresponds ot the local path of a sub kg.
    pass


def execute(args):
    # TODO: Report the results in a CSV file as we have done it in retieval_eval.py
    # Load the full KG
    symbolic_kb = KnowledgeBase(path=args.path_kg)
    # TODO: What should be directory args.path_kg?
    paths_of_incomplete_kgs = generated_incomplete_kg(path=args.path_kg, directory="", n=10, ratio=0.7)

    def jaccard_similarity(x, y):
        pass

    expressions = None

    for path_of_an_incomplete_kgs in paths_of_incomplete_kgs:

        # Train a KGE, retrieval eval vs KGE and Symbolic
        # args.ratio_sample_nc
        # args.ratio_sample_object_prob
        subprocess.run(['python', 'examples/retrieval_eval.py', "--path_kg", path_of_an_incomplete_kgs])
        # Load the results on the current view.
        df = pd.read_csv("ALCQHI_Retrieval_Results.csv", index_col=0)
        # Sanity checking
        if expressions is None:
            expressions = df["Expressions"].values
        else:
            assert expressions == df["Expressions"].values

        # Iterate
        for row in df:
            target_concept = row["Expression"]
            # TODO: str -> owlapy.owl_classexpression object
            goal_retrieval = {i.str for i in symbolic_kb.individuals(target_concept)}
            result_symbolic: Set[str]
            result_neural_symbolic: Set[str]

            jaccard_sim_symbolic = jaccard_similarity(row["Symbolic_Retrieval_Neural"], goal_retrieval)

            jaccard_sim_neural = jaccard_similarity(row["Symbolic_Retrieval_Neural"], goal_retrieval)

            # Ideally
            jaccard_sim_neural > jaccard_sim_symbolic


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
