"""python examples/retrieval_eval.py"""

from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.utils import jaccard_similarity
from owlapy.class_expression import (
    OWLQuantifiedObjectRestriction,
    OWLObjectCardinalityRestriction,
)
from owlapy.class_expression import (
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectOneOf,
)
import time
from typing import List, Tuple, Union, Set, Iterable, Callable
import pandas as pd
from owlapy import owl_expression_to_dl
from itertools import chain
from argparse import ArgumentParser
import os
from tqdm import tqdm

# TODO:CD: Fix the seed
import random
import itertools


# @TODO Move into ontolearn.utils
def concept_reducer(concepts, opt):
    result = set()
    for i in concepts:
        for j in concepts:
            result.add(opt((i, j)))
    return result


# @TODO Move into ontolearn.utils
def concept_reducer_properties(
    concepts: Set, properties, cls: Callable = None, cardinality: int = 2
) -> Set[Union[OWLQuantifiedObjectRestriction, OWLObjectCardinalityRestriction]]:
    """
    Map a set of owl concepts and a set of properties into OWL Restrictions

    Args:
        concepts:
        properties:
        cls (Callable): An owl Restriction class
        cardinality: A positive Integer

    Returns: List of OWL Restrictions

    """
    assert isinstance(concepts, Iterable), "Concepts must be an Iterable"
    assert isinstance(properties, Iterable), "properties must be an Iterable"
    assert isinstance(cls, Callable), "cls must be an Callable"
    assert cardinality > 0
    result = set()
    for i in concepts:
        for j in properties:
            if cls == OWLObjectMinCardinality or cls == OWLObjectMaxCardinality:
                result.add(cls(cardinality=cardinality, property=j, filler=i))
                continue
            result.add(cls(j, i))
    return result


# @TODO: CD: Perhaps we can remove this function.
def concept_to_retrieval(concepts, retriever) -> List[Tuple[float, Set[str]]]:
    results = []
    for c in concepts:
        start_time_ = time.time()
        retrieval = {i.str for i in retriever.individuals(c)}
        results.append((time.time() - start_time_, retrieval))
    return results


# @TODO: CD: Perhaps we can remove this function.
def retrieval_eval(expressions, y, yhat, verbose=1):
    assert len(y) == len(yhat)
    similarities = []
    runtime_diff = []
    number_of_concepts = len(expressions)
    for expressions, y_report_i, yhat_report_i in zip(expressions, y, yhat):
        runtime_y_i, y_i = y_report_i
        runtime_yhat_i, yhat_i = yhat_report_i

        jaccard_sim = jaccard_similarity(y_i, yhat_i)
        runtime_benefits = runtime_y_i - runtime_yhat_i
        if verbose > 0:
            print(
                f"Concept:{expressions}\tTrue Size:{len(y_i)}\tPredicted Size:{len(yhat_i)}\tRetrieval Similarity:{jaccard_sim}\tRuntime Benefit:{runtime_benefits:.3f}"
            )
        similarities.append(jaccard_sim)
        runtime_diff.append(runtime_benefits)
    avg_jaccard_sim = sum(similarities) / len(similarities)
    avg_runtime_benefits = sum(runtime_diff) / len(runtime_diff)
    return number_of_concepts, avg_jaccard_sim, avg_runtime_benefits


def execute(args):
    # (1) Initialize knowledge base.
    assert os.path.isfile(args.path_kg)
    if args.endpoint_triple_store:
        symbolic_kb = TripleStore(url="http://localhost:3030/family")
    else:
        symbolic_kb = KnowledgeBase(path=args.path_kg)
    # (2) Initialize Neural OWL Reasoner.
    if args.path_kge_model:
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_neural_embedding=args.path_kge_model, gamma=args.gamma
        )
    else:
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb=args.path_kg, gamma=args.gamma
        )
    ###################################################################
    # GENERATE ALCQ CONCEPTS TO EVALUATE RETRIEVAL PERFORMANCES
    # (3) R: Extract object properties.
    object_properties = {i for i in symbolic_kb.get_object_properties()}
    # (4) R⁻: Inverse of object properties.
    object_properties_inverse = {i.get_inverse_property() for i in object_properties}
    # (5) R*: R UNION R⁻.
    object_properties_and_inverse = object_properties.union(object_properties_inverse)
    # (6) NC: Named owl concepts.
    nc = {i for i in symbolic_kb.get_concepts()}
    # (7) NC⁻: Complement of NC.
    nnc = {i.get_object_complement_of() for i in nc}
    # (8) UNNC: NC UNION NC⁻.
    unnc = nc.union(nnc)
    # (9) Retrieve 10 random Nominals.
    nominals = set(random.sample(symbolic_kb.all_individuals_set(), 10))
    # (10) All Combinations of 3 for Nominals.
    nominal_combinations = set(
        OWLObjectOneOf(combination)
        for combination in itertools.combinations(nominals, 3)
    )
    # (11) NC UNION NC.
    unions = concept_reducer(nc, opt=OWLObjectUnionOf)
    # (12) NC INTERSECTION NC.
    intersections = concept_reducer(nc, opt=OWLObjectIntersectionOf)
    # (13) UNNC UNION UNNC.
    unions_unnc = concept_reducer(unnc, opt=OWLObjectUnionOf)
    # (14) UNNC INTERACTION UNNC.
    intersections_unnc = concept_reducer(unnc, opt=OWLObjectIntersectionOf)

    # (15) \exist r. C s.t. C \in UNNC and r \in R* .
    exist_unnc = concept_reducer_properties(
        concepts=unnc,
        properties=object_properties_and_inverse,
        cls=OWLObjectSomeValuesFrom,
    )
    # (16) \forall r. C s.t. C \in UNNC and r \in R* .
    for_all_unnc = concept_reducer_properties(
        concepts=unnc,
        properties=object_properties_and_inverse,
        cls=OWLObjectAllValuesFrom,
    )
    # (17) >= n r. C  and =< n r. C, s.t. C \in UNNC and r \in R* .
    min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3 = (
        concept_reducer_properties(
            concepts=unnc,
            properties=object_properties_and_inverse,
            cls=OWLObjectMinCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3 = (
        concept_reducer_properties(
            concepts=unnc,
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
    ###################################################################

    # Retrieval Results

    def concept_retrieval(retriever_func, c) -> Tuple[Set[str], float]:
        start_time = time.time()
        return {i.str for i in retriever_func.individuals(c)}, time.time() - start_time

    data = []
    # Converted to list so that the progress bar works.
    concepts = list(
        chain(
            nc,
            unions,
            intersections,
            nnc,
            unnc,
            unions_unnc,
            intersections_unnc,
            exist_unnc,
            for_all_unnc,
            min_cardinality_unnc_1,
            min_cardinality_unnc_2,
            min_cardinality_unnc_3,
            max_cardinality_unnc_1,
            max_cardinality_unnc_2,
            max_cardinality_unnc_3,
            exist_nominals,
        )
    )
    # Shuffled the data so that the progress bar is not influenced by the order of concepts.
    random.shuffle(concepts)
    # Converted to list so that the progress bar works.
    for expression in (tqdm_bar := tqdm(concepts, position=0, leave=True)):
        retrieval_y, runtime_y = concept_retrieval(symbolic_kb, expression)
        retrieval_neural_y, runtime_neural_y = concept_retrieval(
            neural_owl_reasoner, expression
        )
        jaccard_sim = jaccard_similarity(retrieval_y, retrieval_neural_y)
        data.append(
            {
                "Expression": owl_expression_to_dl(expression),
                "Type": type(expression).__name__,
                "Jaccard Similarity": jaccard_sim,
                "Runtime Benefits": runtime_y - runtime_neural_y,
            }
        )
        tqdm_bar.set_description_str(
            f"Expression: {owl_expression_to_dl(expression)} | Jaccard Similarity:{jaccard_sim:.4f} | Runtime Benefits:{runtime_y - runtime_neural_y:.3f}"
        )

    df = pd.DataFrame(data)
    assert df["Jaccard Similarity"].mean() == 1.0

    df.to_csv(args.path_report)
    del df
    df = pd.read_csv(args.path_report, index_col=0)
    numerical_df = df.select_dtypes(include=["number"])
    df_g = df.groupby(by="Type")
    print(df_g["Type"].count())
    mean_df = df_g[numerical_df.columns].mean()
    print(mean_df)


def get_default_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--path_kg", type=str, default="KGs/Family/family-benchmark_rich_background.owl"
    )
    parser.add_argument("--path_kge_model", type=str, default=None)
    parser.add_argument("--endpoint_triple_store", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--path_report", type=str, default="ALCQ_Retrieval_Results.csv")
    return parser.parse_args()


if __name__ == "__main__":
    execute(get_default_arguments())
