"""python examples/retrieval_eval.py"""
from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.utils import jaccard_similarity
from owlapy.class_expression import OWLQuantifiedObjectRestriction, OWLObjectCardinalityRestriction
from owlapy.class_expression import (
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
)
import time
from typing import List, Tuple, Union, Set, Iterable, Callable
import pandas as pd
from owlapy import owl_expression_to_dl
from itertools import chain
from argparse import ArgumentParser
import os
from tqdm import tqdm


# @TODO Move into ontolearn.utils
def concept_reducer(concepts, opt):
    result = set()
    for i in concepts:
        for j in concepts:
            result.add(opt((i, j)))
    return result


# @TODO Move into ontolearn.utils
def concept_reducer_properties(concepts: Set,
                               properties, cls: Callable = None,
                               cardinality: int = 2) -> Set[
    Union[OWLQuantifiedObjectRestriction, OWLObjectCardinalityRestriction]]:
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
    neural_owl_reasoner = TripleStoreNeuralReasoner(path_of_kb=args.path_kg, gamma=args.gamma)
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
    # (9) NC UNION NC.
    unions = concept_reducer(nc, opt=OWLObjectUnionOf)
    # (10) NC INTERSECTION NC.
    intersections = concept_reducer(nc, opt=OWLObjectIntersectionOf)
    # (11) UNNC UNION UNNC.
    unions_unnc = concept_reducer(unnc, opt=OWLObjectUnionOf)
    # (12) UNNC INTERACTION UNNC.
    intersections_unnc = concept_reducer(unnc, opt=OWLObjectIntersectionOf)

    # (13) \exist r. C s.t. C \in UNNC and r \in R* .
    exist_unnc = concept_reducer_properties(concepts=unnc,
                                            properties=object_properties_and_inverse,
                                            cls=OWLObjectSomeValuesFrom)
    # (15) \forall r. C s.t. C \in UNNC and r \in R* .
    for_all_unnc = concept_reducer_properties(concepts=unnc,
                                              properties=object_properties_and_inverse,
                                              cls=OWLObjectAllValuesFrom)
    # (16) >= n r. C  and =< n r. C, s.t. C \in UNNC and r \in R* .
    min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3 = (
        concept_reducer_properties(concepts=unnc, properties=object_properties_and_inverse, cls=OWLObjectMinCardinality,
                                   cardinality=i)
        for i in [1, 2, 3]
    )
    max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3 = (
        concept_reducer_properties(concepts=unnc,
                                   properties=object_properties_and_inverse,
                                   cls=OWLObjectMaxCardinality,
                                   cardinality=i)
        for i in [1, 2, 3]
    )

    ###################################################################

    # Retrieval Results

    def concept_retrieval(retriever_func, c) -> Tuple[Set[str], float]:
        start_time = time.time()
        return {i.str for i in retriever_func.individuals(c)}, time.time() - start_time

    data = []

    # Converted to list so that the progress bar works.
    for expression in (tqdm_bar := tqdm(list(chain(nc, unions, intersections,
                              nnc, unnc, unions_unnc, intersections_unnc,
                              exist_unnc, for_all_unnc,
                              min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3,
                              max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3)), position=0, leave=True)):
        retrieval_y, runtime_y = concept_retrieval(symbolic_kb, expression)
        retrieval_neural_y, runtime_neural_y = concept_retrieval(neural_owl_reasoner, expression)
        jaccard_sim = jaccard_similarity(retrieval_y, retrieval_neural_y)
        data.append({"Expression": owl_expression_to_dl(expression),
                     "Type": type(expression).__name__,
                     "Jaccard Similarity": jaccard_sim,
                     "Runtime Benefits": runtime_y - runtime_neural_y
                     })
        tqdm_bar.set_description_str(
            f"Expression: {owl_expression_to_dl(expression)} | Jaccard Similarity:{jaccard_sim:.4f} | Runtime Benefits:{runtime_y - runtime_neural_y:.3f}")

    df = pd.DataFrame(data)
    assert df["Jaccard Similarity"].mean() == 1.0

    df.to_csv(args.path_report)
    del df
    df = pd.read_csv(args.path_report, index_col=0)
    numerical_df = df.select_dtypes(include=['number'])
    df_g = df.groupby(by="Type")
    print(df_g["Type"].count())
    mean_df = df_g[numerical_df.columns].mean()
    print(mean_df)


def get_default_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path_kg", type=str,
                        default="KGs/Family/family-benchmark_rich_background.owl")
    parser.add_argument("--endpoint_triple_store", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--path_report", type=str, default="ALCQ_Retrieval_Results.csv")
    return parser.parse_args()


if __name__ == '__main__':
    execute(get_default_arguments())
# @TODO:CD:I guess we can remove the below part. What do you think Luke ?
"""
nc_retrieval_results = retrieval_eval(
    expressions=nc,
    y=concept_to_retrieval(nc, symbolic_kb),
    yhat=concept_to_retrieval(nc, neural_owl_reasoner),
)

unions_nc_retrieval_results = retrieval_eval(
    expressions=unions,
    y=concept_to_retrieval(unions, symbolic_kb),
    yhat=concept_to_retrieval(unions, neural_owl_reasoner),
)
intersections_nc_retrieval_results = retrieval_eval(
    expressions=intersections,
    y=concept_to_retrieval(intersections, symbolic_kb),
    yhat=concept_to_retrieval(intersections, neural_owl_reasoner),
)
nnc_retrieval_results = retrieval_eval(
    expressions=nnc,
    y=concept_to_retrieval(nnc, symbolic_kb),
    yhat=concept_to_retrieval(nnc, neural_owl_reasoner),
)
unnc_retrieval_results = retrieval_eval(
    expressions=unnc,
    y=concept_to_retrieval(unnc, symbolic_kb),
    yhat=concept_to_retrieval(unnc, neural_owl_reasoner),
)
unions_unnc_retrieval_results = retrieval_eval(
    expressions=unions_unnc,
    y=concept_to_retrieval(unions_unnc, symbolic_kb),
    yhat=concept_to_retrieval(unions_unnc, neural_owl_reasoner),
)
intersections_unnc_retrieval_results = retrieval_eval(
    expressions=intersections_unnc,
    y=concept_to_retrieval(intersections_unnc, symbolic_kb),
    yhat=concept_to_retrieval(intersections_unnc, neural_owl_reasoner),
)
exist_unnc_retrieval_results = retrieval_eval(
    expressions=exist_unnc,
    y=concept_to_retrieval(exist_unnc, symbolic_kb),
    yhat=concept_to_retrieval(exist_unnc, neural_owl_reasoner),
)
for_all_unnc_retrieval_results = retrieval_eval(
    expressions=for_all_unnc,
    y=concept_to_retrieval(for_all_unnc, symbolic_kb),
    yhat=concept_to_retrieval(for_all_unnc, neural_owl_reasoner),
)

(
    min_cardinality_unnc_1_retrieval_results,
    min_cardinality_unnc_2_retrieval_results,
    min_cardinality_unnc_3_retrieval_results,
) = (
    retrieval_eval(
        expressions=expressions,
        y=concept_to_retrieval(expressions, symbolic_kb),
        yhat=concept_to_retrieval(expressions, neural_owl_reasoner),
    )
    for expressions in [
    min_cardinality_unnc_1,
    min_cardinality_unnc_2,
    min_cardinality_unnc_3,
]
)

(
    max_cardinality_unnc_1_retrieval_results,
    max_cardinality_unnc_2_retrieval_results,
    max_cardinality_unnc_3_retrieval_results,
) = (
    retrieval_eval(
        expressions=expressions,
        y=concept_to_retrieval(expressions, symbolic_kb),
        yhat=concept_to_retrieval(expressions, neural_owl_reasoner),
    )
    for expressions in [
    max_cardinality_unnc_1,
    max_cardinality_unnc_2,
    max_cardinality_unnc_3,
]
)

results = {
    "nc_retrieval_results": nc_retrieval_results,
    "unions_nc_retrieval_results": unions_nc_retrieval_results,
    "intersections_nc_retrieval_results": intersections_nc_retrieval_results,
    "nnc_retrieval_results": nnc_retrieval_results,
    "unnc_retrieval_results": unnc_retrieval_results,
    "unions_unnc_retrieval_results": unions_unnc_retrieval_results,
    "intersections_unnc_retrieval_results": intersections_unnc_retrieval_results,
    "exist_unnc_retrieval_results": exist_unnc_retrieval_results,
    "for_all_unnc_retrieval_results": for_all_unnc_retrieval_results,
    "min_cardinality_unnc_1_retrieval_results": min_cardinality_unnc_1_retrieval_results,
    "min_cardinality_unnc_2_retrieval_results": min_cardinality_unnc_2_retrieval_results,
    "min_cardinality_unnc_3_retrieval_results": min_cardinality_unnc_3_retrieval_results,
    "max_cardinality_unnc_1_retrieval_results": max_cardinality_unnc_1_retrieval_results,
    "max_cardinality_unnc_2_retrieval_results": max_cardinality_unnc_2_retrieval_results,
    "max_cardinality_unnc_3_retrieval_results": max_cardinality_unnc_3_retrieval_results,
}


# logger that prints the results
def print_results(results):
    print(f"Number of named and negated named concepts: {len(unnc)}")
    print(
        f"Number of object properties and their inverses: {len(object_properties_and_inverse)}"
    )
    print("\n")
    print("(Number of Concepts, Jaccard Similarity, Runtime Benefits)")
    for k, v in results.items():
        print("\n")
        print(f"{k}:")
        print(v)


print_results(results)
"""
