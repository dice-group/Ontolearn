from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.utils import jaccard_similarity
from owlapy.class_expression import (
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
)
import time
from typing import List, Tuple, Set

symbolic_kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
# symbolic_kb = TripleStore(url="http://localhost:3030/family")
neural_owl_reasoner = TripleStoreNeuralReasoner(
    path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.8
)
object_properties = {i for i in symbolic_kb.get_object_properties()}
object_properties_inverse = {i.get_inverse_property() for i in object_properties}
object_properties_and_inverse = object_properties.union(object_properties_inverse)
# named concepts
nc = {i for i in symbolic_kb.get_concepts()}
# negated named concepts
nnc = {i.get_object_complement_of() for i in nc}
# union of named and negated named concepts
unnc = nc.union(nnc)


def concept_reducer(concepts, opt):
    result = set()
    for i in concepts:
        for j in concepts:
            result.add(opt((i, j)))
    return result


def concept_reducer_properties(concepts, opt, cardinality=2):
    result = set()
    for i in concepts:
        for j in object_properties_and_inverse:
            if opt == OWLObjectMinCardinality or opt == OWLObjectMaxCardinality:
                result.add(opt(cardinality, j, i))
                continue
            result.add(opt(j, i))
    return result


def concept_to_retrieval(concepts, retriever) -> List[Tuple[float, Set[str]]]:
    results = []
    for c in concepts:
        start_time_ = time.time()
        retrieval = {i.str for i in retriever.individuals(c)}
        results.append((time.time() - start_time_, retrieval))
    return results


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
        if verbose > 1:  # todo: set to 0
            print(
                f"Concept:{expressions}\tTrue Size:{len(y_i)}\tPredicted Size:{len(yhat_i)}\tRetrieval Similarity:{jaccard_sim}\tRuntime Benefit:{runtime_benefits:.3f}"
            )
        similarities.append(jaccard_sim)
        runtime_diff.append(runtime_benefits)
    avg_jaccard_sim = sum(similarities) / len(similarities)
    avg_runtime_benefits = sum(runtime_diff) / len(runtime_diff)
    return number_of_concepts, avg_jaccard_sim, avg_runtime_benefits


unions = concept_reducer(nc, opt=OWLObjectUnionOf)
intersections = concept_reducer(nc, opt=OWLObjectIntersectionOf)
unions_unnc = concept_reducer(unnc, opt=OWLObjectUnionOf)
intersections_unnc = concept_reducer(unnc, opt=OWLObjectIntersectionOf)
exist_unnc = concept_reducer_properties(unnc, opt=OWLObjectSomeValuesFrom)
for_all_unnc = concept_reducer_properties(unnc, opt=OWLObjectAllValuesFrom)

min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3 = (
    concept_reducer_properties(unnc, opt=OWLObjectMinCardinality, cardinality=i)
    for i in [1, 2, 3]
)
max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3 = (
    concept_reducer_properties(unnc, opt=OWLObjectMaxCardinality, cardinality=i)
    for i in [1, 2, 3]
)

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
    print("(Number of Concepts, Jackard Similarity, Runtime Benefits)")
    for k, v in results.items():
        print("\n")
        print(f"{k}:")
        print(v)


print_results(results)
