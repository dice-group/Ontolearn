from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.utils import jaccard_similarity
from owlapy.class_expression import OWLObjectUnionOf, OWLObjectIntersectionOf
import time
from typing import List, Tuple, Set

symbolic_kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
neural_owl_reasoner = TripleStoreNeuralReasoner(path_of_kb="KGs/Family/family-benchmark_rich_background.owl",
                                                gamma=0.1)
named_concepts = {i for i in symbolic_kb.get_concepts()}


def concept_reducer(concepts, opt):
    result = set()
    for i in concepts:
        for j in concepts:
            result.add(opt((i, j)))
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
    for expressions, y_report_i, yhat_report_i in zip(expressions, y, yhat):
        runtime_y_i, y_i = y_report_i
        runtime_yhat_i, yhat_i = yhat_report_i

        jaccard_sim = jaccard_similarity(y_i, yhat_i)
        runtime_benefits = runtime_y_i - runtime_yhat_i
        if verbose > 0:
            print(
                f"Concept:{expressions}\tTrue Size:{len(y_i)}\tPredicted Size:{len(yhat_i)}\tRetrieval Similarity:{jaccard_sim}\tRuntime Benefit:{runtime_benefits:.3f}")
        similarities.append(jaccard_sim)
        runtime_diff.append(runtime_benefits)
    avg_jaccard_sim = sum(similarities) / len(similarities)
    avg_runtime_benefits = sum(runtime_diff) / len(runtime_diff)
    return avg_jaccard_sim, avg_runtime_benefits


unions = concept_reducer(named_concepts, opt=OWLObjectUnionOf)
intersections = concept_reducer(named_concepts, opt=OWLObjectIntersectionOf)

named_concept_retrieval_results = retrieval_eval(expressions=named_concepts,
                                                 y=concept_to_retrieval(named_concepts, symbolic_kb),
                                                 yhat=concept_to_retrieval(named_concepts, neural_owl_reasoner))
unions_concept_retrieval_results = retrieval_eval(expressions=unions,
                                                  y=concept_to_retrieval(unions, symbolic_kb),
                                                  yhat=concept_to_retrieval(unions, neural_owl_reasoner))
intersections_concept_retrieval_results = retrieval_eval(expressions=intersections,
                                                         y=concept_to_retrieval(intersections, symbolic_kb),
                                                         yhat=concept_to_retrieval(intersections, neural_owl_reasoner))
print(named_concept_retrieval_results)
print(unions_concept_retrieval_results)
print(intersections_concept_retrieval_results)