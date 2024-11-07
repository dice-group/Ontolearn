import unittest
import random
import itertools

from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner

from owlapy.owl_property import OWLObjectProperty, OWLProperty
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.class_expression import *
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.utils import jaccard_similarity
from owlapy.class_expression import (
    OWLClass,
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
)
import time
from typing import List, Tuple, Set
from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
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


class TestNeuralRetrieval:
    def __init__(self):
        self.neural_owl_reasoner = TripleStoreNeuralReasoner(path_of_kb="KGs/Family/father.owl", gamma=0.8)

    def test_retrieval_single_individual_father_owl(self):
        triples_about_anna = {(s, p, o) for s, p, o in self.neural_owl_reasoner.abox("http://example.com/father#anna")}

        sanity_checking = {
            (OWLNamedIndividual("http://example.com/father#anna"),
             OWLObjectProperty("http://example.com/father#hasChild"),
             OWLNamedIndividual("http://example.com/father#heinz")),
            (OWLNamedIndividual("http://example.com/father#anna"),
             OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
             OWLClass("http://example.com/father#female")),
            (OWLNamedIndividual("http://example.com/father#anna"),
             OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
             OWLClass("http://www.w3.org/2002/07/owl#NamedIndividual"))}

        assert triples_about_anna == sanity_checking

    def test_retrieval_named_concepts_family(self):
        symbolic_kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        benchmark_dataset = [(concept,{individual.str for individual in symbolic_kb.individuals(concept)})
                             for concept in symbolic_kb.get_concepts()]

        neural_owl_reasoner = TripleStoreNeuralReasoner(path_of_kb="KGs/Family/family-benchmark_rich_background.owl",
                                                        gamma=0.1)

        avg_jaccard_index = 0
        for concept, symbolic_retrieval in benchmark_dataset:
            neural_retrieval = {i.str for i in neural_owl_reasoner.instances(concept)}
            v = jaccard_similarity(symbolic_retrieval, neural_retrieval)
            assert v == 1.0 or v == 0.0
            avg_jaccard_index += v

        assert avg_jaccard_index / len(benchmark_dataset) == 1.0

    def test_de_morgan_male_and_father_father(self):
        male_and_father = OWLObjectIntersectionOf(
            [
                OWLClass("http://example.com/father#male"),
                OWLObjectMinCardinality(
                    cardinality=1,
                    property=OWLObjectProperty("http://example.com/father#hasChild"),
                    filler=OWLObjectOneOf(
                        OWLNamedIndividual("http://example.com/father#anna")
                    ))])
        individuals = set(self.neural_owl_reasoner.instances(male_and_father))
        not_female_or_not_mother = OWLObjectComplementOf(
            OWLObjectUnionOf(
                [
                    OWLClass("http://example.com/father#female"),
                    OWLObjectComplementOf(
                        OWLObjectMinCardinality(
                            cardinality=1,
                            property=OWLObjectProperty(
                                "http://example.com/father#hasChild"
                            ),
                            filler=OWLObjectOneOf(
                                OWLNamedIndividual("http://example.com/father#anna")
                            ),
                        )
                    ),
                ]
            )
        )
        print(individuals)
        individuals_2 = set(self.neural_owl_reasoner.instances(not_female_or_not_mother))
        assert individuals == individuals_2

    def test_de_morgan_male_and_has_daughter_family(self):

        prefix = "http://www.benchmark.org/family#"
        male_and_has_daughter = OWLObjectIntersectionOf(
            [
                OWLClass(prefix + "Male"),
                OWLObjectMinCardinality(
                    cardinality=1,
                    property=OWLObjectProperty(prefix + "hasChild"),
                    filler=OWLClass(prefix + "Female"),
                ),
            ]
        )
        individuals = set(self.neural_owl_reasoner.instances(male_and_has_daughter))
        not_male_or_not_has_daughter = OWLObjectComplementOf(
            OWLObjectUnionOf(
                [
                    OWLClass(prefix + "Female"),
                    OWLObjectComplementOf(
                        OWLObjectMinCardinality(
                            cardinality=1,
                            property=OWLObjectProperty(prefix + "hasChild"),
                            filler=OWLClass(prefix + "Female"),
                        )
                    ),
                ]
            )
        )
        individuals_2 = set(self.neural_owl_reasoner.instances(not_male_or_not_has_daughter))
        assert individuals == individuals_2

    def test_de_morgan_not_father_or_brother_family(self):
        prefix = "http://www.benchmark.org/family#"
        not_father_or_brother = OWLObjectComplementOf(
            OWLObjectUnionOf(
                [
                    OWLClass(prefix + "Father"),
                    OWLClass(prefix + "Brother"),
                ]
            )
        )
        individuals = set(self.neural_owl_reasoner.instances(not_father_or_brother))
        father_and_brother = OWLObjectIntersectionOf(
            [
                OWLObjectComplementOf(OWLClass(prefix + "Father")),
                OWLObjectComplementOf(OWLClass(prefix + "Brother")),
            ]
        )
        individuals_2 = set(self.neural_owl_reasoner.instances(father_and_brother))
        assert individuals == individuals_2

    def test_complement_for_all_named_concepts_family(self):
        symbolic_kb = KnowledgeBase(
            path="KGs/Family/family-benchmark_rich_background.owl"
        )
        named_concepts = symbolic_kb.get_concepts()
        benchmark_dataset = [
            (
                OWLObjectComplementOf(concept),
                {
                    individual.str
                    for individual in symbolic_kb.individuals(
                    OWLObjectComplementOf(concept)
                )
                },
            )
            for concept in named_concepts
        ]
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.1
        )
        for concept, symbolic_retrieval in benchmark_dataset:
            neural_retrieval = {i.str for i in neural_owl_reasoner.instances(concept)}
            assert jaccard_similarity(symbolic_retrieval, neural_retrieval)

    def test_retrieval_named_concepts_in_abox_family(self):
        symbolic_kb = KnowledgeBase(
            path="KGs/Family/family-benchmark_rich_background.owl"
        )
        named_concepts_having_at_least_single_indv = [
            i
            for i in symbolic_kb.get_concepts()
            if i.str
               not in [
                   "http://www.benchmark.org/family#PersonWithASibling",
                   "http://www.benchmark.org/family#Child",
                   "http://www.benchmark.org/family#Parent",
                   "http://www.benchmark.org/family#Grandparent",
                   "http://www.benchmark.org/family#Grandchild",
               ]
        ]
        benchmark_dataset_named = [
            (
                concept,
                {individual.str for individual in symbolic_kb.individuals(concept)},
            )
            for concept in named_concepts_having_at_least_single_indv
        ]

        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.1
        )
        for concept, symbolic_retrieval in benchmark_dataset_named:
            neural_retrieval = {i.str for i in neural_owl_reasoner.instances(concept)}
            assert jaccard_similarity(symbolic_retrieval, neural_retrieval)

    def test_negated_retrieval_named_concepts_in_abox_family(self):
        symbolic_kb = KnowledgeBase(
            path="KGs/Family/family-benchmark_rich_background.owl"
        )
        named_concepts_having_at_least_single_indv = [
            i.get_object_complement_of()
            for i in symbolic_kb.get_concepts()
            if i.str
               not in [
                   "http://www.benchmark.org/family#PersonWithASibling",
                   "http://www.benchmark.org/family#Child",
                   "http://www.benchmark.org/family#Parent",
                   "http://www.benchmark.org/family#Grandparent",
                   "http://www.benchmark.org/family#Grandchild",
               ]
        ]
        benchmark_dataset_named = [
            (
                concept,
                {individual.str for individual in symbolic_kb.individuals(concept)},
            )
            for concept in named_concepts_having_at_least_single_indv
        ]

        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.1
        )
        for concept, symbolic_retrieval in benchmark_dataset_named:
            neural_retrieval = {i.str for i in neural_owl_reasoner.instances(concept)}
            assert jaccard_similarity(symbolic_retrieval, neural_retrieval)

    def test_regression_alcq_family(self):

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

        def concept_retrieval(retriever_func, c) -> Tuple[Set[str], float]:
            start_time = time.time()
            return {i.str for i in retriever_func.individuals(c)}, time.time() - start_time

        # (1) Initialize knowledge base.
        symbolic_kb = KnowledgeBase(path="KGs/Family/family-benchmark_rich_background.owl")
        # symbolic_kb = TripleStore(url="http://localhost:3030/family")
        # (2) Initialize Neural OWL Reasoner.
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.8
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
            concept_reducer_properties(concepts=unnc, properties=object_properties_and_inverse,
                                       cls=OWLObjectMinCardinality,
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

        ## RETRIEVAL RESULTS

        data = []

        for i in chain(nc, unions, intersections,
                       nnc, unnc, unions_unnc, intersections_unnc,
                       exist_unnc, for_all_unnc,
                       min_cardinality_unnc_1, min_cardinality_unnc_2, min_cardinality_unnc_3,
                       max_cardinality_unnc_1, max_cardinality_unnc_2, max_cardinality_unnc_3):
            retrieval_y, runtime_y = concept_retrieval(symbolic_kb, i)
            retrieval_neural_y, runtime_neural_y = concept_retrieval(neural_owl_reasoner, i)
            jaccard_sim = jaccard_similarity(retrieval_y, retrieval_neural_y)
            data.append({"Expression": owl_expression_to_dl(i),
                         "Type": type(i).__name__,
                         "Jaccard Similarity": jaccard_sim,
                         "Runtime Benefits": runtime_y - runtime_neural_y
                         })

        df = pd.DataFrame(data)
        assert df["Jaccard Similarity"].mean() == 1.0