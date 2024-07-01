import unittest
import random
import itertools

from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner

from owlapy.owl_property import OWLObjectProperty, OWLProperty
from owlapy.class_expression import OWLClass
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.class_expression import *

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.utils import jaccard_similarity


class Test_Neural_Retrieval:

    def test_retrieval_single_individual_father_owl(self):
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/father.owl", gamma=0.8
        )
        triples_about_anna = {
            (s, p, o)
            for s, p, o in neural_owl_reasoner.abox("http://example.com/father#anna")
        }

        sanity_checking = {
            (
                OWLNamedIndividual("http://example.com/father#anna"),
                OWLObjectProperty("http://example.com/father#hasChild"),
                OWLNamedIndividual("http://example.com/father#heinz"),
            ),
            (
                OWLNamedIndividual("http://example.com/father#anna"),
                OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                OWLClass("http://example.com/father#female"),
            ),
            (
                OWLNamedIndividual("http://example.com/father#anna"),
                OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                OWLClass("http://www.w3.org/2002/07/owl#NamedIndividual"),
            ),
        }

        assert triples_about_anna == sanity_checking

    def test_retrieval_named_concepts_family(self):
        symbolic_kb = KnowledgeBase(
            path="KGs/Family/family-benchmark_rich_background.owl"
        )
        benchmark_dataset = [
            (
                concept,
                {individual.str for individual in symbolic_kb.individuals(concept)},
            )
            for concept in symbolic_kb.get_concepts()
        ]

        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.1
        )

        avg_jaccard_index = 0
        for concept, symbolic_retrieval in benchmark_dataset:
            neural_retrieval = {i.str for i in neural_owl_reasoner.instances(concept)}
            v = jaccard_similarity(symbolic_retrieval, neural_retrieval)
            assert v == 1.0 or v == 0.0
            avg_jaccard_index += v

        assert 0.73 > avg_jaccard_index / len(benchmark_dataset) >= 0.72

    def test_de_morgan_male_and_father_father(self):
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/father.owl", gamma=0.8
        )

        male_and_father = OWLObjectIntersectionOf(
            [
                OWLClass("http://example.com/father#male"),
                OWLObjectMinCardinality(
                    cardinality=1,
                    property=OWLObjectProperty("http://example.com/father#hasChild"),
                    filler=OWLNamedIndividual("http://example.com/father#anna"),
                ),
            ]
        )
        individuals = set(neural_owl_reasoner.instances(male_and_father))
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
        individuals_2 = set(neural_owl_reasoner.instances(not_female_or_not_mother))
        assert individuals == individuals_2

    def test_de_morgan_male_and_has_daughter_family(self):
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.8
        )
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
        individuals = set(neural_owl_reasoner.instances(male_and_has_daughter))
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
        individuals_2 = set(neural_owl_reasoner.instances(not_male_or_not_has_daughter))
        assert individuals == individuals_2

    def test_de_morgan_not_father_or_brother_family(self):
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.8
        )
        prefix = "http://www.benchmark.org/family#"
        not_father_or_brother = OWLObjectComplementOf(
            OWLObjectUnionOf(
                [
                    OWLClass(prefix + "Father"),
                    OWLClass(prefix + "Brother"),
                ]
            )
        )
        individuals = set(neural_owl_reasoner.instances(not_father_or_brother))
        father_and_brother = OWLObjectIntersectionOf(
            [
                OWLObjectComplementOf(OWLClass(prefix + "Father")),
                OWLObjectComplementOf(OWLClass(prefix + "Brother")),
            ]
        )
        individuals_2 = set(neural_owl_reasoner.instances(father_and_brother))
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

    def test_for_all_family(self):
        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.1
        )
        symbolic_kb = TripleStore(url="http://localhost:3030/family")
        named_classes = symbolic_kb.get_concepts()  # Retrieve all named classes
        properties = symbolic_kb.get_object_properties()  # Retrieve all properties

        with open("match_log.txt", "w") as file:
            file.write("Matches found in the neural reasoner evaluations:\n")

            # Iterate over all properties and named classes
            for property in properties:
                for named_class in named_classes:
                    # Creating the expressions for ¬∃r.¬C and ∀r.C using property.str
                    neg_exist_neg = OWLObjectComplementOf(
                        OWLObjectSomeValuesFrom(
                            property=OWLObjectProperty(property.str),
                            filler=OWLObjectComplementOf(named_class),
                        )
                    )
                    all_values = OWLObjectAllValuesFrom(
                        property=OWLObjectProperty(property.str), filler=named_class
                    )

                    # Using the neural reasoner to get instances
                    neural_neg_exist_neg = {
                        i.str for i in neural_owl_reasoner.instances(neg_exist_neg)
                    }
                    neural_all_values = {
                        i.str for i in neural_owl_reasoner.instances(all_values)
                    }

                    # Check for matches and write to file if equal
                    if neural_neg_exist_neg == neural_all_values:
                        file.write(
                            f"Property: {property.str}, Class: {named_class.str}\n"
                        )
                        file.write(
                            f"¬∃{property.str}.¬{named_class.str} and ∀{property.str}.{named_class.str} match\n"
                        )
                        file.write(f"Matching Instances: {neural_neg_exist_neg}\n\n")

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

    """
    def test_regression_named_concepts_owl(self):
        symbolic_kb = KnowledgeBase(path="KGs/Family/father.owl")
        benchmark_dataset = [
            (
                concept,
                {individual.str for individual in symbolic_kb.individuals(concept)},
            )
            for concept in symbolic_kb.get_concepts()
        ]

        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/father.owl", gamma=0.8
        )

        avg_jaccard_index = 0
        for concept, symbolic_retrieval in benchmark_dataset:
            neural_retrieval = {i.str for i in neural_owl_reasoner.instances(concept)}
            v = jaccard_similarity(symbolic_retrieval, neural_retrieval)
            avg_jaccard_index += v
        assert avg_jaccard_index / len(benchmark_dataset) >= 0.66

    def compute_avg_jaccard_index(self, benchmark_dataset, neural_owl_reasoner):
        avg_jaccard_index = 0
        for expression, benchmark_retrieval in benchmark_dataset:
            neural_retrieval = {
                i.str for i in neural_owl_reasoner.instances(expression)
            }
            v = jaccard_similarity(benchmark_retrieval, neural_retrieval)
            avg_jaccard_index += v
        avg_jaccard_index = avg_jaccard_index / len(benchmark_dataset)
        print(avg_jaccard_index)
        return avg_jaccard_index

    def test_object_intersection_2_classes(self):
        print("Running test_object_intersection_2_classes")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        if len(classes) < 2:
            assert True
        pairs = list(itertools.combinations(classes, 2))
        benchmark_dataset = [
            (
                OWLObjectIntersectionOf([class1, class2]),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectIntersectionOf([class1, class2])
                    )
                },
            )
            for class1, class2 in pairs
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    def test_object_union_2_classes(self):
        print("Running test_object_union_2_classes")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        if len(classes) < 2:
            assert True
        pairs = list(itertools.combinations(classes, 2))
        benchmark_dataset = [
            (
                OWLObjectUnionOf([class1, class2]),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectUnionOf([class1, class2])
                    )
                },
            )
            for class1, class2 in pairs
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    def test_object_union_3_classes(self):
        print("Running test_object_union_3_classes")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        if len(classes) < 3:
            assert True
        pairs = list(itertools.combinations(classes, 3))
        benchmark_dataset = [
            (
                OWLObjectUnionOf([class1, class2, class3]),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectUnionOf([class1, class2, class3])
                    )
                },
            )
            for class1, class2, class3 in pairs
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    def test_object_complement(self):
        print("Running test_object_complement")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        if len(classes) < 1:
            assert True
        benchmark_dataset = [
            (
                OWLObjectComplementOf(class1),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectComplementOf(class1)
                    )
                },
            )
            for class1 in classes
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    # NOTE: Might take too long to run on bigger KGs
    def test_object_some_values_from(self):
        print("Running test_object_some_values_from")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        if len(classes) < 1:
            assert True
        object_properties = list(benchmark_kb.get_object_properties())
        pairs = list(itertools.product(classes, object_properties))
        benchmark_dataset = [
            (
                OWLObjectSomeValuesFrom(object_property, class1),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectSomeValuesFrom(object_property, class1)
                    )
                },
            )
            for class1, object_property in pairs
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    def test_object_all_values_from(self):
        print("Running test_object_all_values_from")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        if len(classes) < 1:
            assert True
        object_properties = list(benchmark_kb.get_object_properties())
        pairs = list(itertools.product(classes, object_properties))
        benchmark_dataset = [
            (
                OWLObjectAllValuesFrom(object_property, class1),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectAllValuesFrom(object_property, class1)
                    )
                },
            )
            for class1, object_property in pairs
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    def test_object_min_cardinality_2(self):
        print("Running test_object_min_cardinality")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        if len(classes) < 1:
            assert True
        object_properties = list(benchmark_kb.get_object_properties())
        pairs = list(itertools.product(classes, object_properties))
        benchmark_dataset = [
            (
                OWLObjectMinCardinality(2, object_property, class1),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectMinCardinality(2, object_property, class1)
                    )
                },
            )
            for class1, object_property in pairs
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    def test_object_max_cardinality_1(self):
        print("Running test_object_max_cardinality")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        if len(classes) < 1:
            assert True
        object_properties = list(benchmark_kb.get_object_properties())
        pairs = list(itertools.product(classes, object_properties))
        benchmark_dataset = [
            (
                OWLObjectMaxCardinality(1, object_property, class1),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectMaxCardinality(1, object_property, class1)
                    )
                },
            )
            for class1, object_property in pairs
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    def test_object_one_of_5(self):
        print("Running test_object_one_of")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        individuals = list(benchmark_kb.individuals())
        combinations = list(itertools.combinations(individuals, 3))
        benchmark_dataset = [
            (
                OWLObjectOneOf(*[combination]),
                {
                    individual.str
                    for individual in benchmark_kb.individuals(
                        OWLObjectOneOf(*[combination])
                    )
                },
            )
            for combination in combinations
        ]
        assert (
            self.compute_avg_jaccard_index(benchmark_dataset, neural_owl_reasoner)
            >= 0.66
        )

    def test_individuals_in_signature(self):
        print("Running test_individuals_in_signature")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        neural_individuals = neural_owl_reasoner.individuals_in_signature()
        benchmark_dataset = set(benchmark_kb.individuals())
        print(
            j_similarity := jaccard_similarity(
                benchmark_dataset, set(neural_individuals)
            )
        )

        assert j_similarity >= 0.66

    def test_classes_in_signature(self):
        print("Running test_classes_in_signature")
        benchmark_kb, classes, neural_owl_reasoner = self.setup_method()
        neural_classes = neural_owl_reasoner.classes_in_signature()
        print(j_similarity := jaccard_similarity(set(classes), set(neural_classes)))

        assert j_similarity >= 0.66

    def setup_method(self):
        try:
            benchmark_kb = TripleStore(url="localhost:3030/family-benchmark")
            classes = list(benchmark_kb.get_classes_in_signature())

        except:
            benchmark_kb = KnowledgeBase(
                path="KGs/Family/family-benchmark_rich_background.owl"
            )
            classes = list(benchmark_kb.get_classes_in_signature())

        neural_owl_reasoner = TripleStoreNeuralReasoner(
            path_of_kb="KGs/Family/family-benchmark_rich_background.owl", gamma=0.8
        )
        return benchmark_kb, classes, neural_owl_reasoner

        """


"""
if __name__ == "__main__":
    unittest.main()
"""
