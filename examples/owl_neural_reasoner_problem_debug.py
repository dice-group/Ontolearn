from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.class_expression import (
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectOneOf,
    OWLClass,
)
from owlapy.owl_property import OWLObjectProperty

symbolic_kb = KnowledgeBase(path="KGs/Mutagenesis/mutagenesis_cleaned.owl")

neural_owl_reasoner = TripleStoreNeuralReasoner(
    path_neural_embedding="KGs_KeciMutagenesisRun/",
    gamma=0.9,
    inferred_owl_individuals={i.str for i in symbolic_kb.individuals()},
    inferred_object_properties={i.str for i in symbolic_kb.get_object_properties()},
    inferred_named_owl_classes={i.str for i in symbolic_kb.get_concepts()},
)


def get_concept_instances(reasoner, concept):
    if reasoner is symbolic_kb:
        print("Symbolic KB")
        return list(reasoner.reasoner.instances(concept))
    return list(reasoner.individuals(concept))


def Jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if len(s1.intersection(s2)) == 0 and len(s1.union(s2)) == 0:
        return 1
    if len(s1.union(s2)) == 0 or len(s1.intersection(s2)) == 0:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))


nc = {i for i in symbolic_kb.get_concepts()}

#  ∃ inBond.Bond-7
concept = OWLObjectSomeValuesFrom(
    property=OWLObjectProperty("http://dl-learner.org/mutagenesis#inBond"),
    filler=OWLClass("http://dl-learner.org/mutagenesis#Bond-7"),
)

print("Concept: ", concept)
instances_symbolic = get_concept_instances(symbolic_kb, concept)
instances_neural = get_concept_instances(neural_owl_reasoner, concept)
print("Instances Symbolic: ", len(instances_symbolic))
print("Instances Neural: ", len(instances_neural))
print("Jaccard Similarity: ", Jaccard_similarity(instances_symbolic, instances_neural))
