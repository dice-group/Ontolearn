# To train a neural link predictor
# dicee --path_single_kg "KGs/Family/father.owl" --model Keci --p 0 --q 1 --path_to_store_single_run "KeciFatherRun" --backend rdflib --eval_model None --embedding_dim 128
# dicee --path_single_kg "KGs/Family/family-benchmark_rich_background.owl" --model Keci --p 0 --q 1 --path_to_store_single_run "KeciFamilyRun" --backend rdflib --eval_model None --embedding_dim 256 --scoring_technique AllvsAll
from dicee import KGE
from ontolearn.triple_store import TripleStore, NeuralReasoner, TripleStoreReasonerOntology
from owlapy.class_expression import OWLClassExpression
from typing import List, Set
from ontolearn.learners import Drill, TDL
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
from owlapy.class_expression import OWLClass
from generate_valid_class_expression import *
from ontolearn.utils import jaccard_similarity, compute_f1_score
from owlapy.parser import DLSyntaxParser
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.class_expression import OWLClassExpression, OWLThing, OWLClass, OWLObjectSomeValuesFrom, OWLObjectOneOf, \
    OWLObjectMinCardinality, OWLDataSomeValuesFrom, OWLDataOneOf, OWLObjectComplementOf, OWLObjectUnionOf, \
    OWLObjectIntersectionOf, OWLObjectAllValuesFrom, OWLObjectMaxCardinality
from owlapy.owl_property import OWLDataProperty, OWLObjectPropertyExpression, OWLObjectInverseOf, OWLObjectProperty, \
    OWLProperty
from owlapy.iri import IRI
import time
import json


kb = KnowledgeBase(path="/home/dice/Desktop/Ontolearn/KGs/Family/family-benchmark_rich_background.owl")
# kb = KnowledgeBase(path="/home/dice/Desktop/Ontolearn/KGs/Family/father.owl")

namespace = list(kb.ontology.classes_in_signature())[0].iri.get_namespace()

parser = DLSyntaxParser(namespace) 

def get_reasoner_instances(reasoner: NeuralReasoner, class_expression: OWLClassExpression) -> Set[str]:
    """
    Get instances of a given class expression using our neural reasoner.
    :param reasoner: The NeuralReasoner instance.
    :param class_expression: The OWLClassExpression.
    :return: A set of instance URIs.
    """
    instances = set()
    for instance in reasoner.instances(class_expression):
        instances.add(instance.str)#instances.add(instance)
    return instances


TS = TripleStoreReasonerOntology(url="http://localhost:3030/family.owl")
reasoner = NeuralReasoner(KGE(f"KeciFamilyRun"))


#########################################################################################

# test min cardinality: ∃r.C = ≥ 1 r.C

#########################################################################################


class_expr_1 =  OWLObjectMinCardinality(property=OWLObjectProperty(IRI('http://www.benchmark.org/family#','hasChild')),cardinality = 1,filler=OWLClass(IRI('http://www.benchmark.org/family#','Brother')))
class_expr_2 =  OWLObjectSomeValuesFrom(property=OWLObjectProperty(IRI('http://www.benchmark.org/family#','hasChild')),filler=OWLClass(IRI('http://www.benchmark.org/family#','Brother')))

reasoner_instances_1 = get_reasoner_instances(reasoner, class_expr_1)
reasoner_instances_2 = get_reasoner_instances(reasoner, class_expr_2)

# ground_truth_1 = {i.str for i in kb.individuals(class_expr_1)}
# ground_truth_2 = {i.str for i in kb.individuals(class_expr_2)}
ground_truth_1 = {i.str for i in TS.instances(class_expr_1)}
ground_truth_2 = {i.str for i in TS.instances(class_expr_2)}

assert ground_truth_1 == ground_truth_2

print(len(ground_truth_1))
print(len(ground_truth_2))
print(len(reasoner_instances_2))
print(len(reasoner_instances_1))

assert reasoner_instances_1 == reasoner_instances_2, "Reasoner fails"

#########################################################################################

# test max cardunality: ∀r.C = ≤ 1 r.C

#########################################################################################
# OWLObjectMaxCardinality(property=OWLObjectProperty(IRI('http://www.benchmark.org/family#','hasChild')),1,filler=OWLObjectComplementOf(OWLClass(IRI('http://www.benchmark.org/family#','Brother'))))

class_expr_1 =  OWLObjectMaxCardinality(property=OWLObjectProperty(IRI('http://www.benchmark.org/family#','hasChild')),cardinality = 1,filler=OWLObjectComplementOf(OWLClass(IRI('http://www.benchmark.org/family#','Brother'))))

class_expr_2 =  OWLObjectAllValuesFrom(property=OWLObjectProperty(IRI('http://www.benchmark.org/family#','hasChild')),filler=OWLObjectComplementOf(OWLClass(IRI('http://www.benchmark.org/family#','Brother'))))

reasoner_instances_1 = get_reasoner_instances(reasoner, class_expr_1)
reasoner_instances_2 = get_reasoner_instances(reasoner, class_expr_2)

ground_truth_1 = {i.str for i in kb.individuals(class_expr_1)}
ground_truth_2 = {i.str for i in kb.individuals(class_expr_2)}
# ground_truth_1 = {i.str for i in TS.instances(class_expr_1)}
# ground_truth_2 = {i.str for i in TS.instances(class_expr_2)}

# assert ground_truth_1 == ground_truth_2

print(len(ground_truth_1))
print(len(ground_truth_2))
print(len(reasoner_instances_2))
print(len(reasoner_instances_1))

assert reasoner_instances_1 == reasoner_instances_2, "Reasoner fails"



