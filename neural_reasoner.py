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




def evaluate_reasoner(kb, reasoner, triple_store, concept_type):

     class_expressions, length = generate_class_expressions(kb, concept_type)
    #  print(class_expressions)
    #  exit(0)

     jaccard_scores = []
     f1_scores = []
     start_time = time.time()
     for class_expr in class_expressions:

        # ground_truth = {i.str for i in kb.individuals(parser.parse_expression(class_expr))}

        
        ground_truth = {i.str for i in triple_store.instances(parser.parse_expression(class_expr))}

        # print(ground_truth)
        # exit(0)

        reasoner_instances = get_reasoner_instances(reasoner, parser.parse_expression(class_expr))

        jaccard_score = jaccard_similarity(ground_truth, reasoner_instances)
        f1_score = compute_f1_score(ground_truth, reasoner_instances)
        jaccard_scores.append(jaccard_score)
        f1_scores.append(f1_score)

        #   if jaccard_score == 0:
        print(f"Class Expression: {parser.parse_expression(class_expr)}")
        exit(0)
        # print(f"Ground Truth: {ground_truth}")
        # print(f"Reasoner Instances: {reasoner_instances}")
        print(f"Jaccard Similarity: {jaccard_score}")
        # print("---------------------------------------------------")

     end_time = time.time()
     duration = end_time-start_time

     avg_jaccard_similarity = sum(jaccard_scores) / len(jaccard_scores)
     avg_f1_score = sum(f1_scores)/ len(f1_scores)

     print(f"Average Jaccard Similarity: {avg_jaccard_similarity}")
     print(f"Elapsed time: {duration} seconds")

     return avg_jaccard_similarity, avg_f1_score ,length, duration
     # print(jaccard_scores)

TS = TripleStoreReasonerOntology(url="http://localhost:3030/family.owl")
reasoner = NeuralReasoner(KGE(f"KeciFamilyRun"))

# evaluate_reasoner(kb, reasoner, TS, concept_type="exist")


#########################################################################################
# reasoner_instances = get_reasoner_instances(reasoner,OWLObjectComplementOf(OWLClass(IRI('http://www.benchmark.org/family#','Person'))))


class_expr_1 =  OWLObjectMinCardinality(property=OWLObjectProperty(IRI('http://www.benchmark.org/family#','hasChild')),cardinality = 1,filler=OWLClass(IRI('http://www.benchmark.org/family#','Brother')))
class_expr_2 =  OWLObjectSomeValuesFrom(property=OWLObjectProperty(IRI('http://www.benchmark.org/family#','hasChild')),filler=OWLClass(IRI('http://www.benchmark.org/family#','Brother')))

reasoner_instances_1 = get_reasoner_instances(reasoner, class_expr_1)
reasoner_instances_2 = get_reasoner_instances(reasoner, class_expr_2)

ground_truth_1 = {i.str for i in kb.individuals(class_expr_1)}
ground_truth_2 = {i.str for i in kb.individuals(class_expr_2)}
# ground_truth_1 = {i.str for i in TS.instances(class_expr_1)}
# ground_truth_2 = {i.str for i in TS.instances(class_expr_2)}

assert ground_truth_1 == ground_truth_2

print(len(reasoner_instances_2))
print(len(reasoner_instances_1))

assert reasoner_instances_1 == reasoner_instances_2, "Reasoner fails"

# print("hey"*100)
# print(ground_truth)
# print(len(ground_truth))


 
#, "negated",  "intersect", "union", "exist", "universal", "All"

################################################################################

# concept_types = ["name", "nega",  "intersect", "union", "exist", "universal", "min_card", "max_card", "exact_card",\
#                   "exist_inv", "universal_inv", "min_card_inv", "max_card_inv", "All"]

# concept_types = ["exist"]
# data = {}

# for concept_type in concept_types:

#      jacc, f1_score, length, running_time = evaluate_reasoner(kb, reasoner, TS, concept_type= concept_type)
#      data.update({concept_type: [jacc, length, f1_score, running_time]})

# print(data)
# # Save to a JSON file
# with open('data.json', 'w') as json_file:
#     json.dump(data, json_file, indent=4)



# embedding_dim = [16, 32, 64, 128, 256, 512, 1024]
# data = {}

# for dim in embedding_dim:

#      reasoner = NeuralReasoner(KGE(f"KeciFamilyRun_{dim}"))
#      data.update({dim: evaluate_reasoner(kb, reasoner, TS)})

# print(data)





# (2) Send (1) into Triplestore Class
# kb = TripleStore(reasoner=NeuralReasoner(KGE("KeciFatherRun")))

# # (2) Initialize a learner.
# model = Drill(knowledge_base=kb, use_data_properties=False, use_inverse=False)

# # (3) Define a description logic concept learning problem.
# lp = PosNegLPStandard(pos={OWLNamedIndividual("http://example.com/father#anna")},
#                       neg={OWLNamedIndividual("http://example.com/father#stefan"),
#                            OWLNamedIndividual("http://example.com/father#heinz"),
#                            OWLNamedIndividual("http://example.com/father#michelle")})

# # (4) Learn description logic concepts best fitting (3). 
# h = model.fit(learning_problem=lp).best_hypotheses()
# print(h)
# print(owl_expression_to_dl(h))
# print(owl_expression_to_sparql(expression=h))
