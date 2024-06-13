# To train a neural link predictor
# dicee --path_single_kg "KGs/Family/father.owl" --model Keci --p 0 --q 1 --path_to_store_single_run "KeciFatherRun" --backend rdflib --eval_model None --embedding_dim 128
from dicee import KGE
from ontolearn.triple_store import TripleStore, NeuralReasoner
from owlapy.class_expression import OWLClassExpression
from typing import List, Set
from ontolearn.learners import Drill, TDL
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
from owlapy.class_expression import OWLClass
from generate_valid_class_expression import *
from ontolearn.utils import jaccard_similarity
from owlapy.parser import DLSyntaxParser
from ontolearn.knowledge_base import KnowledgeBase



# kb = KnowledgeBase(path="/home/dice/Desktop/Ontolearn/KGs/Family/family-benchmark_rich_background.owl")
kb = KnowledgeBase(path="/home/dice/Desktop/Ontolearn/KGs/Family/father.owl")

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
        instances.add(instance.str)
    return instances



def evaluate_reasoner(kb, reasoner):

     class_expressions = generate_class_expressions(kb)

     jaccard_scores = []

     for class_expr in class_expressions:

          ground_truth = set(kb.individuals(parser.parse_expression(class_expr)))
          reasoner_instances = get_reasoner_instances(reasoner, parser.parse_expression(class_expr))

          jaccard_score = jaccard_similarity(ground_truth, reasoner_instances)
          jaccard_scores.append(jaccard_score)

          # print(f"Class Expression: {class_expr}")
          # print("\n here: ")
          # print(f"Ground Truth: {ground_truth}")
          print(f"Reasoner Instances: {reasoner_instances}")
          # print(f"Jaccard Similarity: {jaccard_score}")
          print("---------------------------------------------------")

     avg_jaccard_similarity = sum(jaccard_scores) / len(jaccard_scores)
     print(f"Average Jaccard Similarity: {avg_jaccard_similarity}")



reasoner = NeuralReasoner(KGE("KeciFatherRun"))

# evaluate_reasoner(kb, reasoner)

# class_expressions = generate_class_expressions(kb)
# for class_exp in class_expressions:
#     print(class_exp)



# # (2) Send (1) into Triplestore Class
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
