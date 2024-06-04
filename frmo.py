# To train a neural link predictor
# dicee --path_single_kg "KGs/Family/father.owl" --model Keci --p 0 --q 1 --path_to_store_single_run "KeciFatherRun" --backend rdflib --eval_model None --embedding_dim 128
from dicee import KGE
from ontolearn.triple_store import TripleStore
from neural_louis import *
import torch

from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl



dat=build_data(KGE("KeciFatherRun"), "/home/dice/Desktop/Ontolearn/KeciFatherRun/train_set.npy",\
                                          "/home/dice/Desktop/Ontolearn/KeciFatherRun/Keci_entity_embeddings.csv"\
                                             , "/home/dice/Desktop/Ontolearn/KeciFatherRun/Keci_relation_embeddings.csv")


data = torch.tensor(dat.embed_data())

input_size = data.size(1)
hidden_size, output_size, lr, num_epochs = 16, input_size, .01, 100

reasoner = NeuralReasoner(input_size, hidden_size, output_size, lr, num_epochs)

reasoner.trainer(data=data)

# print(data.size(1))



# # (2) Send (1) into Triplestore Class
# kb = TripleStore(reasoner=NeuralReasoner(KGE("KeciFatherRun"), "/home/dice/Desktop/Ontolearn/KeciFatherRun/train_set.npy",\
#                                           "/home/dice/Desktop/Ontolearn/KeciFatherRun/Keci_entity_embeddings.csv", "/home/dice/Desktop/Ontolearn/KeciFatherRun/Keci_relation_embeddings.csv"))
# # (2) Initialize a learner.
# model = TDL(knowledge_base=kb)
# # (3) Define a description logic concept learning problem.
# lp = PosNegLPStandard(pos={OWLNamedIndividual("http://example.com/father#stefan")},
#                       neg={OWLNamedIndividual("http://example.com/father#heinz"),
#                            OWLNamedIndividual("http://example.com/father#anna"),
#                            OWLNamedIndividual("http://example.com/father#michelle")})
# # (4) Learn description logic concepts best fitting (3).
# h = model.fit(learning_problem=lp).best_hypotheses()
# print(h)
# print(owl_expression_to_dl(h))
# print(owl_expression_to_sparql(expression=h))


