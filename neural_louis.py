from dicee import KGE
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralReasoner(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, lr, num_epochs):
        super(NeuralReasoner, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def trainer(self, data):

        data = data.float()

        model = NeuralReasoner(self.input_size, self.hidden_size, self.output_size, self.lr, self.num_epochs)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            model.train()
            outs = model(data)
            loss = self.criterion(outs, data)  # Using data as its own target for autoencoding
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}') 

            



    

    def abox(self, str_iri: str = None):
        assert str_iri, f"{str_iri} cannot be None"
        index_str_iri=self.neural_link_predictor.entity_to_idx[str_iri]
        print(str_iri)
        print(index_str_iri)
        exit(1)
        # raise NotImplementedError("Not implemented yet")




class build_data:
    def __init__(self, neural_link_predictor:KGE, path_to_idx_train_data, path_entity_embeddings, path_rel_embeddings):

        self.neural_link_predictor = neural_link_predictor
        self.path_entity_embeddings = path_entity_embeddings
        self.path_rel_embeddings = path_rel_embeddings

        self.ent_emb = pd.read_csv(self.path_entity_embeddings, index_col=0) #all entities embeddings
        self.rel_emb = pd.read_csv(self.path_rel_embeddings, index_col=0) #all relation embeddings
        self.embedding_dim = self.ent_emb.shape[-1]

        assert self.embedding_dim == self.rel_emb.shape[-1]

        self.entity_to_idx = self.neural_link_predictor.entity_to_idx
        idx_to_entity = self.neural_link_predictor.idx_to_entity
        self.relation_to_idx = self.neural_link_predictor.relation_to_idx
        idx_to_relation = self.neural_link_predictor.idx_to_relations

        self.num_entities = len(self.entity_to_idx)
        self.num_relations = len(self.relation_to_idx)

        assert len(self.entity_to_idx) == len(idx_to_entity)
        assert len(self.relation_to_idx) == len(idx_to_relation)

        self.path_to_idx_train_data = path_to_idx_train_data 
        self.idx_train_data = np.load(path_to_idx_train_data) #triple indeces


    def embed_data(self):

        ent_keys = list(self.ent_emb.index) # convert the csv to dictionnary
        ent_values = self.ent_emb.values.tolist()
        ent_emb = dict(zip(ent_keys, ent_values))
        
        rel_keys = list(self.rel_emb.index)
        rel_values = self.rel_emb.values.tolist()
        rel_emb = dict(zip(rel_keys, rel_values))

        entities_emb_array = np.zeros((self.num_entities, self.embedding_dim))
        relations_emb_array = np.zeros((self.num_relations, self.embedding_dim))

        for entity, index in self.entity_to_idx.items():
            entities_emb_array[index] = ent_emb[entity]

        for relation, index in self.relation_to_idx.items():
            relations_emb_array[index] = rel_emb[relation]

        h_indices = self.idx_train_data[:, 0]
        rel_indices = self.idx_train_data[:, 1]
        t_indices = self.idx_train_data[:, 2]

        h_embeddings = entities_emb_array[h_indices]
        rel_embeddings = relations_emb_array[rel_indices]
        t_embeddings = entities_emb_array[t_indices]

        data = np.concatenate([h_embeddings, rel_embeddings, t_embeddings], axis=1)

        return data


    
    
   

        


    # def temp_abox(self, str_iri: str) -> Generator[Tuple[
    #     Tuple[OWLNamedIndividual, OWLProperty, OWLClass],
    #     Tuple[OWLObjectProperty, OWLObjectProperty, OWLNamedIndividual],
    #     Tuple[OWLObjectProperty, OWLDataProperty, OWLLiteral]], None, None]:

    #     sparql_query = f"SELECT DISTINCT ?p ?o WHERE {{ <{str_iri}> ?p ?o }}"
    #     subject_ = OWLNamedIndividual(str_iri)
    #     for binding in self.query(sparql_query).json()["results"]["bindings"]:
    #         p, o = binding["p"], binding["o"]
    #         # ORDER MATTERS
    #         if p["value"] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
    #             yield subject_, OWLProperty("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), OWLClass(o["value"])
    #         elif o["type"] == "uri":
    #             #################################################################
    #             # IMPORTANT
    #             # Can we assume that if o has URI and is not owl class, then o can be considered as an individual ?
    #             #################################################################
    #             yield subject_, OWLObjectProperty(p["value"]), OWLNamedIndividual(o["value"])
    #         elif o["type"] == "literal":
    #             if o["datatype"] == "http://www.w3.org/2001/XMLSchema#boolean":
    #                 yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=bool(o["value"]))
    #             elif o["datatype"] == "http://www.w3.org/2001/XMLSchema#double":
    #                 yield subject_, OWLDataProperty(p["value"]), OWLLiteral(value=float(o["value"]))
    #             else:
    #                 raise NotImplementedError(f"Currently this type of literal is not supported:{o} "
    #                                           f"but can done easily let us know :)")
    #         else:
    #             raise RuntimeError(f"Unrecognized type {subject_} ({p}) ({o})")
