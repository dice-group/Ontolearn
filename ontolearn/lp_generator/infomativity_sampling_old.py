import numpy as np

import os
from owlapy.class_expression import OWLThing
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils.static_funcs import concept_len
from ontolearn.lp_generator.helper_funcs import generate_concepts, sample_examples, generate_data_description
import json


def get_euclidean_distance_similarity(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)
    similarity = 1 / (1 + distance)  # Convert distance to similarity (range: (0, 1])
    return similarity
def get_manhattan_distance_similarity(vec1, vec2):
    distance = np.sum(np.abs(vec1 - vec2))
    similarity = 1 / (1 + distance)  # Convert distance to similarity (range: (0, 1])
    return similarity


def get_sigmoid_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    else:
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        # sigmoid normalization to convert cosine similarity to range (0, 1)
        normalized_similarity = 1 / (1 + np.exp(-cosine_similarity/0.1))
        return normalized_similarity



def concept_to_ind_similarity(local_pos, ind, embeddings, similarity_func='get_euclidean_distance_similarity'):
    ind_embedding = embeddings.loc[ind].values
    local_pos_embeddings = [embeddings.loc[pos].values for pos in local_pos]
    if similarity_func == 'get_euclidean_distance_similarity':
        sim_local_pos_ind = [get_euclidean_distance_similarity(ind_embedding, pos_emb) for pos_emb in local_pos_embeddings]
    elif similarity_func == 'get_manhattan_distance_similarity':
        sim_local_pos_ind = [get_manhattan_distance_similarity(ind_embedding, pos_emb) for pos_emb in local_pos_embeddings]
    elif similarity_func == 'get_cosine_similarity':
        sim_local_pos_ind = [get_sigmoid_cosine_similarity(ind_embedding, pos_emb) for pos_emb in local_pos_embeddings]

    #print(f"Similarity between {ind} and local positive examples: {sim_local_pos_ind}")
    print(sim_local_pos_ind)
    normalized_sim = sum(sim_local_pos_ind) / len(sim_local_pos_ind)
    return normalized_sim



def get_informativity(golbal_pos,local_pos, embeddings,epsilon, similarity_func='get_euclidean_distance_similarity'):
    if len(golbal_pos) == 0:
        return 1.0
    else:
        """if similarity_func == get_cosine_similarity:  
            sim_localpos_globalpos = [concept_to_ind_similarity(local_pos, pos, embeddings, similarity_func= ) for pos in golbal_pos]"""
        if similarity_func == 'get_euclidean_distance_similarity':
            sim_localpos_globalpos = [concept_to_ind_similarity(local_pos, pos, embeddings, similarity_func='get_euclidean_distance_similarity') for pos in golbal_pos]
        elif similarity_func == 'get_manhattan_distance_similarity':
            sim_localpos_globalpos = [concept_to_ind_similarity(local_pos, pos, embeddings, similarity_func='get_manhattan_distance_similarity') for pos in golbal_pos]
        elif similarity_func == 'get_cosine_similarity':
            sim_localpos_globalpos = [concept_to_ind_similarity(local_pos, pos, embeddings, similarity_func='get_cosine_similarity') for pos in golbal_pos]
        #print(f"Similarity between global positive examples and local positive examples: {sim_localpos_globalpos}")
        print(sim_localpos_globalpos)
        normalized_sim = sum(sim_localpos_globalpos) / len(sim_localpos_globalpos)
        informativity = epsilon + (1 - epsilon) * (1 - normalized_sim)
        return informativity


class InformativityBasedLPGenerator:
    def __init__(self, knowledge_base, max_iterations=1000, probability=0.5, sample_ind_size=100,
                 min_example_percent=0.1, seed=42, generate_data_description=True, save_path=None, beyond_alc=False, embeddings=None, inf_threshold=0.8):
        self.kb = knowledge_base
        self.max_itter = max_iterations
        self.probability = probability
        self.sample_ind_size = sample_ind_size
        self.dataset = dict()
        self.root = OWLThing
        self.curr_itter = 0
        self.concept_dict = dict()
        self.concepts = None
        self.all_individuals = set(self.kb.individuals())
        self.beyond_alc = beyond_alc
        if self.beyond_alc:
            self.refinement_operator = ExpressRefinement(knowledge_base=self.kb, max_child_length=10,
                                    sample_fillers_count=5,
                                    downsample=True, use_inverse=True,
                                    use_card_restrictions=True,
                                    use_numeric_datatypes=True, use_time_datatypes=True,
                                    use_boolean_datatype=True,
                                    expressivity=0.8)
        else:
            self.refinement_operator = ExpressRefinement(knowledge_base=self.kb, max_child_length=10,
                                    sample_fillers_count=5,
                                    downsample=True, use_inverse=False,
                                    use_card_restrictions=False,
                                    use_numeric_datatypes=False, use_time_datatypes=False,
                                    use_boolean_datatype=False,
                                    expressivity=0.8)
        self.min_example_percent = min_example_percent
        self.seed = seed
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.generate_data_description = generate_data_description
        self.save_path = save_path
        self.positive_examples_in_current_dataset = set()
        self.embeddings = embeddings
        self.inform_threshold = inf_threshold

    def get_concepts(self):
        self.concept_dict, self.concepts = generate_concepts(self.kb, self.root, self.refinement_operator,
                                                                 self.min_example_percent)

    def generate_lps(self):
        self.get_concepts()
        for concept in self.concepts:
            raw_pos_set = self.concept_dict[concept]
            concept_informativity = get_informativity(self.positive_examples_in_current_dataset, raw_pos_set, self.embeddings, epsilon=0.1)
            if concept_informativity >= self.inform_threshold:
                raw_neg_set = self.all_individuals - raw_pos_set
                pos, neg = sample_examples(raw_pos_set, raw_neg_set, self.sample_ind_size)
                self.dataset[self.dl_syntax_renderer.render(concept)] = {
                    'positive examples': pos,
                    'negative examples': neg,
                    'length': concept_len(concept)
                }
                self.positive_examples_in_current_dataset.update(raw_pos_set)
        if self.generate_data_description:
            description = generate_data_description(self.dataset)
            print(description)
        if self.save_path is not None:
            dataset = list(self.dataset.items())
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))
            with open(self.save_path, 'w', encoding="utf-8") as f:
                json.dump(dataset, f, indent=3, ensure_ascii=False)






