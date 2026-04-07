from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm

from ontolearn.refinement_operators import ExpressRefinement
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils.static_funcs import concept_len
from ontolearn.lp_generator.helper_funcs import generate_concepts, generate_data_description, \
    construct_exemplars, sample_examples_from_clusters, get_ind_to_emb
import json

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def euclidean_similarity(vec1, vec2):
    dist = cdist([vec1], [vec2], metric='euclidean')[0][0]
    sim = 1 / (1 + dist)  # Convert distance to similarity in range (0, 1)
    return sim

def calculate_similarity(existing_data, new_lp, kg_emb_model='DeCaL'):
    if kg_emb_model == 'DeCaL':
        """
        - existing data is matrix of [num_existing_pos, embedding_dim] 
        - new_lp is a matrix of [num_new_pos, embedding_dim]
        - The cosine similarity between each new positive individual and each existing positive individual 
        is calculated. Then the average similarity across all new positive examples is computed to get a single 
        similarity score for the new LP. 
        - Finally, a sigmoid normalization is applied to convert the similarity score to a value between 0 and 1.
        - A scaling factor in the sigmoid function is used to control the steepness of the curve [0.1,0.2,0.3,0.5]
        """
        if existing_data.shape[0] == 0:
            return 0.0
        else:
            similarity_matrix = cosine_similarity(new_lp, existing_data)
            average_similarity = np.mean(similarity_matrix)
            normalized_similarity = 1 / (1 + np.exp(-average_similarity/0.3))
            return normalized_similarity

def get_informativity(golbal_pos,local_pos, epsilon, kg_emb_model='DeCaL'):
        similarity_global_local = calculate_similarity(golbal_pos, local_pos, kg_emb_model)
        informativity = epsilon + (1 - epsilon) * (1 - similarity_global_local)
        return informativity

class InformativityBasedLPGenerator:
    def __init__(self, knowledge_base, sample_ind_size=100, min_example_percent=0.1, seed=42,
                 generate_data_description=True, save_path=None, beyond_alc=False, embeddings=None, inf_threshold=None,
                 max_child_length = 10, max_length =10,refinement_expressivity=0.2, second_level_refinement = False,
                 num_sub_roots = 50, epsilon = 0.01):
        self.kb = knowledge_base
        self.sample_ind_size = sample_ind_size
        self.dataset = dict()
        self.root = self.kb.generator.thing
        self.curr_itter = 0
        self.concept_dict = dict()
        self.concepts = None
        self.all_individuals = set(self.kb.individuals())
        self.beyond_alc = beyond_alc

        if self.beyond_alc:
            self.refinement_operator = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length,
                                    sample_fillers_count=5,
                                    downsample=True, use_inverse=True,
                                    use_card_restrictions=True,
                                    use_numeric_datatypes=True, use_time_datatypes=True,
                                    use_boolean_datatype=True,
                                    expressivity=refinement_expressivity)
        else:
            self.refinement_operator = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length,
                                    sample_fillers_count=5,
                                    downsample=True, use_inverse=False,
                                    use_card_restrictions=False,
                                    use_numeric_datatypes=False, use_time_datatypes=False,
                                    use_boolean_datatype=False,
                                    expressivity=refinement_expressivity)
        self.min_example_percent = min_example_percent
        self.seed = seed
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.generate_data_description = generate_data_description
        self.save_path = save_path
        self.global_positive_examples = set()
        self.global_positive_examples_embeddings = None
        self.embeddings = embeddings
        self.inform_threshold = inf_threshold
        self.informativity_scores = defaultdict(list)
        self.max_length = max_length
        self.second_level_refinement = second_level_refinement
        self.num_sub_roots = num_sub_roots
        self.epsilon = epsilon

    def get_concepts(self):
        self.concept_dict, self.concepts = generate_concepts(self.kb, self.root, self.refinement_operator,
                                                             self.min_example_percent, self.max_length,
                                                             self.num_sub_roots, self.second_level_refinement)

    def generate_lps(self):
        print("generating concepts using the refinement operator...")
        self.get_concepts()
        self.ind_to_emb = get_ind_to_emb(self.embeddings, self.concept_dict)

        print("clustering individuals based on their embeddings...")
        self.ind_to_cluster, self.clusters_to_ind, _ = construct_exemplars(self.ind_to_emb)
        all_cluster_ids = set(self.clusters_to_ind.keys())

        print("processing concepts and sampling examples based on informativity...")

        for concept in tqdm(self.concepts, desc="Processing concepts"):
            raw_pos_set = self.concept_dict[concept]
            raw_neg_set = self.all_individuals - raw_pos_set
            pos, neg = sample_examples_from_clusters(all_cluster_ids, self.ind_to_cluster, self.clusters_to_ind, raw_pos_set, raw_neg_set, self.sample_ind_size)#all_cluster_ids, ind_to_cluster, clusters_to_ind, raw_pos, sample_ind_size
            if len(self.global_positive_examples) == 0:
                self.global_positive_examples = pos
                emb_pos = [self.ind_to_emb[ind] for ind in pos]
                self.global_positive_examples_embeddings = np.array(emb_pos)
                #current_concept_embeddings = self.global_positive_examples_embeddings
                #informativity = get_informativity(self.global_positive_examples_embeddings, current_concept_embeddings,
                #                                  epsilon=0.1, kg_emb_model='DeCaL')
                informativity = 1.0  # if there are no existing positive examples, we can consider the informativity to be 1, as it will add new information to the dataset
            else:
                emb_pos = [self.ind_to_emb[ind] for ind in pos]
                current_concept_embeddings = np.array(emb_pos)
                try:
                    informativity = get_informativity(self.global_positive_examples_embeddings, current_concept_embeddings,
                                                      epsilon=self.epsilon, kg_emb_model='DeCaL')
                except Exception as e:
                    print(f"Error calculating informativity for concept {self.dl_syntax_renderer.render(concept)}: {e}")
                    informativity = 0.0  # if there is an error in calculating informativity, we can consider it to be 0 to avoid adding potentially redundant LPs to the dataset

            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            self.informativity_scores[concept_name].append(informativity)

            if informativity > self.inform_threshold:

                self.dataset[concept_name] = {
                    'positive examples': list(pos),
                    'negative examples': list(neg),
                    'length': concept_len(concept)
                }
                # update the global positive examples and their embeddings with the new positive examples without
                # repeating existing examples
                new_global_pos_set = pos - self.global_positive_examples
                if new_global_pos_set:
                    self.global_positive_examples.update(new_global_pos_set)
                    new_emb_list = [self.ind_to_emb[ind] for ind in new_global_pos_set]
                    new_emb_array = np.array(new_emb_list)
                    self.global_positive_examples_embeddings = np.vstack((self.global_positive_examples_embeddings, new_emb_array))
        if self.generate_data_description:
            description = generate_data_description(self.dataset)
            print(description)
        if self.save_path is not None:
            dataset = list(self.dataset.items())
            save_path_lp = os.path.join(self.save_path, 'generated_lps_with_inf.json')
            save_path_inf = os.path.join(self.save_path, 'informativity_scores.json')
            if not os.path.exists(os.path.dirname(save_path_lp)):
                os.makedirs(os.path.dirname(save_path_lp))
            if not os.path.exists(os.path.dirname(save_path_inf)):
                os.makedirs(os.path.dirname(save_path_inf))
            with open(save_path_lp, 'w', encoding="utf-8") as f1:
                json.dump(dataset, f1, indent=4, ensure_ascii=False)
            with open(save_path_inf, 'w', encoding="utf-8") as f2:
                json.dump(self.informativity_scores, f2, indent=4, ensure_ascii=False)




