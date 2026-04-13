from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm

from ontolearn.refinement_operators import ExpressRefinement
from owlapy.render import DLSyntaxObjectRenderer
from ontolearn.utils.static_funcs import concept_len
from ontolearn.lp_generator.helper_funcs import  generate_data_description, \
    construct_exemplars, sample_examples_from_clusters, get_ind_to_emb, generate_concepts_parallel
import json

from sklearn.metrics.pairwise import cosine_similarity



def get_informativity(golbal_pos, local_pos, epsilon, similarity_lookup):  # To be deleted

    sim_local_to_global = []
    for key, value in golbal_pos.items():
        sim_ind_local_to_ind_global = []
        for local_ind in local_pos:
            for global_ind in value:
                try:
                    sim = similarity_lookup.at[local_ind, global_ind]
                except KeyError:
                    print(KeyError)
                sim_ind_local_to_ind_global.append(sim)

        if sim_ind_local_to_ind_global:
            sim_local_to_global.append(np.mean(sim_ind_local_to_ind_global))

        # check length
        # if len(sim_ind_local_to_ind_global) != len(value)*len(local_pos):
        #    print(f"Length of similarity list is {len(sim_ind_local_to_ind_global)}, expected {len(value)*len(local_pos)}.")

    # if len(sim_local_to_global) != len(golbal_pos):
    #    print(f"Length of local to global similarity list is {len(sim_local_to_global)}, expected {len(golbal_pos)}.")

    if sim_local_to_global:
        sim_local_to_global_mean = np.mean(sim_local_to_global)
    else:
        print("No similarities found between local and global positive examples.")

    sim_local_to_global_mean = max(0,
                                   sim_local_to_global_mean)  # Ensure similarity isn't negative (relevant for Cosine similarity)

    informativity = epsilon + (1 - sim_local_to_global_mean) * (1 - epsilon)
    return informativity


def get_informativity_fast(global_pos, local_pos, epsilon, sim_matrix, id_to_idx):
    """
    Optimized calculation of concept informativeness using NumPy vectorization.
    """

    # Map local individuals to matrix indices once
    local_indices = [id_to_idx[ind] for ind in local_pos if ind in id_to_idx]

    sim_per_concept = []

    for _, concept_instances in global_pos.items():
        global_indices = [id_to_idx[ind] for ind in concept_instances if ind in id_to_idx]

        if global_indices:
            sub_matrix = sim_matrix[np.ix_(local_indices, global_indices)]

            sim_per_concept.append(np.mean(sub_matrix))

    if not sim_per_concept:
        sim_local_to_global_mean = 0.0
    else:
        sim_local_to_global_mean = np.mean(sim_per_concept)

    sim_local_to_global_mean = max(0.0,
                                   sim_local_to_global_mean)  # Ensure similarity isn't negative (relevant for Cosine similarity)

    informativity = epsilon + (1.0 - sim_local_to_global_mean) * (1.0 - epsilon)

    return informativity

class InformativityBasedLPGenerator:
    def __init__(self, knowledge_base, sample_ind_size=100, max_example_percent=0.1, seed=42,
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
        self.max_example_percent = max_example_percent
        self.seed = seed
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.generate_data_description = generate_data_description
        self.save_path = save_path
        self.global_positive_examples = dict()
        #self.global_positive_examples_embeddings = dict()
        self.embeddings = embeddings
        self.inform_threshold = inf_threshold
        self.informativity_scores = defaultdict(list)
        self.max_length = max_length
        self.second_level_refinement = second_level_refinement
        self.num_sub_roots = num_sub_roots
        self.epsilon = epsilon

    def get_concepts(self):
        self.concept_dict, self.concepts = generate_concepts_parallel(self.kb, self.root, self.refinement_operator,
                                                             self.max_example_percent, self.max_length,
                                                             self.num_sub_roots, self.second_level_refinement)


    def get_sim_lookup_for_inds(self):
        all_kb_ind_list = [
            i.str.split("/")[-1] for i in self.all_individuals
            if i.str.split("/")[-1] in self.embeddings.index
        ]
        all_kb_ind_emb = self.embeddings.loc[all_kb_ind_list].values

        self.sim_matrix = cosine_similarity(all_kb_ind_emb)

        self.id_to_idx = {name: i for i, name in enumerate(all_kb_ind_list)}


    def generate_lps(self):
        print("generating concepts using the refinement operator...")
        self.get_concepts()
        self.get_sim_lookup_for_inds()
        self.ind_to_emb = get_ind_to_emb(self.embeddings, self.concept_dict)

        print("clustering individuals based on their embeddings...")
        self.ind_to_cluster, self.clusters_to_ind, _ = construct_exemplars(self.ind_to_emb)
        all_cluster_ids = set(self.clusters_to_ind.keys())

        print("processing concepts and sampling examples based on informativity...")

        for concept in tqdm(self.concepts, desc="Processing concepts"):
            raw_pos_set = self.concept_dict[concept]
            raw_neg_set = self.all_individuals - raw_pos_set
            pos, neg = sample_examples_from_clusters(all_cluster_ids, self.ind_to_cluster, self.clusters_to_ind, raw_pos_set, raw_neg_set, self.sample_ind_size)
            if len(self.global_positive_examples) == 0:
                self.current_positive_examples = pos

                informativity = 1.0  # if there are no existing positive examples, we can consider the informativity to be 1, as it will add new information to the dataset
            else:
                self.current_positive_examples = pos
                try:
                    #informativity = get_informativity(self.global_positive_examples, self.current_positive_examples,
                    #                                  self.epsilon, self.similarity_lookup)
                    informativity = get_informativity_fast(self.global_positive_examples, self.current_positive_examples, self.epsilon, self.sim_matrix, self.id_to_idx )
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
                self.global_positive_examples[concept_name] = pos

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



