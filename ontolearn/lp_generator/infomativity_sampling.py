import json
import os
from collections import defaultdict
import numpy as np
from owlapy.render import DLSyntaxObjectRenderer
from sklearn.metrics.pairwise import cosine_similarity
from ontolearn.lp_generator.helper_funcs import (
    construct_exemplars,
    generate_concepts_parallel,
    generate_data_description,
    get_ind_to_emb,
    sample_examples_from_clusters,
)
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.utils.static_funcs import concept_len

def get_informativity_fast(global_pos, local_pos, epsilon, sim_matrix, id_to_idx, similarity_type='mean'):
    """
    Compute the informativity of a new concept based on the similarity between
    positive examples of the new concept and positive examples of existing concepts.
    """

    # Map local individuals to matrix indices once
    local_indices = [id_to_idx[ind] for ind in local_pos if ind in id_to_idx]

    sim_per_concept = []

    for _, concept_instances in global_pos.items():
        # Map global individuals to matrix indices once
        global_indices = [id_to_idx[ind] for ind in concept_instances if ind in id_to_idx]

        if global_indices:
            sub_matrix = sim_matrix[np.ix_(local_indices, global_indices)]

            if similarity_type == 'max':
                sub_matrix = np.max(sub_matrix, axis=1)

            sim_per_concept.append(np.mean(sub_matrix))

    # Calculate global mean similarity
    if not sim_per_concept:
        sim_local_to_global_mean = 0.0
    else:
        sim_local_to_global_mean = np.mean(sim_per_concept)

    # Ensure similarity isn't negative (relevant for Cosine similarity)
    sim_local_to_global_mean = max(0.0, sim_local_to_global_mean)

    informativity = epsilon + (1.0 - sim_local_to_global_mean) * (1.0 - epsilon)

    return informativity


class InformativityBasedLPGenerator:
    def __init__(self, knowledge_base, sample_ind_size=100, max_example_percent=0.1, seed=42,
                 generate_data_description=True, save_path=None, beyond_alc=False, embeddings=None, inf_threshold=None,
                 max_child_length=10, max_length=10, refinement_expressivity=0.2, second_level_refinement=False,
                 num_sub_roots=50, epsilon=0.01, max_data_len=1000, max_itter=10000, similarity_type='mean'):
        self.kb = knowledge_base
        self.sample_ind_size = sample_ind_size
        self.dataset = dict()
        self.root = self.kb.generator.thing
        self.curr_itter = 0
        self.concept_dict = dict()
        self.concepts = None
        self.all_individuals = set(self.kb.individuals())
        self.all_individuals_str = set(i.str for i in self.all_individuals)

        self.refinement_operator = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length,
                                                     sample_fillers_count=5,
                                                     downsample=True, use_inverse=beyond_alc,
                                                     use_card_restrictions=beyond_alc,
                                                     use_numeric_datatypes=beyond_alc, use_time_datatypes=beyond_alc,
                                                     use_boolean_datatype=beyond_alc,
                                                     expressivity=refinement_expressivity)

        self.max_example_percent = max_example_percent
        self.seed = seed
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.generate_data_description = generate_data_description
        self.save_path = save_path
        self.global_positive_examples = dict()
        # self.global_positive_examples_embeddings = dict()
        self.embeddings = embeddings
        self.inform_threshold = inf_threshold
        self.informativity_scores = defaultdict(list)
        self.max_length = max_length
        self.second_level_refinement = second_level_refinement
        self.num_sub_roots = num_sub_roots
        self.epsilon = epsilon
        self.curr_data_len = 0
        self.curr_itter = 0
        self.max_data_len = max_data_len
        self.max_itter = max_itter
        self.similarity_type = similarity_type

    def get_concepts(self):
        self.concept_dict, self.concepts = generate_concepts_parallel(self.kb, self.root, self.refinement_operator,
                                                                      self.max_example_percent, self.max_length,
                                                                      self.num_sub_roots, self.second_level_refinement)
        # randomly shuffle the concepts

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
        if self.similarity_type == 'mean' or self.similarity_type == 'max':
            self.get_sim_lookup_for_inds()
        self.ind_to_emb = get_ind_to_emb(self.embeddings, self.concept_dict)

        print("clustering individuals based on their embeddings...")
        self.ind_to_cluster, self.clusters_to_ind, _ = construct_exemplars(self.ind_to_emb)
        all_cluster_ids = set(self.clusters_to_ind.keys())

        print("processing concepts and sampling examples based on informativity...")

        while self.curr_itter < self.max_itter and len(self.concepts) > 0:
            idx = np.random.randint(0, len(self.concepts))
            concept = self.concepts[idx]

            raw_pos_set = set(i.str for i in self.concept_dict[concept])
            raw_neg_set = self.all_individuals_str - raw_pos_set

            pos, neg = sample_examples_from_clusters(all_cluster_ids, self.ind_to_cluster, self.clusters_to_ind,
                                                     raw_pos_set, raw_neg_set, self.sample_ind_size)

            self.current_positive_examples = pos

            if len(self.global_positive_examples) == 0:
                informativity = 1.0  # if there are no existing positive examples, we can consider the informativity to be 1, as it will add new information to the dataset

            else:
                try:
                    informativity = get_informativity_fast(self.global_positive_examples,
                                                           self.current_positive_examples, self.epsilon,
                                                           self.sim_matrix, self.id_to_idx, self.similarity_type)
                except Exception as e:
                    print(f"Error calculating informativity for concept {self.dl_syntax_renderer.render(concept)}: {e}")
                    informativity = 0.0  # if there is an error in calculating informativity, we can consider it to be 0 to avoid adding potentially redundant LPs to the dataset

            if informativity > self.inform_threshold:
                concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
                self.concepts[idx] = self.concepts[-1]
                self.concepts.pop()
                self.dataset[concept_name] = {
                    'positive examples': list(pos),
                    'negative examples': list(neg),
                    'length': concept_len(concept)
                }
                self.global_positive_examples[concept_name] = pos
                self.informativity_scores[concept_name].append(informativity)
                self.curr_data_len = len(self.dataset)
                if self.curr_data_len >= self.max_data_len:
                    print(f"Reached maximum data length of {self.max_data_len}. Stopping generation.")
                    break
            self.curr_itter += 1

        if self.generate_data_description:
            description = generate_data_description(self.dataset)
            print(description)
        if self.save_path is not None:
            dataset = list(self.dataset.items())
            os.makedirs(self.save_path, exist_ok=True)  # Creates dir and ignores if it exists

            save_path_lp = os.path.join(self.save_path, 'generated_lps_with_inf.json')
            save_path_inf = os.path.join(self.save_path, 'informativity_scores.json')

            with open(save_path_lp, 'w', encoding="utf-8") as f1:
                json.dump(dataset, f1, indent=4, ensure_ascii=False)
            with open(save_path_inf, 'w', encoding="utf-8") as f2:
                json.dump(self.informativity_scores, f2, indent=4, ensure_ascii=False)
