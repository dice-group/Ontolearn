import random
from collections import defaultdict

from sklearn.preprocessing import normalize
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

def generate_concepts(kb, root, refinement_operator, min_example_percent, max_length = 10, num_sub_roots=50, second_level_refinement = False):
    concept_dict = dict()
    len_kb_individuals = len(kb.individuals())
    print("no. of individuals in the KB: ", len_kb_individuals)
    print("refining root using the refinement operator")
    subroots = {ref for ref in refinement_operator.refine(root,  max_length=max_length)}
    print("|Thing refinements|: ", len(subroots))
    refinements = subroots
    if second_level_refinement:
        subroots_sample = random.sample(list(subroots), k=num_sub_roots)
        for subroot in tqdm(subroots_sample, desc="Refining subroots..."):
            refinements.update(refinement_operator.refine(subroot,  max_length=max_length))
    for concept in tqdm(refinements, desc="Filtering concepts"):
        pos_examples = set(kb.individuals(concept))
        num_pos_examples = len(pos_examples)
        if num_pos_examples == 0:
            continue
        if num_pos_examples < min_example_percent * len_kb_individuals:
            concept_dict[concept] = pos_examples
    concepts = list(concept_dict.keys())
    return concept_dict, concepts

def sample_examples(raw_pos, raw_neg, sample_ind_size):
    n = min(len(raw_pos), len(raw_neg), sample_ind_size)
    pos = random.sample(raw_pos, n)
    neg = random.sample(raw_neg, n)
    return pos, neg

def generate_data_description(dataset):
    no_of_lps = len(dataset)
    avg_pos_examples = sum(len(info['positive examples']) for info in dataset.values()) / no_of_lps
    avg_neg_examples = sum(len(info['negative examples']) for info in dataset.values()) / no_of_lps
    all_individuals = set()
    min_example_lp = min(dataset.items(), key=lambda x: len(x[1]['positive examples']) + len(x[1]['negative examples']))
    max_example_lp = max(dataset.items(), key=lambda x: len(x[1]['positive examples']) + len(x[1]['negative examples']))
    largest_lp = max(dataset.items(), key=lambda x: x[1]['length'])
    smallest_lp = min(dataset.items(), key=lambda x: x[1]['length'])
    for info in dataset.items():
        all_individuals.update(set(info[1]['positive examples']))
        all_individuals.update(set(info[1]['negative examples']))

    total_no_of_individuals = len(all_individuals)

    description = f"Number of learning problems: {no_of_lps}\n" \
                  f"Average number of positive examples: {avg_pos_examples}\n" \
                  f"Average number of negative examples: {avg_neg_examples}\n" \
                  f"Total number of individuals: {total_no_of_individuals}\n" \
                  f"Smallest LP: {min_example_lp[0]} with {len(min_example_lp[1]['positive examples'])} positive examples and {len(min_example_lp[1]['negative examples'])} negative examples\n" \
                  f"Largest LP: {max_example_lp[0]} with {len(max_example_lp[1]['positive examples'])} positive examples and {len(max_example_lp[1]['negative examples'])} negative examples\n" \
                  f"Smallest LP by length: {smallest_lp[0]} with length {smallest_lp[1]['length']}\n" \
                  f"Largest LP by length: {largest_lp[0]} with length {largest_lp[1]['length']}\n"


    return description
def construct_exemplars(ind_to_emb):
    ind_names = list(ind_to_emb.keys())
    ind_embs = np.vstack(list(ind_to_emb.values()))
    ind_emb_norm = normalize(ind_embs, norm='l2') # to make the kmeans clustering spherical and to make cosine similarity and euclidean distance equivalent for clustering
    no_of_clusters = int(np.sqrt(len(ind_names)))
    kmeans = KMeans(
        n_clusters=no_of_clusters,
        random_state=42,
        n_init=10,
        algorithm='elkan'
    ).fit(ind_emb_norm)

    labels = kmeans.labels_

    ind_to_cluster = dict(zip(ind_names, labels))
    clusters_to_ind = defaultdict(list)
    for name, label in zip(ind_names, labels):
        clusters_to_ind[int(label)].append(name)

    return ind_to_cluster, dict(clusters_to_ind), labels

def sample_examples_from_clusters(all_cluster_ids, ind_to_cluster, clusters_to_ind, raw_pos, raw_neg, sample_ind_size):
    pos_by_cluster = {}
    for p in raw_pos:
        p_name = p.str.split("/")[-1]
        if p_name in ind_to_cluster:
            c_id = ind_to_cluster[p_name]
            pos_by_cluster.setdefault(c_id, []).append(p_name)

    pos_cluster_ids = list(pos_by_cluster.keys())
    #neg_cluster_ids = list(all_cluster_ids - set(pos_cluster_ids))

    n_samples = min(len(raw_pos), len(raw_neg), sample_ind_size)
    #print(f'sample size for this concept: {n_samples}, pos_len {len(raw_pos)}, neg_len {len(raw_neg)}')

    if len(pos_cluster_ids) > 0:
        selected_pos, selected_neg = set(), set()
        #counter = 0
        while len(selected_pos) < n_samples:
            # print how many no. of times the while loop runs
            #counter += 1


            for c in pos_cluster_ids:
                if len(pos_by_cluster[c]) > 0:
                    #print(f'Selecting from cluster {c}, remaining examples in cluster: {len(pos_by_cluster[c])}')
                    selected_pos.add(pos_by_cluster[c][0])  # take one example from each positive cluster
                    # remove the selected positive example from the cluster to avoid selecting it again
                    pos_by_cluster[c] = pos_by_cluster[c][1:]
                    if len(selected_pos) >= n_samples:
                        break
                else:
                    continue
                    #print(f'Cluster {c} has no more positive examples to select.')
            #print(f'While loop iteration: {counter}, selected_pos size: {len(selected_pos)}')
       # take n negative examples from raw neg
        selected_neg = set(random.sample([neg.str.split("/")[-1] for neg in raw_neg], n_samples))
        #print(f'Selected {len(selected_neg)} negative examples.')

    else:
        selected_pos, selected_neg = set(), set()
    return selected_pos, selected_neg

def get_short_name(name):
    _name = str(name)
    if "/" in _name:
        return _name.split("/")[-1]
    return _name

def get_ind_to_emb(embeddings, concept_dict):
    ind_to_emb = dict()
    for i in concept_dict:
        pos = concept_dict[i]
        for p in pos:
            p_name = p.str.split("/")[-1]
            if p_name in embeddings.index:
                ind_to_emb[p_name] = embeddings.loc[p_name].values
    return ind_to_emb