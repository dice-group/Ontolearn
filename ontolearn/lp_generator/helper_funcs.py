import random

#helper functions

def generate_concepts(kb, root, refinement_operator, min_example_percent):
    concept_dict = dict()
    refinements = {ref for ref in refinement_operator.refine(root)}
    for concept in refinements:
        pos_examples = set(kb.individuals(concept))
        num_pos_examples = len(pos_examples)
        if num_pos_examples == 0:
            continue
        if num_pos_examples < min_example_percent * len(kb.individuals()):
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
    #avg_pos_examples = sum(len(pos) for pos, neg in dataset.values()) / no_of_lps
    avg_pos_examples = sum(len(info['positive examples']) for info in dataset.values()) / no_of_lps
    #avg_neg_examples = sum(len(neg) for pos, neg in dataset.values()) / no_of_lps
    avg_neg_examples = sum(len(info['negative examples']) for info in dataset.values()) / no_of_lps
    #total_no_of_individuals = sum(len(pos) + len(neg) for pos, neg in dataset.values())
    total_no_of_individuals = sum(len(info['positive examples']) + len(info['negative examples']) for info in dataset.values())
    #smallest_lp = min(dataset.items(), key=lambda x: len(x[1][0]) + len(x[1][1]))
    #largest_lp = max(dataset.items(), key=lambda x: len(x[1][0]) + len(x[1][1]))
    min_example_lp = min(dataset.items(), key=lambda x: len(x[1]['positive examples']) + len(x[1]['negative examples']))
    max_example_lp = max(dataset.items(), key=lambda x: len(x[1]['positive examples']) + len(x[1]['negative examples']))
    largest_lp = max(dataset.items(), key=lambda x: x[1]['length'])
    smallest_lp = min(dataset.items(), key=lambda x: x[1]['length'])
    description = f"Number of learning problems: {no_of_lps}\n" \
                  f"Average number of positive examples: {avg_pos_examples}\n" \
                  f"Average number of negative examples: {avg_neg_examples}\n" \
                  f"Total number of individuals: {total_no_of_individuals}\n" \
                  f"Smallest LP: {min_example_lp[0]} with {len(min_example_lp[1]['positive examples'])} positive examples and {len(smallest_lp[1]['negative examples'])} negative examples\n" \
                  f"Largest LP: {max_example_lp[0]} with {len(max_example_lp[1]['positive examples'])} positive examples and {len(largest_lp[1]['negative examples'])} negative examples\n" \
                  f"Smallest LP by length: {smallest_lp[0]} with length {smallest_lp[1]['length']}\n" \
                  f"Largest LP by length: {largest_lp[0]} with length {largest_lp[1]['length']}\n"


    return description