"""
python examples/generate_learning_problems.py --path_kb datasets/mutagenesis/kb/ontology.owl --path_embeddings datasets/mutagenesis/embeddings/DeCaL_entity_embeddings.csv --get_emb_short_name True --sample_ind_size 1000 --max_example_percent 0.1 --generator infomativity_based_lp --generate_data_description True --save_path sampled_data/informative/mean_sim_50/mutagenesis --beyond_alc False --inf_threshold 0.86 --second_level_refinement False --num_sub_roots 50 --epsilon 0.001 --max_data_len 50 --max_itter 500
"""

import argparse
import random
import time

import numpy as np
import pandas as pd

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.lp_generator.biased_random_sampling import BiasedRandomLPGenerator
#from ontolearn.lp_generator.infomativity_sampling import InformativityBasedLPGenerator
from ontolearn.lp_generator.inf_sampling import InformativityBasedLPGenerator
from ontolearn.lp_generator.helper_funcs import get_short_name
from ontolearn.utils.static_funcs import concept_len


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')


def start(args):
    # Load Knowledge Base
    print(f"Loading Knowledge Base from: {args.path_kb}")
    kb = KnowledgeBase(path=args.path_kb)

    # Load Embeddings (Expects a CSV/Parquet where index is individual name)
    print(f"Loading Embeddings from: {args.path_embeddings}")
    if args.path_embeddings.endswith('.csv'):
        embeddings = pd.read_csv(args.path_embeddings, index_col=0).drop_duplicates(subset="0")
    else:
        raise ValueError("Embeddings file must be .csv")

    if args.get_emb_short_name:
        embeddings.index = embeddings.index.map(get_short_name)

    # Initialize Generator
    if args.sampling_strategy == 'infomativite':
        lp_gen = InformativityBasedLPGenerator(
            knowledge_base=kb,
            embeddings=embeddings,
            sample_ind_size=args.sample_ind_size,
            max_example_percent=args.max_example_percent,
            seed=args.seed,
            #inf_threshold=args.inf_threshold,
            second_level_refinement=args.second_level_refinement,
            save_path=args.save_path,
            beyond_alc=args.beyond_alc,
            max_child_length=args.max_child_length,
            max_length=args.max_length,
            refinement_expressivity=args.refinement_expressivity,
            generate_data_description=args.generate_data_description,
            num_sub_roots=args.num_sub_roots,
            epsilon=args.epsilon,
            max_data_len=args.max_data_len,
            max_itter=args.max_itter,
            similarity_type =args.similarity_type,
            temperature= args.temperature
        )
    elif args.sampling_strategy == 'random':
        lp_gen = BiasedRandomLPGenerator(
            knowledge_base=kb,
            max_iterations=args.max_itter,
            sample_ind_size=args.sample_ind_size,
            max_example_percent=args.max_example_percent,
            seed=args.seed,
            generate_data_description=args.generate_data_description,
            save_path=args.save_path,
            beyond_alc=args.beyond_alc,
            max_child_length=args.max_child_length,
            max_length=args.max_length,
            refinement_expressivity=args.refinement_expressivity,
            second_level_refinement=args.second_level_refinement,
            num_sub_roots=args.num_sub_roots,
            max_data_len=args.max_data_len)

    if args.generate_and_save_concept_pool:
        lp_gen.get_concepts()
        new_concept_dict = {}
        new_concepts = []
        concept_len_dict = {}
        for key, value in lp_gen.concept_dict.items():
            # serialize the key
            serialized_key = lp_gen.dl_syntax_renderer.render(key.get_nnf())
            # serialize the value
            serialized_value = [ind.str for ind in value]
            # update the concept dict with the serialized key and value in the new concept dict
            new_concept_dict[serialized_key] = serialized_value
            # add the serialized key to the new concepts list
            new_concepts.append(serialized_key)
            # add the length of the value to the concept len dict
            concept_len_dict[serialized_key] = concept_len(key)

        # save the new concept dict, new concepts list, and concept len dict to json files
        import json
        with open(args.concept_dict_path, 'w') as f:
            json.dump(new_concept_dict, f, indent=4)
        with open(args.concepts_path, 'w') as f:
            json.dump(new_concepts, f, indent=4)
        with open(args.concept_len_dict_path, 'w') as f:
            json.dump(concept_len_dict, f, indent=4)

    #  Run generation
    strat_time = time.time()
    if args.concept_dict_path and args.concepts_path and args.concept_len_dict_path:
        import json
        with open(args.concept_dict_path, 'r') as f:
            concept_dict = json.load(f)
        with open(args.concepts_path, 'r') as f:
            concepts = json.load(f)
        with open(args.concept_len_dict_path, 'r') as f:
            concept_len_dict = json.load(f)
        lp_gen.generate_lps(concept_dict=concept_dict, concepts=concepts, concept_len_dict=concept_len_dict)
    else:
        lp_gen.generate_lps()
    time_taken = time.time() - strat_time
    print(f"Time taken for generation: {time_taken:.2f} seconds")
    print(f"Generation complete. Results saved to: {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Informativity-based LP Generator')

    # Essential Paths
    parser.add_argument('--path_kb', type=str, required=True, help='Path to OWL ontology')
    parser.add_argument('--path_embeddings', type=str, required=True, help='Path to embeddings CSV')
    parser.add_argument('--save_path', type=str, default='./results', help='Directory to save JSON results')

    # Logic & Refinement Hyperparameters
    parser.add_argument('--beyond_alc', type=str2bool, default=False,
                        help='Use complex constructs (inverse, card, etc.)')
    parser.add_argument('--max_child_length', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=10)
    parser.add_argument('--refinement_expressivity', type=float, default=0.2)
    parser.add_argument('--num_sub_roots', type=int, default=50)
    parser.add_argument('--second_level_refinement', type=str2bool, default=False,
                        help='Wheather to perform second level refinement to grnerate more data')

    # Sampling & Informativity Hyperparameters
    parser.add_argument('--sampling_strategy', type=str, default='infomativite',
                        help='Type of sampling to be used (random, infomativite)')
    #parser.add_argument('--inf_threshold', type=float, default=0.5, help='Informativity threshold to keep an LP')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Small constant for informativity calculation')
    parser.add_argument('--sample_ind_size', type=int, default=100, help='Number of examples to sample')
    parser.add_argument('--max_example_percent', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.01, help='Temperature for sampling (if applicable)')
    parser.add_argument('--similarity_type', type=str, default='mean', help='Type of similarity to use for informativity (mean, max)')

    # Execution Constraints
    parser.add_argument('--max_data_len', type=int, default=1000, help='Max number of LPs to generate')
    parser.add_argument('--max_itter', type=int, default=10000, help='Max iterations to attempt')
    parser.add_argument('--seed', type=int, default=42)

    # Others
    parser.add_argument('--generate_data_description', type=str2bool, default=False,
                        help='Wheather to gernerate data description')
    parser.add_argument('--get_emb_short_name', type=str2bool, default=False,
                        help='Wheather to perform emb shortname mapping')
    parser.add_argument('--concept_dict_path', type=str, default=None, help='Path to predefined concept dict (JSON) for sampling')
    parser.add_argument('--concepts_path', type=str, default=None, help='Path to predefined concepts (TXT) for sampling')
    parser.add_argument('--concept_len_dict_path', type=str, default=None, help='Path to predefined concept length dict (JSON) for sampling')
    parser.add_argument('--generate_and_save_concept_pool', type=str2bool, default=False, help='Whether to generate and save a pool of concepts used during sampling')

    args = parser.parse_args()

    set_seed(args.seed)
    start(args)
