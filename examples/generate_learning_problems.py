"""
python examples/generate_learning_problems.py --path_kb datasets/mutagenesis/kb/ontology.owl --path_embeddings datasets/mutagenesis/embeddings/DeCaL_entity_embeddings.csv --get_emb_short_name True --sample_ind_size 1000 --max_example_percent 0.1 --generator infomativity_based_lp --generate_data_description True --save_path sampled_data/informative/mean_sim_50/mutagenesis --beyond_alc False --inf_threshold 0.86 --second_level_refinement False --num_sub_roots 50 --epsilon 0.001 --max_data_len 50 --max_itter 500
"""

import argparse
import random
import numpy as np
import pandas as pd

from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.lp_generator.biased_random_sampling import BiasedRandomLPGenerator
from ontolearn.lp_generator.infomativity_sampling import InformativityBasedLPGenerator
from ontolearn.lp_generator.helper_funcs import get_short_name


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
    # 1. Load Knowledge Base
    print(f"Loading Knowledge Base from: {args.path_kb}")
    kb = KnowledgeBase(path=args.path_kb)

    # 2. Load Embeddings (Expects a CSV/Parquet where index is individual name)
    print(f"Loading Embeddings from: {args.path_embeddings}")
    if args.path_embeddings.endswith('.csv'):
        embeddings = pd.read_csv(args.path_embeddings, index_col=0).drop_duplicates(subset="0")
    else:
        raise ValueError("Embeddings file must be .csv")

    if args.get_emb_short_name:
        embeddings.index = embeddings.index.map(get_short_name)

    # 3. Initialize Generator
    if args.generator == 'infomativity_based_lp':
        lp_gen = InformativityBasedLPGenerator(
            knowledge_base=kb,
            embeddings=embeddings,
            sample_ind_size=args.sample_ind_size,
            max_example_percent=args.max_example_percent,
            seed=args.seed,
            inf_threshold=args.inf_threshold,
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
            max_itter=args.max_itter
        )
    elif args.generator == 'random_lp':
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

    # 4. Run generation
    lp_gen.generate_lps()
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
    parser.add_argument('--generator', type=str, default='infomativity_based_lp',
                        help='Chose from [random_lp, infomativity_based_lp]')
    parser.add_argument('--inf_threshold', type=float, default=0.5, help='Informativity threshold to keep an LP')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Small constant for informativity calculation')
    parser.add_argument('--sample_ind_size', type=int, default=100, help='Number of examples to sample')
    parser.add_argument('--max_example_percent', type=float, default=0.1)

    # Execution Constraints
    parser.add_argument('--max_data_len', type=int, default=1000, help='Max number of LPs to generate')
    parser.add_argument('--max_itter', type=int, default=10000, help='Max iterations to attempt')
    parser.add_argument('--seed', type=int, default=42)

    # Others
    parser.add_argument('--generate_data_description', type=str2bool, default=False,
                        help='Wheather to gernerate data description')
    parser.add_argument('--get_emb_short_name', type=str2bool, default=False,
                        help='Wheather to perform emb shortname mapping')

    args = parser.parse_args()

    set_seed(args.seed)
    start(args)
