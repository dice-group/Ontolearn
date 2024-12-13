
import argparse 
import pandas as pd
from ontolearn.semantic_caching import run_cache, concept_generator
from plot_metrics import *
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--cache_size_ratios', type=list, default=[.1], help="cache size is proportional to num_concepts, cache size = k * num_concepts")
parser.add_argument('--path_kg', type=str, default=["KGs/Family/father.owl"])
parser.add_argument('--path_kge', type=list, default=None)
parser.add_argument('--name_reasoner', type=str, default='EBR', choices=["EBR",'HermiT', 'Pellet', 'JFact', 'Openllet'])
parser.add_argument('--eviction_strategy', type=str, default='LRU', choices=['LIFO', 'FIFO', 'LRU', 'MRU', 'RP'])
parser.add_argument('--random_seed_for_RP', type=int, default=10, help="Random seed if the eviction startegy is RP")
parser.add_argument('--cache_type', type=str, default='hot', choices=['hot', 'cold'], help="Type of cache to be used. With cold cache we initialize the cache with NC, NNC")
parser.add_argument('--shuffle_concepts', action='store_true',help="If set, we shuffle the concepts for randomness")
args = parser.parse_args()

def get_cache_size(list_k, path_kg):
    
    data_size = len(concept_generator(path_kg))

    return [max(1, int(k * data_size)) for k in list_k]


results = []
detailed_results = []
for path_kg in args.path_kg:
    for cache_size in get_cache_size(args.cache_size_ratios, path_kg):
        for strategy in ['LIFO', 'FIFO', 'LRU', 'MRU', 'RP']:
            result, detailed = run_cache(
                path_kg=path_kg,
                path_kge=args.path_kge,
                cache_size=cache_size,
                name_reasoner=args.name_reasoner,
                eviction=strategy,
                random_seed=args.random_seed_for_RP,
                cache_type=args.cache_type,
                shuffle_concepts=args.shuffle_concepts
            )
            results.append(result)
            detailed_results.append(detailed)

    data_name = result['dataset']
    df = pd.DataFrame(results)
    all_detailed_results = pd.DataFrame([item for sublist in detailed_results for item in sublist])
    print(df)
    
    # Save to CSV
    df.to_csv(f'caching_results_{data_name}/cache_experiments_{args.name_reasoner}_{data_name}_{args.cache_type}.csv', index=False)
    all_detailed_results.to_csv(f'caching_results_{data_name}/detailled_experiments_{args.name_reasoner}_{data_name}_{args.cache_type}.csv', index=False)


