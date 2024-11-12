
import argparse 
import pandas as pd
from semantic_caching import run_cache
from plot_metrics import *

#5, 16, 32, 128, 256, 512, 700, 800, 1024, , "KGs/Family/family.owl"
parser = argparse.ArgumentParser()
parser.add_argument('--cache_size', type=list, default=[5, 16, 32, 128, 256, 512, 700, 800, 1024])
parser.add_argument('--path_kg', type=list, default=["KGs/Family/family.owl", "KGs/Family/father.owl"])
parser.add_argument('--path_kge', type=list, default=None)
args = parser.parse_args()


results = []
detailed_results = []
for path_kg in args.path_kg:
    for cache_size in args.cache_size:
        result, D = run_cache(path_kg=path_kg, path_kge=args.path_kge, cache_size=cache_size) 
        results.append(result)
        detailed_results.append(D)

all_detailed_results = [item for sublist in detailed_results for item in sublist]

results = pd.DataFrame(results)
results.to_csv('caching_results/cache_experiments_0.csv')  

plot_scale_factor(results)    
plot_jaccard_vs_cache_size(results) 


print(results.to_latex(index=False))

all_detailed_results = pd.DataFrame(all_detailed_results)
all_detailed_results.to_csv('caching_results/detailed_cache_experiments_0.csv')

# bar_plot_all_data(all_detailed_results, cache_size=1024)
bar_plot_separate_data(all_detailed_results, cache_size=1024)