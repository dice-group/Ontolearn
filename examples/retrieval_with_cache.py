
import argparse 
import pandas as pd
from semantic_caching import run_cache, concept_generator
from plot_metrics import *
import seaborn as sns

#5, 16, 32, 128, 256, 512, 700, 800, 1024, , "KGs/Family/family.owl" .1, .2, .4, .8, 1.
parser = argparse.ArgumentParser()
parser.add_argument('--cache_size_ratios', type=list, default=[.1, .2, .4, .8, 1.], help="cache size is proportional to num_concepts, cache size = k * num_concepts")
parser.add_argument('--path_kg', type=list, default=["KGs/Family/father.owl"])
parser.add_argument('--path_kge', type=list, default=None)
parser.add_argument('--name_reasoner', type=str, default='EBR', choices=["EBR",'HermiT', 'Pellet', 'JFact', 'Openllet'])
parser.add_argument('--eviction_strategy', type=str, default='MRU', choices=['LIFO', 'FIFO', 'LRU', 'MRU', 'RP'])
parser.add_argument('--random_seed_for_RP', type=int, default=10, help="Random seed if the eviction startegy is RP")
args = parser.parse_args()

def get_cache_size(list_k, path_kg):
    
    data_size = len(concept_generator(path_kg))

    return [max(1, int(k * data_size)) for k in list_k]



# results = []
# for path_kg in args.path_kg:
#     for cache_size in get_cache_size(args.cache_size_ratios, path_kg):
#         for strategy in ['MRU']:#, 'FIFO', 'LRU', 'MRU', 'RP']:
#             result, detailed = run_cache(
#                 path_kg=path_kg,
#                 path_kge=args.path_kge,
#                 cache_size=cache_size,
#                 name_reasoner=args.name_reasoner,
#                 eviction=strategy,
#                 random_seed=args.random_seed_for_RP
#             )
#             results.append(result)

#     data_kg = result['dataset']
#     df = pd.DataFrame(results)
#     print(df)

    # Save to CSV
    # df.to_csv(f'caching_results_{data_kg}/cache_experiments_{args.name_reasoner}_{data_kg}.csv', index=False)


# name_reasoners = ["EBR",'HermiT','Pellet','JFact','Openllet']
# data_kgs = ["family"]

# for data_kg in data_kgs:

#     for name_reasoner in name_reasoners:

#         df = pd.read_csv(f'caching_results_{data_kg}/cache_experiments_{name_reasoner}_{data_kg}.csv')
#         print(df)


        # sns.set_context("talk", font_scale=3.6)

        # plot1 = sns.catplot(
        # data=df,
        # kind="bar",
        # x="cache_size",
        # y="hit_ratio",
        # hue="strategy",
        # col="dataset",
        # height=10,
        # aspect=2
        # )
        # plt.show()
        # plot1.savefig(f'caching_results_{data_kg}/cache_vs_hit_sns_{name_reasoner}_{data_kg}.pdf')


        # plot2 = sns.catplot(
        # data=df,
        # kind="bar",
        # x="cache_size",
        # y="avg_jaccard",
        # hue="strategy",
        # col="dataset",
        # height=10,
        # aspect=2
        # )
        # plt.show()
        # plot2.savefig(f'caching_results_{data_kg}/cache_vs_jaccard_sns_{name_reasoner}_{data_kg}.pdf')


        # plot3 = sns.catplot(esults = []
# detailed_results = []
# for path_kg in args.path_kg:
#     for cache_size in get_cache_size(args.cache_size_ratios, path_kg):
#         result, D = run_cache(path_kg=path_kg, path_kge=args.path_kge, cache_size=cache_size, name_reasoner=args.name_reasoner,\
#                               eviction=args.eviction_strategy, random_seed=args.random_seed_for_RP) 
#         results.append(result)
#         detailed_results.append(D)

# all_detailed_results = [item for sublist in detailed_results for item in sublist]

# results = pd.DataFrame(results)
# results.to_csv(f'caching_results/cache_experiments_{args.name_reasoner}.csv')  

# plot_scale_factor(results, args.name_reasoner)    
# plot_jaccard_vs_cache_size(results, args.name_reasoner) 


# # print(results.to_latex(index=False))

# all_detailed_results = pd.DataFrame(all_detailed_results)
# bar_plot_separate_data(all_detailed_results, cache_size=90, name_reasoner=args.name_reasoner)
        # data=df,
        # kind="bar",
        # x="cache_size",
        # y="RT_cache",
        # hue="strategy",
        # col="dataset",
        # height=10,
        # aspect=2
        # )
        # plt.show()
        # plot3.savefig(f'caching_results_{data_kg}/cache_vs_RT_sns_{name_reasoner}_{data_kg}.pdf')



results = []
detailed_results = []
for path_kg in args.path_kg:
    for cache_size in get_cache_size(args.cache_size_ratios, path_kg):
        result, D = run_cache(path_kg=path_kg, path_kge=args.path_kge, cache_size=cache_size, name_reasoner=args.name_reasoner,\
                              eviction=args.eviction_strategy, random_seed=args.random_seed_for_RP) 
        results.append(result)
        detailed_results.append(D)

all_detailed_results = [item for sublist in detailed_results for item in sublist]

results = pd.DataFrame(results)
results.to_csv(f'caching_results/cache_experiments_{args.name_reasoner}.csv')  

plot_scale_factor(results, args.name_reasoner)    
plot_jaccard_vs_cache_size(results, args.name_reasoner) 

# # print(results.to_latex(index=False))

all_detailed_results = pd.DataFrame(all_detailed_results)
bar_plot_separate_data(all_detailed_results, cache_size=90, name_reasoner=args.name_reasoner)
bar_plot_all_data(all_detailed_results, cache_size=90, name_reasoner=args.name_reasoner)
# all_detailed_results.to_csv(f'caching_results/detailed_cache_experiments_{args.name_reasoner}.csv')

