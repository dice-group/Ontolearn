
echo "Reproduce Our Experiments"
# DL-learner Binaries
path_dl_learner=$PWD'/dllearner-1.4.0/'

# Datasets
family_dataset_path=$PWD'/KGs/Family/family-benchmark_rich_background.owl'
carcinogenesis_dataset_path=$PWD'/KGs/Carcinogenesis/carcinogenesis.owl'
mutagenesis_dataset_path=$PWD'/KGs/Mutagenesis/mutagenesis.owl'
biopax_dataset_path=$PWD'/KGs/Biopax/biopax.owl'

# Benchmark Learning Problems
family_benchmark_lp_path=$PWD'/LPs/Family/lp.json'
carcinogenesis_benchmark_lp_path=$PWD'/LPs/Carcinogenesis/lp.json'
mutagenesis_benchmark_lp_path=$PWD'/LPs/Mutagenesis/lp.json'
biopax_benchmark_lp_path=$PWD'/LPs/Biopax/lp.json'

# Embeddings
family_kge=$PWD'/embeddings/ConEx_Family/ConEx_entity_embeddings.csv'
carcinogenesis_kge=$PWD'/embeddings/Shallom_Carcinogenesis/Shallom_entity_embeddings.csv'
mutagenesis_kge=$PWD'/embeddings/ConEx_Mutagenesis/ConEx_entity_embeddings.csv'
biopax_kge=$PWD'/embeddings/ConEx_Biopax/ConEx_entity_embeddings.csv'

# Pretrained Models
drill_avg_path_family=$PWD'/pre_trained_agents/Family/DrillHeuristic_averaging/DrillHeuristic_averaging.pth'
drill_avg_path_carcinogenesis=$PWD'/pre_trained_agents/Carcinogenesis/DrillHeuristic_averaging/DrillHeuristic_averaging.pth'
drill_avg_path_mutagenesis=$PWD'/pre_trained_agents/Mutagenesis/DrillHeuristic_averaging/DrillHeuristic_averaging.pth'
drill_avg_path_biopax=$PWD'/pre_trained_agents/Biopax/DrillHeuristic_averaging/DrillHeuristic_averaging.pth'


echo "Start Testing on Family on automatically generated learning problems"
python experiments_standard.py --path_lp "$family_benchmark_lp_path" --path_knowledge_base "$family_dataset_path" --path_knowledge_base_embeddings "$family_kge" --pretrained_drill_avg_path "$drill_avg_path_family" --path_dl_learner "$path_dl_learner"
echo "Start Testing on Carcinogenesis on automatically generated learning problems"
python experiments_standard.py --path_lp "$carcinogenesis_benchmark_lp_path" --path_knowledge_base "$carcinogenesis_dataset_path" --path_knowledge_base_embeddings  "$carcinogenesis_kge" --pretrained_drill_avg_path "$drill_avg_path_carcinogenesis" --path_dl_learner $path_dl_learner
echo "Start Testing on Mutagenesis on automatically generated learning problems"
python experiments_standard.py --path_lp "$mutagenesis_benchmark_lp_path" --path_knowledge_base "$mutagenesis_dataset_path" --path_knowledge_base_embeddings  "$mutagenesis_kge" --pretrained_drill_avg_path "$drill_avg_path_mutagenesis" --path_dl_learner "$path_dl_learner"
echo "Start Testing on Biopax on automatically generated learning problems"
python experiments_standard.py --path_lp "$biopax_benchmark_lp_path" --path_knowledge_base "$biopax_dataset_path" --path_knowledge_base_embeddings  "$biopax_kge" --pretrained_drill_avg_path "$drill_avg_path_biopax" --path_dl_learner $path_dl_learner
