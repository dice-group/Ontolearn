import os

import torch

BASE_DATA_DIR = 'data'
os.makedirs(BASE_DATA_DIR, exist_ok=True)

CONFIG = {
    'seed': 42,
    'd_model': 128,
    'nhead': 1,
    'num_layers': 8,
    'dim_feedforward': 512,
    'use_checkpointing': True,
    'pre_norm': True,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'num_epochs': 300,
    'batch_size': 1,
    # 'max_global_seq_len': 1500,
    'max_output_seq_len': 30,
    'log_interval': 5,
    'grad_norm_clip': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'KNOWLEDGE_BASE_PATH': os.path.join(BASE_DATA_DIR, "KGs/Family/family-benchmark_rich_background.owl"),
    'LEARNING_PROBLEM_PATH': os.path.join(BASE_DATA_DIR, "LPs/Family/lps.json"),
    'split_dataset': True,

    # --- Data Augmentation Configuration ---
    # "apply_task_label_logical_aug": False, # Global flag to enable/disable logical aug for task labels
    # "task_label_neg_sample_ratio": 0.0, # Probability of negating an original task label (0.0 to 1.0)
    # "task_label_feat_aug_sample_ratio": 0.2,
    # "individual_feat_aug_sample_ratio": 0.5, # Prob. an individual's features undergo logical content augmentation (0.0 to 1.0)
    # "apply_indv_feat_rand_aug": False, # Global flag to enable/disable individual feature randomization (shuffling)
    # "indv_feat_sample_ratio_for_shuffle": 0.5, # Ratio of individuals whose features will be considered for shuffling for randomization
    # "indv_feat_shuffle_ratio": 0.5, # Ratio of features within a selected individual to shuffle for randomization (0.0 to 1.0)
    'num_dataloader_workers': 0
}

owl_path = CONFIG['KNOWLEDGE_BASE_PATH']
base_folder_name = os.path.basename(os.path.dirname(owl_path))
experiment_dir = os.path.join("experiments", base_folder_name.lower())

# if CONFIG["apply_task_label_logical_aug"]:
#     experiment_dir = os.path.join(experiment_dir, "augment")

os.makedirs(experiment_dir, exist_ok=True)

expr_data = experiment_dir+'/data'

CONFIG['EXPERIMENT_DIR'] = experiment_dir
CONFIG['GENERATED_DATA_PATH'] =  expr_data+'/generated_raw_data.json' 
CONFIG['TASK_LABEL_MAPPING_PATH'] =  expr_data+'/task_label_mappings.json'

expr_fit_data = expr_data+'/fit'

CONFIG['FIT_PATH'] = {
    'GENERATED_DATA_PATH': expr_fit_data + '/generated_raw_data.json',
    'TASK_LABEL_MAPPING_PATH': expr_fit_data + '/task_label_mappings.json'
}