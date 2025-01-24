"""
1. For NCES, run: `python examples/train_nces.py --kb /data/upb/users/n/nkouagou/profiles/unix/cs/Ontolearn/KGs/Family/family-benchmark_rich_background.owl --synthesizer NCES --path_train_data ./NCESData/family/training_data/Data.json --storage_path ./NCESData/family/ --path_temp_embeddings ./NCESData/family/embeddings`

2. For NCES2, run: `python examples/train_nces.py --kb /data/upb/users/n/nkouagou/profiles/unix/cs/Ontolearn/KGs/Family/family-benchmark_rich_background.owl --synthesizer NCES2 --path_train_data ./NCES2Data/family/training_data/Data.json --storage_path ./NCES2Data/family/`

3. For ROCES, run: `python examples/train_nces.py --kb /data/upb/users/n/nkouagou/profiles/unix/cs/Ontolearn/KGs/Family/family-benchmark_rich_background.owl --synthesizer ROCES --path_train_data ./ROCESData/family/training_data/Data.json --storage_path ./ROCESData/family/`

Note: One can leave the option `--path_train_data` and new training data will be generated on the fly. However, this would take some time.
"""


import argparse
import json, os
from ontolearn.concept_learner import NCES, NCES2, ROCES
from transformers import set_seed

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
    assert (args.kb is not None), "Argument 'kb' is required."
    training_data = None
    if args.path_train_data is not None:
        try:
            if os.path.isdir(args.path_train_data):
                with open(args.path_train_data+"/LPs.json") as file:
                    training_data = json.load(file)
                    if isinstance(training_data, dict):
                        training_data = list(training_data.items())
                    else:
                        assert isinstance(training_data, list), "The training data must either be stored as a dictionary ({'expr': {'positive examples': [], 'negative examples': []}, ...,}) or a list of items"
            else:
                with open(args.path_train_data) as file:
                    training_data = json.load(file)
                if isinstance(training_data, dict):
                    training_data = list(training_data.items())
                else:
                    assert isinstance(training_data, list), "The training data must either be stored as a dictionary ({'expr': {'positive examples': [], 'negative examples': []}, ...,}) or a list of items"
        except FileNotFoundError:
            print("Couldn't find training data in the specified path. Defaulting to generating training data.")
    if args.synthesizer == "NCES":
        synthesizer = NCES(knowledge_base_path=args.kb, learner_names=['SetTransformer', 'GRU', 'LSTM'], path_of_embeddings=args.path_of_nces_embeddings, path_temp_embeddings=args.path_temp_embeddings, auto_train=False, dicee_model=args.dicee_model, dicee_emb_dim=args.dicee_emb_dim, dicee_epochs=args.dicee_epochs, dicee_lr=args.dicee_lr, max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=args.load_pretrained, path_of_trained_models=args.path_of_trained_models, verbose=True)
    elif args.synthesizer == "NCES2":
        synthesizer = NCES2(knowledge_base_path=args.kb, auto_train=False, max_length=48, proj_dim=128, embedding_dim=args.embedding_dim,
         drop_prob=0.1, num_heads=4, num_seeds=1, m=[32, 64, 128], load_pretrained=args.load_pretrained, path_of_trained_models=args.path_of_trained_models, verbose=True)
    else:
        synthesizer = ROCES(knowledge_base_path=args.kb, auto_train=False, k=5, max_length=48, proj_dim=128, embedding_dim=args.embedding_dim,
         drop_prob=0.1, num_heads=4, num_seeds=1, m=[32, 64, 128], load_pretrained=args.load_pretrained, path_of_trained_models=args.path_of_trained_models, verbose=True)
    synthesizer.train(training_data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, clip_value=1.0, tmax=args.tmax, max_num_lps=args.max_num_lps, refinement_expressivity=args.refinement_expressivity, refs_sample_size=args.sample_size, storage_path=args.storage_path)

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', type=str, default=None, help='Paths of a knowledge base (OWL file)')
    parser.add_argument('--synthesizer', type=str, default="ROCES", help='Name of the neural synthesizer')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Number of embedding dimensions.')
    parser.add_argument('--refinement_expressivity', type=float, default=0.9, help='The expressivity of the refinement operator during training data generation')
    parser.add_argument('--max_num_lps', type=int, default=20000, help='Maximum number of learning problems to generate if no training data is provided')
    parser.add_argument('--sample_size', type=int, default=200, help='The number of concepts to sample from the refs of $\top$ during learning problem generation')
    parser.add_argument('--path_of_nces_embeddings', type=str, default=None, help='Path to a csv file containing embeddings for the KB.')
    parser.add_argument('--path_temp_embeddings', type=str, default=None, help='A directory where to store embeddings computed through the `dicee` library.')
    parser.add_argument('--path_train_data', type=str, default=None, help='Path to training data')
    parser.add_argument('--storage_path', type=str, default=None, help='Path to save the trained models')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--dicee_model', type=str, default="DeCaL", help='The model to use for DICE embeddings (only for NCES)')
    parser.add_argument('--dicee_emb_dim', type=int, default=128, help='Number of embedding dimensions for DICE embeddings (only for NCES)')
    parser.add_argument('--dicee_epochs', type=int, default=300, help='Number of training epochs for the NCES (DICE) embeddings (only for NCES)')
    parser.add_argument('--dicee_lr', type=float, default=0.01, help='Learning rate for computing DICE embeddings (only for NCES)')
    parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training. The optimizer is Adam.')
    parser.add_argument('--tmax', type=int, default=100, help='Tmax in CosineLR scheduler. The optimizer is Adam.')
    parser.add_argument('--eta_min', type=float, default=1e-4, help='eta_min in CosineLR scheduler. The optimizer is Adam.')
    parser.add_argument('--load_pretrained', type=str2bool, default=False, help='Whether to load the pretrained model')
    parser.add_argument('--path_of_trained_models', type=str, default=None, help='Path to pretrained models in case we want to finetune pretrained models')
    args = parser.parse_args()
    args.tmax = min(args.tmax, args.epochs)
    start(args)
