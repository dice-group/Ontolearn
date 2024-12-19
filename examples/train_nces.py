"""
(1) To get the data: wget https://hobbitdata.informatik.uni-leipzig.de/NCES_Ontolearn_Data/NCESData.zip
(2) pip install ontolearn
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
    assert (args.kbs is not None), "Argument 'kbs' is required."
    for i, knowledge_base_path in enumerate(args.kbs):
        if args.embeddings:
            path_of_embeddings = args.embeddings[i]
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
                for idx in range(len(training_data)):
                    if len(training_data[idx]) != 2:
                        print("****\n",training_data[idx])
            except FileNotFoundError:
                print("Couldn't find training data in the specified path. Defaulting to generating training data.")
        elif args.synthesizer == "NCES":
            print("Could not find training data. Will generate some data and train.")
            training_data = NCES.generate_training_data(knowledge_base_path)
        else:
            print("Could not find training data. Will generate some data and train.")
            training_data = NCES2.generate_training_data(knowledge_base_path, beyond_alc=True)
        if args.synthesizer == "NCES":
            synthesizer = NCES(knowledge_base_path=knowledge_base_path, learner_names=args.models, path_of_embeddings=path_of_embeddings, path_of_trained_models=args.path_of_trained_models,
                max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=args.load_pretrained, verbose=True)
        elif args.synthesizer == "NCES2":
            synthesizer = NCES2(knowledge_base_path=knowledge_base_path, path_of_trained_models=args.path_of_trained_models, nces2_or_roces=True, max_length=48, proj_dim=128, 
                drop_prob=0.1, num_heads=4, num_seeds=1, m=32, verbose=True, load_pretrained=args.load_pretrained)
        else:
            synthesizer = ROCES(knowledge_base_path=knowledge_base_path, path_of_trained_models=args.path_of_trained_models, nces2_or_roces=True, k=5, max_length=48, proj_dim=128, 
                drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=args.load_pretrained, verbose=True)
        synthesizer.train(training_data, epochs=args.epochs, learning_rate=args.learning_rate, num_workers=2, save_model=True, storage_path=args.storage_dir)


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--kbs', type=str, nargs='+', default=None, help='Paths of knowledge bases (OWL files)')
    parser.add_argument('--embeddings', type=str, nargs='+', default=None, help='Paths of embeddings for each KB.')
    parser.add_argument('--synthesizer', type=str, default="NCES", help='Neural synthesizer to train')
    parser.add_argument('--path_train_data', type=str, help='Path to training data')
    parser.add_argument('--path_of_trained_models', type=str, default=None, help='Path to training data')
    parser.add_argument('--storage_dir', type=str, default=None, help='Path to training data')
    parser.add_argument('--models', type=str, nargs='+', default=['SetTransformer', 'LSTM', 'GRU'],
                        help='Neural models')
    parser.add_argument('--load_pretrained', type=str2bool, default=False, help='Whether to load the pretrained model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    start(parser.parse_args())
