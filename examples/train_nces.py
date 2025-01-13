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
        synthesizer = NCES(knowledge_base_path=args.kb, learner_names=['SetTransformer', 'GRU', 'LSTM'], path_of_embeddings=args.path_of_embeddings, auto_train=False,
            max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=args.load_pretrained, verbose=True)
    elif args.synthesizer == "NCES2":
        synthesizer = NCES2(knowledge_base_path=args.kb, auto_train=False, max_length=48, proj_dim=128,
         drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=args.load_pretrained, verbose=True)
    else:
        synthesizer = ROCES(knowledge_base_path=args.kb, auto_train=False, k=5, max_length=48, proj_dim=128,
         drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=args.load_pretrained, verbose=True)
    synthesizer.train(training_data, epochs=args.epochs, max_num_lps=args.max_num_lps, refinement_expressivity=args.refinement_expressivity)

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', type=str, default=None, help='Paths of a knowledge base (OWL file)')
    parser.add_argument('--synthesizer', type=str, default="ROCES", help='Name of the neural synthesizer')
    parser.add_argument('--refinement_expressivity', type=float, default=0.9, help='The expressivity of the refinement operator during training data generation')
    parser.add_argument('--max_num_lps', type=int, default=20000, help='Maximum number of learning problems to generate if no training data is provided')
    parser.add_argument('--path_of_embeddings', type=str, default=None, help='Path to a csv file containing embeddings for the KB.')
    parser.add_argument('--path_train_data', type=str, default=None, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--load_pretrained', type=str2bool, default=False, help='Whether to load the pretrained model')
    start(parser.parse_args())
