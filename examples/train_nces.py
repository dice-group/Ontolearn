"""
(1) To get the data: wget https://hobbitdata.informatik.uni-leipzig.de/NCES_Ontolearn_Data/NCESData.zip
(2) pip install ontolearn
"""


from ontolearn.concept_learner import NCES
import argparse
import json


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
    assert (args.embeddings is not None), "Argument 'embeddings' is required."
    assert (len(args.kbs) == len(args.embeddings)), "There should be embeddings for each knowledge base."
    for i, knowledge_base_path in enumerate(args.kbs):
        path_of_embeddings = args.embeddings[i]
        training_data = None
        if args.path_train_data is not None:
            try:
                with open(args.path_train_data+"/LPs.json") as file:
                    training_data = list(json.load(file).items())
            except FileNotFoundError:
                print("Couldn't find training data in the specified path. Defaulting to generating training data.")
        else:
            print("Could not find training data. Will generate some data and train.")


        nces = NCES(knowledge_base_path=knowledge_base_path, learner_names=args.models,
                    path_of_embeddings=path_of_embeddings, max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1,
                    num_heads=4, num_seeds=1, num_inds=32, verbose=True, load_pretrained=args.load_pretrained)

        nces.train(training_data, epochs=args.epochs, learning_rate=args.learning_rate, num_workers=2, save_model=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kbs', type=str, nargs='+', default=None, help='Paths of knowledge bases')
    parser.add_argument('--embeddings', type=str, nargs='+', default=None, help='Paths of embeddings for each KB.')
    parser.add_argument('--path_train_data', type=str, help='Path to training data')
    parser.add_argument('--models', type=str, nargs='+', default=['SetTransformer', 'LSTM', 'GRU'],
                        help='Neural models')
    parser.add_argument('--load_pretrained', type=str2bool, default=False, help='Whether to load the pretrained model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')

    start(parser.parse_args())
