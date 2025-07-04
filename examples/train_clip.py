import argparse
import json, os
from ontolearn.concept_learner import NCES, NCES2, ROCES, CLIP
from transformers import set_seed
import time
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.lp_generator import LPGen
from ontolearn.refinement_operators import ExpressRefinement


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
                with open(args.path_train_data + "/LPs.json") as file:
                    training_data = json.load(file)
                    if isinstance(training_data, dict):
                        training_data = list(training_data.items())
                    else:
                        assert isinstance(training_data,
                                          list), "The training data must either be stored as a dictionary ({'expr': {'positive examples': [], 'negative examples': []}, ...,}) or a list of items"
            else:
                with open(args.path_train_data) as file:
                    training_data = json.load(file)
                if isinstance(training_data, dict):
                    training_data = list(training_data.items())
                else:
                    assert isinstance(training_data,
                                      list), "The training data must either be stored as a dictionary ({'expr': {'positive examples': [], 'negative examples': []}, ...,}) or a list of items"
        except FileNotFoundError:
            print(
                "Couldn't find training data in the specified path. Defaulting to generating training data.")

    # use clip here
    knowledge_base_path = args.kb
    path_of_embeddings = args.path_of_embeddings
    if os.path.exists(knowledge_base_path):
        KB = KnowledgeBase(path=knowledge_base_path)
        op = ExpressRefinement(knowledge_base=KB, use_inverse=False,
                               use_numeric_datatypes=False)
        clip = CLIP(knowledge_base=KB, path_of_embeddings=path_of_embeddings,
                    refinement_operator=op, predictor_name=args.predictor_name,
                    load_pretrained=args.load_pretrained,
                    pretrained_predictor_name=args.pretrained_predictor_name,
                    max_runtime=args.max_runtime, num_workers=args.num_workers)
        clip.train(training_data, epochs=args.epochs, learning_rate=args.lr,
                   storage_path=args.storage_path)
    else:
        print("Knowledge base or embeddings not found. Please check the paths.")


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', type=str, default=None, help='Paths of a knowledge base (OWL file)')
    parser.add_argument('--path_train_data', type=str, default=None, help='Path to training data')
    parser.add_argument('--path_of_embeddings', type=str, default=None,
                        help='Path to CLIP-compatible embeddings')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training. The optimizer is Adam.')
    parser.add_argument('--predictor_name', type=str, default=None,
                        help='Name of the length predictor to be used. The options are: SetTransformer, GRU, LSTM')
    parser.add_argument('--load_pretrained', type=str2bool, default=True,
                        help='Whether to load the pretrained model')
    parser.add_argument('--pretrained_predictor_name', type=str, default='SetTransformer',
                        help='Name of the pretrained length predictor to be used. The options are: SetTransformer, GRU, LSTM, CNN')
    parser.add_argument('--storage_path', type=str, default=None,
                        help='Path to save the trained models')
    parser.add_argument('--max_runtime', type=int, default=60,
                        help='Maximum runtime in for clip')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')

    args = parser.parse_args()
    # make training data
    # data_generator = LPGen(kb_path= args.kb, storage_path=args.storage_path)
    # data_generator.generate()
    args.tmax = args.epochs
    start(args)
