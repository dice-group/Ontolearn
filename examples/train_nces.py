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


parser = argparse.ArgumentParser()
parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], help='Knowledge base name(s)')
parser.add_argument('--models', type=str, nargs='+', default=['SetTransformer', 'LSTM', 'GRU'], help='Neural models')
parser.add_argument('--load_pretrained', type=str2bool, default=False, help='Whether to load the pretrained model')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
args = parser.parse_args()

for kb in args.kbs:
    knowledge_base_path = f"./NCESData/{kb}/{kb}.owl"
    path_of_embeddings = f"./NCESData/{kb}/embeddings/ConEx_entity_embeddings.csv"
    with open(f"./NCESData/{kb}/training_data/Data.json") as file:
        training_data = list(json.load(file).items())

    nces = NCES(knowledge_base_path=knowledge_base_path, learner_name="SetTransformer",
                path_of_embeddings=path_of_embeddings, max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1,
                num_heads=4, num_seeds=1, num_inds=32, load_pretrained=args.load_pretrained)

    for model in args.models:
        nces.learner_name = model
        nces.pretrained_model_name = model
        nces.refresh()
        nces.train(training_data, epochs=args.epochs, learning_rate=args.learning_rate, save_model=True)
