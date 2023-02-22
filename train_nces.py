from ontolearn.concept_learner import NCES
import argparse, json


parser = argparse.ArgumentParser()
parser.add_argument('--kb', type=str, default='carcinogenesis', help='Knowledge base name')
parser.add_argument('--models', type=str, nargs='+', default=['SetTransformer', 'LSTM', 'GRU'], help='Neural models')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
args = parser.parse_args()

knowledge_base_path = f"./NCESData/{args.kb}/{args.kb}.owl"
path_of_embeddings = f"./NCESData/{args.kb}/embeddings/ConEx_entity_embeddings.csv"
with open(f"./NCESData/{args.kb}/training_data/Data.json") as file:
    training_data = list(json.load(file).items())

nces = NCES(knowledge_base_path=knowledge_base_path, learner_name="SetTransformer",
     path_of_embeddings=path_of_embeddings, max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4, num_seeds=1, num_inds=32, load_pretrained=False)

for model in args.models:
    nces.learner_name = model
    nces.refresh()
    nces.train(training_data, epochs=args.epochs, learning_rate=args.learning_rate, save_model=True)
