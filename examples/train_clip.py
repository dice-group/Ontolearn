import argparse, json, os, random
import torch, numpy as np
from ontolearn.concept_learner import CLIP
from ontolearn.knowledge_base import KnowledgeBase

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument("--kb_name", type=str, nargs="+", default=["carcinogenesis"])
parser.add_argument("--predictor_name", type=str, nargs="+", default=["SetTransformer"])
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--learning_rate", type=float, default=1e-3)

args = parser.parse_args()

for kb_name in args.kb_name:
    print(f'Dataset: {kb_name.capitalize()}')
    kb = KnowledgeBase(path=f"../CLIPData/{kb_name}/{kb_name}.owl")

    with open(f"../CLIPData/{kb_name}/LPs.json") as file:
        training_data = list(json.load(file).items())

    print("\nTraining data size:", len(training_data))
    print()

    for predictor_name in args.predictor_name:
        clip = CLIP(knowledge_base=kb, knowledge_base_path=f"../CLIPData/{kb_name}/{kb_name}.owl", output_size=15,
                    path_of_embeddings=f"../CLIPData/{kb_name}/embeddings/ConEx_entity_embeddings.csv", predictor_name=predictor_name)

        clip.train(training_data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                   shuffle_examples=True, storage_path=f"../CLIPData/{kb_name}")