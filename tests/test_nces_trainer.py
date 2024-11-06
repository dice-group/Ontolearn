from ontolearn.concept_learner import NCES
import time
import random
import unittest
import os
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

def seed_everything():
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('-----Seed Set!-----')

seed_everything()

class TestNCESTrainer(unittest.TestCase):

    def test_trainer_family(self):
        knowledge_base_path="./NCESData/family/family.owl"
        path_of_embeddings="./NCESData/family/embeddings/ConEx_entity_embeddings.csv"
        if os.path.exists(knowledge_base_path) and os.path.exists(path_of_embeddings):
            nces = NCES(knowledge_base_path=knowledge_base_path, num_predictions=100,
                        path_of_embeddings=path_of_embeddings,
                        load_pretrained=False)
            with open("./NCESData/family/training_data/Data.json") as f:
                data = json.load(f)
            nces.train(list(data.items())[-100:], epochs=5, learning_rate=0.001, save_model=False, record_runtime=False, storage_path=f"./NCES-{time.time()}/")
if __name__ == "__main__":
    test = TestNCESTrainer()
    test.test_trainer_family()
    print("\nDone.\n")