from ontolearn.concept_learner import NCES
import random
import unittest
import os
import numpy as np
import torch
import pathlib
import warnings

warnings.filterwarnings("ignore")

base_path = pathlib.Path(__file__).parent.resolve()._str
knowledge_base_path = base_path[:base_path.rfind("/")+1] + "KGs/Family/family-benchmark_rich_background.owl"

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
        nces = NCES(knowledge_base_path=knowledge_base_path, learner_names=['SetTransformer', 'GRU', 'LSTM'], path_of_embeddings=None, auto_train=False,
            max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=False, verbose=True)
        nces.train(data=None, epochs=5, max_num_lps=1000, refinement_expressivity=0.1)
if __name__ == "__main__":
    test = TestNCESTrainer()
    test.test_trainer_family()
    print("\nDone.\n")