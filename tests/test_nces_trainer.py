from ontolearn.learners import NCES
from ontolearn.knowledge_base import KnowledgeBase
import contextlib
import random
import tempfile
import unittest
import os
import numpy as np
import torch
import pathlib
import warnings

warnings.filterwarnings("ignore")

base_path = pathlib.Path(__file__).parent.resolve()._str
knowledge_base_path = base_path[:base_path.rfind("/")+1] + "KGs/Family/family-benchmark_rich_background.owl"

# Reuse the shipped embeddings so the test does not train them with a dicee subprocess.
pretrained_embeddings = os.path.join(os.path.dirname(base_path), "NCESData", "family", "embeddings", "DeCaL_entity_embeddings.csv")


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
    def setUp(self):
        # NCES-family learners write experiment/temp directories into the cwd; run in a throwaway dir.
        self.enterContext(contextlib.chdir(self.enterContext(tempfile.TemporaryDirectory())))

    def test_trainer_family(self):
        KB = KnowledgeBase(path=knowledge_base_path)
        nces = NCES(knowledge_base = KB, learner_names=['SetTransformer', 'GRU', 'LSTM'], path_of_embeddings=pretrained_embeddings, auto_train=False,
            max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=False, verbose=True)
        nces.train(data=None, epochs=1, max_num_lps=200, refinement_expressivity=0.1) #give it lps as data to observe
if __name__ == "__main__":
    test = TestNCESTrainer()
    test.test_trainer_family()
    print("\nDone.\n")