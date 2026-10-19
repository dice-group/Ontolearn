from ontolearn.learners import NCES
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser
from ontolearn.learning_problem import PosNegLPStandard
import contextlib
import random
import tempfile
import unittest
import os
import torch
import numpy as np
import pathlib
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

base_path = pathlib.Path(__file__).parent.resolve()._str

class TestNCES(unittest.TestCase):

    def setUp(self):
        # NCES-family learners write experiment/temp directories into the cwd; run in a throwaway dir.
        self.enterContext(contextlib.chdir(self.enterContext(tempfile.TemporaryDirectory())))

    def test_prediction_quality_family(self):
        knowledge_base_path = base_path[:base_path.rfind("/")+1] + "KGs/Family/family-benchmark_rich_background.owl"
        KB = KnowledgeBase(path=knowledge_base_path)
        # Use the shipped pretrained model/embeddings (NCESData.zip): no training, so this runs in seconds.
        ncesdata = os.path.join(os.path.dirname(base_path), "NCESData", "family")
        model = NCES(knowledge_base = KB, learner_names=['SetTransformer'],
                     path_of_embeddings=os.path.join(ncesdata, "embeddings", "DeCaL_entity_embeddings.csv"),
                     path_of_trained_models=os.path.join(ncesdata, "trained_models"), auto_train=False,
                     max_length=48, proj_dim=128, rnn_n_layers=2, drop_prob=0.1, num_heads=4, num_seeds=1, m=32, load_pretrained=True, verbose=True)
        dl_parser = DLSyntaxParser(model.kb_namespace)
        brother = dl_parser.parse('Brother')
        daughter = dl_parser.parse('Daughter')
        pos = set(KB.individuals(brother)).union(set(KB.individuals(daughter)))
        neg = set(KB.individuals())-set(pos)
        learning_problem = PosNegLPStandard(pos=pos, neg=neg)
        node = list(model.fit(learning_problem).best_predictions)[0]
        print("Quality:", node.quality)
        assert node.quality > 0.4
    
if __name__ == "__main__":
    test = TestNCES()
    test.test_prediction_quality_family()