from ontolearn.concept_learner import NCES
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser
from ontolearn.metrics import F1
from ontolearn.learning_problem import PosNegLPStandard
import time
import random
import unittest
import os
import torch
import numpy as np

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

class TestNCES(unittest.TestCase):

    def test_prediction_quality_family(self):
        knowledge_base_path="./NCESData/family/family.owl"
        path_of_embeddings="./NCESData/family/embeddings/ConEx_entity_embeddings.csv"
        if os.path.exists(knowledge_base_path) and os.path.exists(path_of_embeddings):
            nces = NCES(knowledge_base_path=knowledge_base_path, quality_func=F1(), num_predictions=100,
                        path_of_embeddings=path_of_embeddings,
                        learner_names=["LSTM", "GRU", "SetTransformer"])
            KB = KnowledgeBase(path=nces.knowledge_base_path)
            dl_parser = DLSyntaxParser(nces.kb_namespace)
            brother = dl_parser.parse('Brother')
            daughter = dl_parser.parse('Daughter')
            pos = set(KB.individuals(brother)).union(set(KB.individuals(daughter)))
            neg = set(KB.individuals())-set(pos)
            learning_problem = PosNegLPStandard(pos=pos, neg=neg)
            node = list(nces.fit(learning_problem).best_predictions)[0]
            print("Quality:", node.quality)
            assert node.quality > 0.95

    def test_prediction_quality_mutagenesis(self):
        knowledge_base_path="./NCESData/mutagenesis/mutagenesis.owl"
        path_of_embeddings="./NCESData/mutagenesis/embeddings/ConEx_entity_embeddings.csv"
        if os.path.exists(knowledge_base_path) and os.path.exists(path_of_embeddings):
            nces = NCES(knowledge_base_path=knowledge_base_path, quality_func=F1(), num_predictions=100,
                        path_of_embeddings=path_of_embeddings,
                        learner_names=["LSTM", "GRU", "SetTransformer"])
            KB = KnowledgeBase(path=nces.knowledge_base_path)
            dl_parser = DLSyntaxParser(nces.kb_namespace)
            exists_inbond = dl_parser.parse('∃ hasStructure.Benzene')
            not_bond7 = dl_parser.parse('¬Bond-7')
            pos = set(KB.individuals(exists_inbond)).intersection(set(KB.individuals(not_bond7)))
            neg = sorted(set(KB.individuals())-pos)
            if len(pos) > 500:
                pos = set(np.random.choice(list(pos), size=min(500, len(pos)), replace=False))
            neg = set(neg[:min(1000-len(pos), len(neg))])
            learning_problem = PosNegLPStandard(pos=pos, neg=neg)
            node = list(nces.fit(learning_problem).best_predictions)[0]
            print("Quality:", node.quality)
            assert node.quality > 0.95
        
if __name__ == "__main__":
    test = TestNCES()
    test.test_prediction_quality_family()
    test.test_prediction_quality_mutagenesis()