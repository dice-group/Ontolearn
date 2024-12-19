from ontolearn.concept_learner import NCES
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser
from ontolearn.metrics import F1
from ontolearn.learning_problem import PosNegLPStandard
import subprocess
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
        knowledge_base_path = "./KGs/Family/family-benchmark_rich_background.owl"
        path_of_embeddings = None
        if not os.path.exists(knowledge_base_path):
            raise ValueError(f"{knowledge_base_path} not found")
        if not path_of_embeddings or not os.path.exists(path_of_embeddings):
            print("\n"+"\x1b[0;30;43m"+"Embeddings not found. Will quickly train embeddings beforehand"+"\x1b[0m"+"\n")
            subprocess.run("dicee --path_single_kg knowledge_base_path --path_to_store_single_run temp_embeddings --dataset_dir None --sparql_endpoint None --backend rdflib --save_embeddings_as_csv --num_epochs 5 --model DeCaL",
             shell = True, executable="/bin/bash")
            assert os.path.exists("./temp_embeddings/DeCaL_entity_embeddings.csv"), "It seems that embeddings were not stored at the expected directory."
            path_of_embeddings = "./temp_embeddings/DeCaL_entity_embeddings.csv"
            print("\n"+"\x1b[0;30;43m"+"Will also generate some training data and train NCES for 5 epochs"+"\x1b[0m"+"\n")
            nces = NCES(knowledge_base_path=knowledge_base_path, quality_func=F1(), num_predictions=100,
                        path_of_embeddings=path_of_embeddings,
                        learner_names=["SetTransformer"])
            nces.train(epochs=5, batch_size=64, num_lps=500, storage_path="./temp_models")

        KB = KnowledgeBase(path=nces.knowledge_base_path)
        dl_parser = DLSyntaxParser(nces.kb_namespace)
        brother = dl_parser.parse('Brother')
        daughter = dl_parser.parse('Daughter')
        pos = set(KB.individuals(brother)).union(set(KB.individuals(daughter)))
        neg = set(KB.individuals())-set(pos)
        learning_problem = PosNegLPStandard(pos=pos, neg=neg)
        node = list(nces.fit(learning_problem).best_predictions)[0]
        print("Quality:", node.quality)
        assert node.quality > 0.1
    """
    def test_prediction_quality_mutagenesis(self):
        knowledge_base_path = "./KGs/Mutagenesis/mutagenesis.owl"
        path_of_embeddings = None
        if os.path.exists(knowledge_base_path) and os.path.exists(path_of_embeddings):
            nces = NCES(knowledge_base_path=knowledge_base_path, quality_func=F1(), num_predictions=100,
                        path_of_embeddings=path_of_embeddings,
                        learner_names=["LSTM", "GRU", "SetTransformer"])
            KB = KnowledgeBase(path=nces.knowledge_base_path)
            dl_parser = DLSyntaxParser(nces.kb_namespace)
            exists_inbond = dl_parser.parse('∃ hasStructure.Benzene')
            not_bond7 = dl_parser.parse('¬Bond-7')
            pos = set(KB.individuals(exists_inbond)).intersection(set(KB.individuals(not_bond7)))
            neg = sorted(set(KB.individuals()) - pos)
            if len(pos) > 500:
                pos = set(np.random.choice(list(pos), size=min(500, len(pos)), replace=False))
            neg = set(neg[:min(1000 - len(pos), len(neg))])
            learning_problem = PosNegLPStandard(pos=pos, neg=neg)
            node = list(nces.fit(learning_problem).best_predictions)[0]
            print("Quality:", node.quality)
            assert node.quality > 0.1
    """

if __name__ == "__main__":
    test = TestNCES()
    test.test_prediction_quality_family()
    #test.test_prediction_quality_mutagenesis()