from ontolearn.concept_learner import CLIP
from ontolearn.refinement_operators import ExpressRefinement
from ontolearn.knowledge_base import KnowledgeBase
import time
import numpy as np
import torch
import os
import json
import random
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


class TestCLIPTrainer:

    def test_trainer_family(self):
        knowledge_base_path="./CLIPData/family/family.owl"
        path_of_embeddings="./CLIPData/family/embeddings/ConEx_entity_embeddings.csv"
        if os.path.exists(knowledge_base_path) and os.path.exists(path_of_embeddings):
            KB = KnowledgeBase(path="./CLIPData/family/family.owl")
            op = ExpressRefinement(knowledge_base=KB, use_inverse=False,
                              use_numeric_datatypes=False)
            clip = CLIP(knowledge_base=KB, path_of_embeddings="./CLIPData/family/embeddings/ConEx_entity_embeddings.csv",
                 refinement_operator=op, load_pretrained=True, max_runtime=60)
            with open("./CLIPData/family/LPs.json") as f:
                data = json.load(f)
            clip.train(list(data.items())[-100:], epochs=5, learning_rate=0.001, save_model=False, record_runtime=False, storage_path=f"./CLIP-{time.time()}/")
