import os
import torch
from torch.optim import Adam, AdamW, SGD, Optimizer
#from cnir.utils import ADOPT
from datetime import datetime


class BaseTrainer:
    def __init__(self, model, data, embeddings, all_individuals, concept_to_instance_set, num_examples, th, optimizer, output_dir=None, epochs=300, batch_size=256, num_workers=4, lr=3.5e-4):
        self.model = model
        self.data = data
        self.embeddings = embeddings
        self.all_individuals = all_individuals
        self.concept_to_instance_set = concept_to_instance_set
        self.num_examples = num_examples
        self.th = th
        optimizer_map = {"adam": Adam, "adamw": AdamW, "sgd": SGD}
        if isinstance(optimizer, Optimizer) and lr:
            optimizer.param_groups[0]['lr'] = lr
        elif lr:
            assert isinstance(optimizer, str), f"optimizer must be either a string or an instance of torch.optim.Optimizer, got {type(optimizer)}"
            optimizer = optimizer_map[optimizer](self.model.parameters())
            optimizer.param_groups[0]['lr'] = lr
        else:
            assert isinstance(optimizer, Optimizer), f"Got unkown optimizer type: {type(optimizer)}"
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if output_dir is None or not os.path.exists(output_dir):
            if output_dir is None:
                output_dir = "./TrainerOutput" + datetime.now().isoformat(timespec='minutes')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        
class BaseDataset:
    def __init__(self, data, all_individuals, embeddings):
        self.data = data
        self.embeddings = embeddings
        self.all_individuals = all_individuals
        
        
    def __len__(self):
        pass
          
    def __getitem__(self, idx):
        pass