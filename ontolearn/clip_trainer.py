import numpy as np
import copy
import torch
from tqdm import trange
from collections import defaultdict
import os
import json
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score
import time



class CLIPTrainer:
    """CLIP trainer."""
    def __init__(self, clip, epochs=300, learning_rate=1e-4, decay_rate=0, clip_value=5.0,
                 storage_path="./"):
        self.clip = clip
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.clip_value = clip_value
        self.storage_path = storage_path

    def compute_eval_metric(self, target, prediction):
        f1 = 100*f1_score(target, prediction, average="micro")
        acc = 100*accuracy_score(target, prediction)
        return f1, acc

    def get_optimizer(self, length_predictor, optimizer='Adam'):
        if optimizer == 'Adam':
            return torch.optim.Adam(length_predictor.parameters(), lr=self.learning_rate)
        elif optimizer == 'SGD':
            return torch.optim.SGD(length_predictor.parameters(), lr=self.learning_rate)
        elif optimizer == 'RMSprop':
            return torch.optim.RMSprop(length_predictor.parameters(), lr=self.learning_rate)
        else:
            raise ValueError
            print('Unsupported optimizer')

    def show_num_learnable_params(self):
        print("*"*20+"Trainable model size"+"*"*20)
        size = sum([p.numel() for p in self.clip.length_predictor.parameters()])
        size_ = 0
        print("Length Predictor: ", size)
        print("*"*20+"Trainable model size"+"*"*20)
        print()
        return size
    
    def train(self, train_dataloader, save_model=True, optimizer='Adam', record_runtime=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.clip.length_predictor, list):
            self.clip.length_predictor = copy.deepcopy(self.clip.length_predictor[0])
        model_size = self.show_num_learnable_params()
        if device.type == "cpu":
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")
        print()
        print("#"*50)
        print()
        print("{} starts training... \n".format(self.clip.length_predictor.name))
        print("#"*50, "\n")
        length_predictor = copy.deepcopy(self.clip.length_predictor).train()
        desc = length_predictor.name
        if device.type == "cuda":
            length_predictor.cuda()
        opt = self.get_optimizer(length_predictor=length_predictor, optimizer=optimizer)
        if self.decay_rate:
            self.scheduler = ExponentialLR(opt, self.decay_rate)
        Train_loss = []
        F1, Acc = [], []
        best_score = 0.
        if record_runtime:
            t0 = time.time()
        Epochs = trange(self.epochs, desc=f'Loss: {np.nan}, F1: {np.nan}, Acc: {np.nan}', leave=True)
        for e in Epochs:
            f1s, accs = [], []
            train_losses = []
            for x1, x2, labels in train_dataloader:
                if device.type == "cuda":
                    x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
                scores = length_predictor(x1, x2)
                loss = length_predictor.loss(scores, labels)
                predictions = scores.argmax(1).detach().cpu().numpy()
                f1, acc = self.compute_eval_metric(labels.cpu().numpy(), predictions)
                f1s.append(f1)
                accs.append(acc)
                train_losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                clip_grad_value_(length_predictor.parameters(), clip_value=self.clip_value)
                opt.step()
                if self.decay_rate:
                    self.scheduler.step()
            F1.append(np.mean(f1s))
            Acc.append(np.mean(accs))
            Train_loss.append(np.mean(train_losses))
            Epochs.set_description('Loss: {:.4f}, F1: {:.2f}%, Acc: {:.2f}%'.format(Train_loss[-1],
                                                                                        F1[-1],
                                                                                        Acc[-1]))
            Epochs.refresh()
            weights = copy.deepcopy(length_predictor.state_dict())
            if Acc and Acc[-1] > best_score:
                best_score = Acc[-1]
                best_weights = weights
        length_predictor.load_state_dict(best_weights)
        if record_runtime:
            duration = time.time()-t0
            runtime_info = {"Architecture": length_predictor.name,
                            "Number of Epochs": self.epochs, "Runtime (s)": duration}
            if not os.path.exists(self.storage_path+"/runtime/"):
                os.mkdir(self.storage_path+"/runtime/")
            with open(self.storage_path+"/runtime/runtime"+"_"+desc+".json", "w") as file:
                json.dump(runtime_info, file, indent=3)
        results_dict = dict()
        print("Top performance: loss: {:.4f}, f1: {:.2f}% ... "
              "acc: {:.2f}%".format(min(Train_loss), max(F1), max(Acc)), "weights saved based on Acc best score!")
        print()
        results_dict.update({"Train Max F1": max(F1), "Train Acc": max(Acc),
                             "Train Min Loss": min(Train_loss)})
        if not os.path.exists(self.storage_path+"/results/"):
            os.mkdir(self.storage_path+"/results/")
        with open(self.storage_path+"/results/"+"results"+"_"+desc+".json", "w") as file:
            json.dump(results_dict, file, indent=3)
        if save_model:
            if not os.path.exists(self.storage_path+"/trained_models/"):
                os.mkdir(self.storage_path+"/trained_models/")
            torch.save(length_predictor.state_dict(), self.storage_path+"/trained_models/"+"trained_"+desc+".pt")
            print("{} saved".format(length_predictor.name))
        if not os.path.exists(self.storage_path+"/metrics/"):
            os.mkdir(self.storage_path+"/metrics/")
        with open(self.storage_path+"/metrics/"+"metrics_"+desc+".json", "w") as plot_file:
            json.dump({"f1": F1, "acc": Acc, "loss": Train_loss}, plot_file,
                      indent=3)
