"""NCES trainer."""
import numpy as np
import copy
import torch
from tqdm import trange
from collections import defaultdict
import os
import json
from ontolearn.data_struct import NCESBaseDataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pad_sequence
import time


def before_pad(arg):
    arg_temp = []
    for atm in arg:
        if atm == 'PAD':
            break
        arg_temp.append(atm)
    if len(set(arg_temp)) == 3 and ('⊓' in arg_temp or '⊔' in arg_temp):
        return arg_temp[0]
    return arg_temp


class NCESTrainer:
    """NCES trainer."""
    def __init__(self, nces, epochs=300, learning_rate=1e-4, decay_rate=0, clip_value=5.0, num_workers=8,
                 storage_path="./"):
        self.nces = nces
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.clip_value = clip_value
        self.num_workers = num_workers
        self.storage_path = storage_path

    @staticmethod
    def compute_accuracy(prediction, target):
        def soft(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = set(before_pad(NCESBaseDataLoader.decompose(arg1_)))
            else:
                arg1_ = set(before_pad(arg1_))
            if isinstance(arg2_, str):
                arg2_ = set(before_pad(NCESBaseDataLoader.decompose(arg2_)))
            else:
                arg2_ = set(before_pad(arg2_))
            return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))

        def hard(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = before_pad(NCESBaseDataLoader.decompose(arg1_))
            else:
                arg1_ = before_pad(arg1_)
            if isinstance(arg2_, str):
                arg2_ = before_pad(NCESBaseDataLoader.decompose(arg2_))
            else:
                arg2_ = before_pad(arg2_)
            return 100*float(sum(map(lambda x, y: x == y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
        soft_acc = sum(map(soft, prediction, target))/len(target)
        hard_acc = sum(map(hard, prediction, target))/len(target)
        return soft_acc, hard_acc

    def get_optimizer(self, synthesizer, optimizer='Adam'):
        if optimizer == 'Adam':
            return torch.optim.Adam(synthesizer.parameters(), lr=self.learning_rate)
        elif optimizer == 'SGD':
            return torch.optim.SGD(synthesizer.parameters(), lr=self.learning_rate)
        elif optimizer == 'RMSprop':
            return torch.optim.RMSprop(synthesizer.parameters(), lr=self.learning_rate)
        else:
            raise ValueError
            print('Unsupported optimizer')

    def show_num_learnable_params(self):
        print("*"*20+"Trainable model size"+"*"*20)
        size = sum([p.numel() for p in self.nces.model.parameters()])
        size_ = 0
        print("Synthesizer: ", size)
        print("*"*20+"Trainable model size"+"*"*20)
        print()
        return size

    def collate_batch(self, batch):
        pos_emb_list = []
        neg_emb_list = []
        target_labels = []
        for pos_emb, neg_emb, label in batch:
            if pos_emb.ndim != 2:
                pos_emb = pos_emb.reshape(1, -1)
            if neg_emb.ndim != 2:
                neg_emb = neg_emb.reshape(1, -1)
            pos_emb_list.append(pos_emb)
            neg_emb_list.append(neg_emb)
            target_labels.append(label)
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.nces.num_examples - pos_emb_list[0].shape[0]),
                                "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.nces.num_examples - neg_emb_list[0].shape[0]),
                                "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
        return pos_emb_list, neg_emb_list, target_labels

    def map_to_token(self, idx_array):
        return self.nces.model.inv_vocab[idx_array]

    def train(self, train_dataloader, save_model=True, optimizer='Adam', record_runtime=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.nces.model, list):
            self.nces.model = copy.deepcopy(self.nces.model[0])
        model_size = self.show_num_learnable_params()
        if device.type == "cpu":
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")
        print()
        print("#"*50)
        print()
        print("{} starts training... \n".format(self.nces.model.name))
        print("#"*50, "\n")
        synthesizer = copy.deepcopy(self.nces.model).train()
        desc = synthesizer.name
        if device.type == "cuda":
            synthesizer.cuda()
        opt = self.get_optimizer(synthesizer=synthesizer, optimizer=optimizer)
        if self.decay_rate:
            self.scheduler = ExponentialLR(opt, self.decay_rate)
        Train_loss = []
        Train_acc = defaultdict(list)
        best_score = 0.
        if record_runtime:
            t0 = time.time()
        s_acc, h_acc = 0, 0
        Epochs = trange(self.epochs, desc=f'Loss: {np.nan}, Soft Acc: {s_acc}, Hard Acc: {h_acc}', leave=True)
        for e in Epochs:
            soft_acc, hard_acc = [], []
            train_losses = []
            for x1, x2, labels in train_dataloader:
                target_sequence = self.map_to_token(labels)
                if device.type == "cuda":
                    x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
                pred_sequence, scores = synthesizer(x1, x2)
                loss = synthesizer.loss(scores, labels)
                s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                soft_acc.append(s_acc)
                hard_acc.append(h_acc)
                train_losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                clip_grad_value_(synthesizer.parameters(), clip_value=self.clip_value)
                opt.step()
                if self.decay_rate:
                    self.scheduler.step()
            train_soft_acc, train_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            Train_loss.append(np.mean(train_losses))
            Train_acc['soft'].append(train_soft_acc)
            Train_acc['hard'].append(train_hard_acc)
            Epochs.set_description('Loss: {:.4f}, Soft Acc: {:.2f}%, Hard Acc: {:.2f}%'.format(Train_loss[-1],
                                                                                               train_soft_acc,
                                                                                               train_hard_acc))
            Epochs.refresh()
            weights = copy.deepcopy(synthesizer.state_dict())
            if Train_acc['hard'] and Train_acc['hard'][-1] > best_score:
                best_score = Train_acc['hard'][-1]
                best_weights = weights
        synthesizer.load_state_dict(best_weights)
        if record_runtime:
            duration = time.time()-t0
            runtime_info = {"Architecture": synthesizer.name,
                            "Number of Epochs": self.epochs, "Runtime (s)": duration}
            if not os.path.exists(self.storage_path+"/runtime/"):
                os.mkdir(self.storage_path+"/runtime/")
            with open(self.storage_path+"/runtime/runtime"+"_"+desc+".json", "w") as file:
                json.dump(runtime_info, file, indent=3)
        results_dict = dict()
        print("Top performance: loss: {:.4f}, soft accuracy: {:.2f}% ... "
              "hard accuracy: {:.2f}%".format(min(Train_loss), max(Train_acc['soft']), max(Train_acc['hard'])))
        print()
        results_dict.update({"Train Max Soft Acc": max(Train_acc['soft']), "Train Max Hard Acc": max(Train_acc['hard']),
                             "Train Min Loss": min(Train_loss)})
        if not os.path.exists(self.storage_path+"/results/"):
            os.mkdir(self.storage_path+"/results/")
        with open(self.storage_path+"/results/"+"results"+"_"+desc+".json", "w") as file:
            json.dump(results_dict, file, indent=3)
        if save_model:
            if not os.path.exists(self.storage_path+"/trained_models/"):
                os.mkdir(self.storage_path+"/trained_models/")
            torch.save(synthesizer.state_dict(), self.storage_path+"/trained_models/"+"trained_"+desc+".pt")
            print("{} saved".format(synthesizer.name))
        if not os.path.exists(self.storage_path+"/metrics/"):
            os.mkdir(self.storage_path+"/metrics/")
        with open(self.storage_path+"/metrics/"+"metrics_"+desc+".json", "w") as plot_file:
            json.dump({"soft acc": Train_acc['soft'], "hard acc": Train_acc['hard'], "loss": Train_loss}, plot_file,
                      indent=3)
