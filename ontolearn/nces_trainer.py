# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""Trainer for NCES instances"""
import numpy as np
import copy
import torch
from tqdm import trange
from collections import defaultdict
import os
import json
from ontolearn.data_struct import NCESBaseDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pad_sequence
import time
from collections import defaultdict


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
    """Trainer for neural class expression synthesizers, i.e., NCES, NCES2, ROCES."""
    def __init__(self, synthesizer, epochs=300, batch_size=128, learning_rate=1e-4, decay_rate=0,
                 clip_value=5.0, num_workers=8, nces2_or_roces=False, storage_path="./"):
        self.synthesizer = synthesizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.clip_value = clip_value
        self.num_workers = num_workers
        self.nces2_or_roces = nces2_or_roces
        self.storage_path = storage_path

    @staticmethod
    def compute_accuracy(prediction, target):
        def soft(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = set(before_pad(NCESBaseDataset.decompose(arg1_)))
            else:
                arg1_ = set(before_pad(arg1_))
            if isinstance(arg2_, str):
                arg2_ = set(before_pad(NCESBaseDataset.decompose(arg2_)))
            else:
                arg2_ = set(before_pad(arg2_))
            return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))

        def hard(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = before_pad(NCESBaseDataset.decompose(arg1_))
            else:
                arg1_ = before_pad(arg1_)
            if isinstance(arg2_, str):
                arg2_ = before_pad(NCESBaseDataset.decompose(arg2_))
            else:
                arg2_ = before_pad(arg2_)
            return 100*float(sum(map(lambda x, y: x == y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
        soft_acc = sum(map(soft, prediction, target))/len(target)
        hard_acc = sum(map(hard, prediction, target))/len(target)
        return soft_acc, hard_acc

    def get_optimizer(self, synthesizer, optimizer='Adam'):  # pragma: no cover
        if optimizer == 'Adam':
            return torch.optim.Adam(synthesizer.parameters(), lr=self.learning_rate)
        elif optimizer == 'SGD':
            return torch.optim.SGD(synthesizer.parameters(), lr=self.learning_rate)
        elif optimizer == 'RMSprop':
            return torch.optim.RMSprop(synthesizer.parameters(), lr=self.learning_rate)
        else:
            raise ValueError
            print('Unsupported optimizer')
            
    def get_data_idxs(self):
        data_idxs = [(self.synthesizer.triples_data.entity2idx.loc[t[0]].values[0],
                      self.synthesizer.triples_data.relation2idx.loc[t[1]].values[0],
                      self.synthesizer.triples_data.entity2idx.loc[t[2]].values[0]) for t in self.synthesizer.triples_data.triples]
        return data_idxs
    
    def get_er_vocab(self):
        er_vocab = defaultdict(list)
        data_idxs = self.get_data_idxs()
        for triple in data_idxs:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
    
    

    @staticmethod
    def show_num_trainable_params(synthesizer):
        size_emb_model = 0 # If training NCES there is no embedding model to train
        size_model = sum([p.numel() for p in synthesizer["model"].parameters()])
        if synthesizer["emb_model"]:
            size_emb_model = sum([p.numel() for p in synthesizer["emb_model"].parameters()])
        print("#"*30+"Trainable model size"+"#"*30)
        print("Synthesizer: ", size_model)
        print("Embedding model: ", size_emb_model)
        print("#"*30+"Trainable model size"+"#"*30)

    def collate_batch(self, batch):  # pragma: no cover
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
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.synthesizer.num_examples - pos_emb_list[0].shape[0]),
                                "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.synthesizer.num_examples - neg_emb_list[0].shape[0]),
                                "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
        return pos_emb_list, neg_emb_list, target_labels

    def map_to_token(self, idx_array):
        return self.synthesizer.inv_vocab[idx_array]
    
    
    def train_step(self, batch, model, optimizer, emb_model=None, triples_dataloader=None):
        soft_acc, hard_acc = [], []
        train_losses = []
        if emb_model:
            try:
                triples_batch = next(triples_dataloader)
            except:
                triples_dataloader = iter(DataLoader(TriplesDataset(er_vocab=self.er_vocab, num_e=len(self.synthesizer.triples_data.entities)),
                                              batch_size=2*self.batch_size, num_workers=self.num_workers, shuffle=True))
                triples_batch = next(triples_dataloader)
                
        x_pos, x_neg, labels = batch
        target_sequence = self.map_to_token(labels)
        if device.type == "cuda":
            x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
        pred_sequence, scores = model(x1, x2)
        loss = model.loss(scores, labels)
        # Forward triples to embedding model
        if emb_model:
            loss_ = model.loss(scores, labels)
            loss = loss + loss_
        s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
        opt.zero_grad()
        loss.backward()
        clip_grad_value_(synthesizer.parameters(), clip_value=self.clip_value)
        opt.step()
        if self.decay_rate:
            self.scheduler.step()
        return loss.item(), s_acc, h_acc
        

    def train(self, data, shuffle_examples=False, example_sizes=None,
              save_model=True, optimizer='Adam', record_runtime=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name in self.synthesizer.model:
            self.show_num_trainable_params(self.synthesizer.model[model_name])
            if device.type == "cpu":
                print("Training on CPU, it may take long...")
            else:
                print("GPU available !")
            print()
            print("#"*50)
            print()
            model = copy.deepcopy(self.synthesizer.model[model_name])
            print("{} starts training... \n".format(model["model"].name))
            print("#"*50, "\n")
            desc = model["model"].name
            if device.type == "cuda":
                model["model"].cuda()
                if model["emb_model"]:
                    model["emb_model"].cuda()
            optimizer = self.get_optimizer(model=model, optimizer=optimizer)
            if self.decay_rate:
                self.scheduler = ExponentialLR(opt, self.decay_rate)
            if model["emb_model"]:
                self.er_vocab = self.get_er_vocab()
                triples_dataloader = iter(DataLoader(TriplesDataset(er_vocab=self.er_vocab, num_e=len(self.synthesizer.triples_data.entities)),
                                          batch_size=2*self.batch_size, num_workers=self.num_workers, shuffle=True))
            else:
                assert hasattr(self.synthesizer, "instance_embeddings"), "If no embedding model is available, `instance_embeddings` must be an attribute of the synthesizer since you are probably training NCES"
                train_dataset = DataLoader(NCESDataset(data, self.synthesizer.instance_embeddings, self.synthesizer.vocab, self.synthesizer.inv_vocab,
                                                       shuffle_examples=shuffle_examples, max_length=self.synthesizer.max_length, example_sizes=example_sizes),
                                                       batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=True)
            Train_loss = []
            Train_acc = defaultdict(list)
            best_score = 0
            if record_runtime:
                t0 = time.time()
            s_acc, h_acc = 0, 0
            Epochs = trange(self.epochs, desc=f'Loss: {np.nan}, Soft Acc: {s_acc}, Hard Acc: {h_acc}', leave=True)
            for e in Epochs:
                soft_acc, hard_acc = [], []
                train_losses = []
                num_batches = len(data) // self.batch_size if len(data) % self.batch_size == 0 else len(data) // self.batch_size + 1
                batch_data = trange(len(data), desc=f'Train: <Batch: {batch_count}/{num_batches}, Loss: {np.nan}, Soft Acc: {s_acc}, Hard Acc: {h_acc}>', leave=False)
                batch_count = 0
                if model["emb_model"]:
                    # When there is no embedding_model, then we are training NCES2 or ROCES and need to use slicing and shuffling to construct input batches since we need to repeatedly query the embedding model for the updated embeddings
                    random.shuffle(data)
                    train_dataset = ROCESDataset(data, self.synthesizer.triples_data, self.synthesizer.vocab, self.synthesizer.inv_vocab,
                                             sampling_strategy=self.synthesizer.sampling_strategy, max_length=self.synthesizer.max_length)
                    for _, train_idx in zip(batch_data, range(0, len(data), self.batch_size)):
                        batch = train_dataset[train_idx:train_idx+self.batch_size]
                        loss, s_acc, h_acc = self.train_step(batch, model["model"], optimizer, model["emb_model"], triples_dataloader)
                        batch_count += 1
                        batch_data.set_description('Train: <Batch: {}/{}, Loss: {:.4f}, Soft Acc: {:.2f}, Hard Acc: {:.2f}>'.format(batch_count, num_batches, loss, s_acc, h_acc))
                        batch_data.refresh()
                        soft_acc.append(s_acc)
                        hard_acc.append(h_acc)
                        train_losses.append(loss)
                else:
                    # When an embedding model is None, then we are training NCES and the training data is a torch.utils.data.DataLoader object
                    for _, batch in zip(batch_data, train_dataset):
                        loss, s_acc, h_acc = self.train_step(batch, model["model"], optimizer, model["emb_model"], triples_dataloader)
                        batch_count += 1
                        batch_data.set_description('Train: <Batch: {}/{}, Loss: {:.4f}, Soft Acc: {:.2f}, Hard Acc: {:.2f}>'.format(batch_count, num_batches, loss, s_acc, h_acc))
                        batch_data.refresh()
                        soft_acc.append(s_acc)
                        hard_acc.append(h_acc)
                        train_losses.append(loss)
                        
                train_soft_acc, train_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
                Train_loss.append(np.mean(train_losses))
                Train_acc['soft'].append(train_soft_acc)
                Train_acc['hard'].append(train_hard_acc)
                Epochs.set_description('<Epoch: {}/{}> Loss: {:.4f}, Soft Acc: {:.2f}%, Hard Acc: {:.2f}%'.format(e, self.epochs, Train_loss[-1], train_soft_acc, train_hard_acc))
                Epochs.refresh()
                #### Continue here
                weights = copy.deepcopy(synthesizer.state_dict())
                if Train_acc['hard'] and Train_acc['hard'][-1] > best_score:
                    best_score = Train_acc['hard'][-1]
                    best_weights = weights
            synthesizer.load_state_dict(best_weights)
            if record_runtime:  # pragma: no cover
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
            
            if save_model:  # pragma: no cover
                if not os.path.exists(self.storage_path+"/results/"):
                    os.mkdir(self.storage_path+"/results/")
                with open(self.storage_path+"/results/"+"results"+"_"+desc+".json", "w") as file:
                    json.dump(results_dict, file, indent=3)

                if not os.path.exists(self.storage_path+"/trained_models/"):
                    os.mkdir(self.storage_path+"/trained_models/")
                torch.save(synthesizer.state_dict(), self.storage_path+"/trained_models/"+"trained_"+desc+".pt")
                print("{} saved".format(synthesizer.name))
                if not os.path.exists(self.storage_path+"/metrics/"):
                    os.mkdir(self.storage_path+"/metrics/")
                with open(self.storage_path+"/metrics/"+"metrics_"+desc+".json", "w") as plot_file:
                    json.dump({"soft acc": Train_acc['soft'], "hard acc": Train_acc['hard'], "loss": Train_loss}, plot_file,
                              indent=3)
