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

"""Data structures."""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from collections import deque
import pandas as pd
import numpy as np
import random


class PrepareBatchOfPrediction(torch.utils.data.Dataset):  # pragma: no cover

    def __init__(self, current_state: torch.FloatTensor, next_state_batch: torch.FloatTensor, p: torch.FloatTensor,
                 n: torch.FloatTensor):
        assert len(p) > 0 and len(n) > 0
        num_next_states = len(next_state_batch)
        current_state = current_state.repeat(num_next_states, 1, 1)
        p = p.repeat((num_next_states, 1, 1))
        n = n.repeat((num_next_states, 1, 1))
        # batch, 4, dim
        self.X = torch.cat([current_state, next_state_batch, p, n], 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def get_all(self):
        return self.X


class PrepareBatchOfTraining(torch.utils.data.Dataset):  # pragma: no cover

    def __init__(self, current_state_batch: torch.Tensor, next_state_batch: torch.Tensor, p: torch.Tensor,
                 n: torch.Tensor, q: torch.Tensor):
        # Sanity checking
        if torch.isnan(current_state_batch).any() or torch.isinf(current_state_batch).any():
            raise ValueError('invalid value detected in current_state_batch,\n{0}'.format(current_state_batch))
        if torch.isnan(next_state_batch).any() or torch.isinf(next_state_batch).any():
            raise ValueError('invalid value detected in next_state_batch,\n{0}'.format(next_state_batch))
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise ValueError('invalid value detected in p,\n{0}'.format(p))
        if torch.isnan(n).any() or torch.isinf(n).any():
            raise ValueError('invalid value detected in p,\n{0}'.format(n))
        if torch.isnan(q).any() or torch.isinf(q).any():
            raise ValueError('invalid Q value  detected during batching.')

        self.S = current_state_batch
        self.S_Prime = next_state_batch
        self.y = q.view(len(q), 1)
        assert self.S.shape == self.S_Prime.shape
        assert len(self.y) == len(self.S)
        try:
            self.Positives = p.expand(next_state_batch.shape)
        except RuntimeError as e:
            print(p.shape)
            print(next_state_batch.shape)
            print(e)
            raise
        self.Negatives = n.expand(next_state_batch.shape)

        assert self.S.shape == self.S_Prime.shape == self.Positives.shape == self.Negatives.shape
        assert self.S.dtype == self.S_Prime.dtype == self.Positives.dtype == self.Negatives.dtype == torch.float32
        self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1)

        num_points, depth, dim = self.X.shape
        # self.X = self.X.view(num_points, depth, 1, dim)
        # X[0] => corresponds to a data point, X[0] \in R^{4 \times 1 \times dim}
        # where X[0][0] => current state representation R^{1 \times dim}
        # where X[0][1] => next state representation R^{1 \times dim}
        # where X[0][2] => positive example representation R^{1 \times dim}
        # where X[0][3] => negative example representation R^{1 \times dim}

        if torch.isnan(self.X).any() or torch.isinf(self.X).any():
            print('invalid input detected during batching in X')
            raise ValueError
        if torch.isnan(self.y).any() or torch.isinf(self.y).any():
            print('invalid Q value  detected during batching in Y')
            raise ValueError

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Experience:  # pragma: no cover
    """
    A class to model experiences for Replay Memory.
    """

    def __init__(self, maxlen: int):
        # @TODO we may want to not forget experiences yielding high rewards
        self.current_states = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)

    def __len__(self):
        assert len(self.current_states) == len(self.next_states) == len(self.rewards)
        return len(self.current_states)

    def append(self, e):
        """
        Append.
        Args:
            e: A tuple of s_i, s_j and reward, where s_i and s_j represent refining s_i and reaching s_j.

        """
        assert len(self.current_states) == len(self.next_states) == len(self.rewards)
        s_i, s_j, r = e
        assert s_i.embeddings.shape == s_j.embeddings.shape
        self.current_states.append(s_i.embeddings)
        self.next_states.append(s_j.embeddings)
        self.rewards.append(r)

    def retrieve(self):
        return list(self.current_states), list(self.next_states), list(self.rewards)

    def clear(self):
        self.current_states.clear()
        self.next_states.clear()
        self.rewards.clear()


class NCESBaseDataset:  # pragma: no cover

    def __init__(self, vocab, inv_vocab):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.vocab_df = pd.DataFrame(self.vocab.values(), index=self.vocab.keys())

    @staticmethod
    def decompose(concept_name: str) -> list:
        """ Decomposes a class expression into a sequence of tokens (atoms) """
        def is_number(char):
            """ Checks if a character can be converted into a number """
            try:
                int(char)
                return True
            except:
                return False
        specials = ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', ' ', '(', ')',\
                    '⁻', '≤', '≥', '{', '}', ':', '[', ']']
        list_ordered_pieces = []
        i = 0
        while i < len(concept_name):
            concept = ''
            while i < len(concept_name) and not concept_name[i] in specials:
                if concept_name[i] == '.' and not is_number(concept_name[i-1]):
                    break
                concept += concept_name[i]
                i += 1
            if concept and i < len(concept_name):
                list_ordered_pieces.extend([concept, concept_name[i]])
            elif concept:
                list_ordered_pieces.append(concept)
            elif i < len(concept_name):
                list_ordered_pieces.append(concept_name[i])
            i += 1
            
        return list_ordered_pieces

    def get_labels(self, target):
        target = self.decompose(target)
        labels = [self.vocab[atm] for atm in target]
        
        return labels, len(target)


class NCESDataset(NCESBaseDataset torch.utils.data.Dataset):  # pragma: no cover

    def __init__(self, data: list, embeddings, vocab, inv_vocab, shuffle_examples, max_length, example_sizes=None,
                 sorted_examples=True):
        super().__init__(vocab, inv_vocab)
        self.data = data
        self.embeddings = embeddings
        self.max_length = max_length
        self.shuffle_examples = shuffle_examples
        self.example_sizes = example_sizes
        self.sorted_examples = sorted_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, value = self.data[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        if self.example_sizes is not None:
            k_pos, k_neg = random.choice(self.example_sizes)
            k_pos = min(k_pos, len(pos))
            k_neg = min(k_neg, len(neg))
            selected_pos = random.sample(pos, k_pos)
            selected_neg = random.sample(neg, k_neg)
        else:
            selected_pos = pos
            selected_neg = neg
            
        datapoint_pos = torch.FloatTensor(self.embeddings.loc[selected_pos].values.squeeze())
        datapoint_neg = torch.FloatTensor(self.embeddings.loc[selected_neg].values.squeeze())
        labels, length = self.get_labels(key)
        
        return datapoint_pos, datapoint_neg, torch.cat([torch.tensor(labels),
                                                        self.vocab['PAD'] * torch.ones(
                                                            max(0, self.max_length-length))]).long()


class NCESDatasetInference(NCESBaseDataset, torch.utils.data.Dataset):  # pragma: no cover

    def __init__(self, data: list, embeddings, vocab, inv_vocab, shuffle_examples, sorted_examples=True):
        super().__init__(vocab, inv_vocab)
        self.data = data
        self.embeddings = embeddings
        self.shuffle_examples = shuffle_examples
        self.sorted_examples = sorted_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, pos, neg = self.data[idx]
        if self.sorted_examples:
            pos, neg = sorted(pos), sorted(neg)
        elif self.shuffle_examples:
            random.shuffle(pos)
            random.shuffle(neg)
            
        datapoint_pos = torch.FloatTensor(self.embeddings.loc[pos].values.squeeze())
        datapoint_neg = torch.FloatTensor(self.embeddings.loc[neg].values.squeeze())
        
        return datapoint_pos, datapoint_neg

    
class CLIPDataset(torch.utils.data.Dataset):  # pragma: no cover

    def __init__(self, data: list, embeddings, shuffle_examples, example_sizes: list=None,
                 k=5, sorted_examples=True):
        super().__init__()
        self.data = data
        self.embeddings = embeddings
        self.shuffle_examples = shuffle_examples
        self.example_sizes = example_sizes
        self.k = k
        self.sorted_examples = sorted_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, value = self.data[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        length = value['length']
        if self.example_sizes is not None:
            k_pos, k_neg = random.choice(self.example_sizes)
            k_pos = min(k_pos, len(pos))
            k_neg = min(k_neg, len(neg))
            selected_pos = random.sample(pos, k_pos)
            selected_neg = random.sample(neg, k_neg)
        elif self.k is not None:
            prob_pos_set = 1.0/(1+np.array(range(min(self.k, len(pos)), len(pos)+1, self.k)))
            prob_pos_set = prob_pos_set/prob_pos_set.sum()
            prob_neg_set = 1.0/(1+np.array(range(min(self.k, len(neg)), len(neg)+1, self.k)))
            prob_neg_set = prob_neg_set/prob_neg_set.sum()
            k_pos = np.random.choice(range(min(self.k, len(pos)), len(pos)+1, self.k), replace=False, p=prob_pos_set)
            k_neg = np.random.choice(range(min(self.k, len(neg)), len(neg)+1, self.k), replace=False, p=prob_neg_set)
            selected_pos = random.sample(pos, k_pos)
            selected_neg = random.sample(neg, k_neg)
        else:
            selected_pos = pos
            selected_neg = neg
        if self.shuffle_examples:
            random.shuffle(selected_pos)
            random.shuffle(selected_neg)
            
        datapoint_pos = torch.FloatTensor(self.embeddings.loc[selected_pos].values.squeeze())
        datapoint_neg = torch.FloatTensor(self.embeddings.loc[selected_neg].values.squeeze())
        
        return datapoint_pos, datapoint_neg, torch.LongTensor([length])
    
    
class CLIPDatasetInference(torch.utils.data.Dataset):  # pragma: no cover

    def __init__(self, data: list, embeddings, shuffle_examples,
                 sorted_examples=True):
        super().__init__()
        self.data = data
        self.embeddings = embeddings
        self.shuffle_examples = shuffle_examples
        self.sorted_examples = sorted_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, pos, neg = self.data[idx]
        if self.sorted_examples:
            pos, neg = sorted(pos), sorted(neg)
        elif self.shuffle_examples:
            random.shuffle(pos)
            random.shuffle(neg)
            
        datapoint_pos = torch.FloatTensor(self.embeddings.loc[pos].values.squeeze())
        datapoint_neg = torch.FloatTensor(self.embeddings.loc[pos].values.squeeze())
        
        return datapoint_pos, datapoint_neg
    
    

class ROCESDataset(BaseDataset, torch.utils.data.Dataset):
    
    def __init__(self, data, triples_data, vocab, inv_vocab, sampling_strategy="p"):
        super(ROCESDataset, self).__init__(vocab, inv_vocab)
        self.data = data
        self.triples_data = triples_data
        self.k = k
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.sampling_strategy = sampling_strategy
        
    def load_embeddings(self, embedding_model):
        embeddings, _ = embedding_model.get_embeddings()
        self.embeddings = embeddings.cpu()
        

    def set_k(self, k):
        self.k = k
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key, value = self.data[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        if self.sampling_strategy != 'uniform':
            prob_pos_set = 1.0/(1+np.array(range(min(self.k, len(pos)), len(pos)+1, self.k)))
            prob_pos_set = prob_pos_set/prob_pos_set.sum()
            prob_neg_set = 1.0/(1+np.array(range(min(self.k, len(neg)), len(neg)+1, self.k)))
            prob_neg_set = prob_neg_set/prob_neg_set.sum()
            k_pos = np.random.choice(range(min(self.k, len(pos)), len(pos)+1, self.k), replace=False, p=prob_pos_set)
            k_neg = np.random.choice(range(min(self.k, len(neg)), len(neg)+1, self.k), replace=False, p=prob_neg_set)
        else:
            k_pos = np.random.choice(range(min(self.k, len(pos)), len(pos)+1, self.k), replace=False)
            k_neg = np.random.choice(range(min(self.k, len(neg)), len(neg)+1, self.k), replace=False)
        selected_pos = random.sample(pos, k_pos)
        selected_neg = random.sample(neg, k_neg)
        
        datapoint_pos = self.embeddings[self.triples_data.entity2idx.loc[selected_pos].values.squeeze()]
        datapoint_neg = self.embeddings[self.triples_data.entity2idx.loc[selected_neg].values.squeeze()]
        labels, length = self.get_labels(key)
        
        return datapoint_pos, datapoint_neg, torch.cat([torch.tensor(labels), self.vocab['PAD']*torch.ones(max(0,self.max_length-length))]).long()
    
    
class ROCESDatasetInference(BaseDataset, torch.utils.data.Dataset):
    
    def __init__(self, data, triples_data, k, vocab, inv_vocab, sampling_strategy, num_examples, num_pred_per_lp=1, random_sample=False):
        super(ROCESDatasetInference, self).__init__(vocab, inv_vocab)
        self.data = data
        self.triples_data = triples_data
        self.k = k
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.sampling_strategy = sampling_strategy
        self.num_examples = num_examples
        self.num_pred_per_lp = num_pred_per_lp
        
    def load_embeddings(self, embedding_model):
        embeddings, _ = embedding_model.get_embeddings()
        self.embeddings = embeddings.cpu()
        
    def set_k(self, k):
        self.k = k

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _, value = self.data[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        
        if self.sampling_strategy != 'uniform':
            prob_pos_set = 1.0/(1+np.array(range(min(self.k, len(pos)), len(pos)+1, self.k)))
            prob_pos_set = prob_pos_set/prob_pos_set.sum()
            prob_neg_set = 1.0/(1+np.array(range(min(self.k, len(neg)), len(neg)+1, self.k)))
            prob_neg_set = prob_neg_set/prob_neg_set.sum()
            k_pos = np.random.choice(range(min(self.k, len(pos)), len(pos)+1, self.k),
                                     size=(self.num_predictions,),
                                     replace=True, p=prob_pos_set)
            k_neg = np.random.choice(range(min(self.k, len(neg)), len(neg)+1, self.k),
                                     size=(self.num_predictions,),
                                     replace=True, p=prob_neg_set)
        else:
            k_pos = np.random.choice(range(min(self.k, len(pos)), len(pos)+1, self.k), 
                                     size=(self.num_predictions,), replace=True)
            k_neg = np.random.choice(range(min(self.k, len(neg)), len(neg)+1, self.k),
                                     size=(self.num_predictions,), replace=True)
            
        selected_pos = [random.sample(pos, k) for k in k_pos]
        selected_neg = [random.sample(neg, k) for k in k_neg]
        
        pos_emb_list = [self.embeddings[self.triples_data.entity2idx.loc[pos_ex].values.squeeze()] for pos_ex in selected_pos]
        neg_emb_list = [self.embeddings[self.triples_data.entity2idx.loc[neg_ex].values.squeeze()] for neg_ex in selected_neg]
        
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.num_examples - pos_emb_list[0].shape[0]), "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0).squeeze()
        
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.num_examples - neg_emb_list[0].shape[0]), "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0).squeeze()
        
        return pos_emb_list, neg_emb_list
        
        
class HeadAndRelationBatchLoader(torch.utils.data.Dataset):
    
    def __init__(self, er_vocab, num_e):
        self.num_e = num_e
        head_rel_idx = torch.Tensor(list(er_vocab.keys())).long()
        self.head_idx = head_rel_idx[:, 0]
        self.rel_idx = head_rel_idx[:, 1]
        self.tail_idx = list(er_vocab.values())
        assert len(self.head_idx) == len(self.rel_idx) == len(self.tail_idx)

    def __len__(self):
        return len(self.tail_idx)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.num_e)
        y_vec[self.tail_idx[idx]] = 1  # given head and rel, set 1's for all tails.
        return self.head_idx[idx], self.rel_idx[idx], y_vec