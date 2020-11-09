from typing import Tuple
import numpy as np
from sklearn import preprocessing
from typing import List
import random
import torch
#from .metrics import F1
#from .util import create_experiment_folder, create_logger

class PriorityQueue:
    def __init__(self):
        from queue import PriorityQueue
        self.struct = PriorityQueue()
        self.items_in_queue = set()
        self.expressionTests = 0

    def __len__(self):
        return len(self.items_in_queue)

    def add_into_queue(self, t: Tuple):
        assert not isinstance(t[0], str)
        if not (t[1].ce.expression in self.items_in_queue):
            self.struct.put(t)
            self.items_in_queue.add(t[1].ce.expression)
            self.expressionTests += 1

    def get_from_queue(self):
        assert len(self.items_in_queue) > 0
        node_to_return = self.struct.get()[1]
        self.items_in_queue.remove(node_to_return.ce.expression)
        return node_to_return


class PropertyHierarchy:

    def __init__(self, onto):
        self.all_properties = [i for i in onto.properties()]

        self.data_properties = [i for i in onto.data_properties()]

        self.object_properties = [i for i in onto.object_properties()]

    def get_most_general_property(self):
        for i in self.all_properties:
            yield i

#@Todo CD:Could we combine PrepareBatchOfPrediction and PrepareBatchOfTraining?

class PrepareBatchOfPrediction(torch.utils.data.Dataset):

    def __init__(self, current_state: torch.FloatTensor, next_state_batch: torch.Tensor, p: torch.FloatTensor,
                 n: torch.FloatTensor):
        self.S_Prime = next_state_batch
        self.S = current_state.expand(self.S_Prime.shape)
        self.Positives = p.expand(next_state_batch.shape)
        self.Negatives = n.expand(next_state_batch.shape)
        assert self.S.shape == self.S_Prime.shape == self.Positives.shape == self.Negatives.shape
        assert self.S.dtype == self.S_Prime.dtype == self.Positives.dtype == self.Negatives.dtype == torch.float32
        # X.shape()=> batch_size,4, embedding dim)
        self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1)
        num_points, depth, dim = self.X.shape
        self.X = self.X.view(num_points, depth, dim)
        # self.X = self.X.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def get_all(self):
        return self.X

class PrepareBatchOfTraining(torch.utils.data.Dataset):

    def __init__(self, current_state_batch: torch.Tensor, next_state_batch: torch.Tensor, p: torch.Tensor,
                 n: torch.Tensor, q: torch.Tensor):
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

        self.Positives = p.expand(next_state_batch.shape)
        self.Negatives = n.expand(next_state_batch.shape)
        assert self.S.shape == self.S_Prime.shape == self.Positives.shape == self.Negatives.shape
        assert self.S.dtype == self.S_Prime.dtype == self.Positives.dtype == self.Negatives.dtype == torch.float32
        # X.shape()=> batch_size,4,embeddingdim)
        self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1)
        num_points, depth, dim = self.X.shape
        self.X = self.X.view(num_points, depth, dim)

        if torch.isnan(self.X).any() or torch.isinf(self.X).any():
            print('invalid input detected during batching in X')
            raise ValueError
        if torch.isnan(self.y).any() or torch.isinf(self.y).any():
            print('invalid Q value  detected during batching in Y')
            raise ValueError
        # self.X, self.y = self.X.to(device), self.y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]