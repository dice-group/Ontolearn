from typing import Tuple
import numpy as np
from sklearn import preprocessing
from typing import List
import random
import torch
from collections import deque


# @Todo CD:Could we combine PrepareBatchOfPrediction and PrepareBatchOfTraining?

class PrepareBatchOfPrediction(torch.utils.data.Dataset):

    def __init__(self, current_state: torch.FloatTensor, next_state_batch: torch.Tensor, p: torch.FloatTensor,
                 n: torch.FloatTensor):
        """
        @param current_state: a Tensor of torch.Size([1, 1, dim]) corresponds to embeddings of current_state
        @param next_state_batch: a Tensor of torch.Size([n, 1, dim]) corresponds to embeddings of next_states, i.e. \rho(current_state)
        @param p:
        @param n:
        """
        self.S_Prime = next_state_batch

        # Expands them into torch.Size([n, 1, dim])
        self.S = current_state.expand(self.S_Prime.shape)
        self.Positives = p.expand(self.S_Prime.shape)
        self.Negatives = n.expand(self.S_Prime.shape)
        assert self.S.shape == self.S_Prime.shape == self.Positives.shape == self.Negatives.shape
        assert self.S.dtype == self.S_Prime.dtype == self.Positives.dtype == self.Negatives.dtype == torch.float32

        self.X = torch.cat([self.S, self.S_Prime, self.Positives, self.Negatives], 1)
        n, depth, dim = self.X.shape
        self.X = self.X.view(n, depth, 1, dim)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def get_all(self):
        return self.X


class PrepareBatchOfTraining(torch.utils.data.Dataset):

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
        self.X = self.X.view(num_points, depth, 1, dim)
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


class Experience:
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
        @param e: a tuple of s_i, s_j and reward, where s_i and s_j represent refining s_i and reaching s_j.
        @return:
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
