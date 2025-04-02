import torch
import torch.nn as nn
from .utils import MAB
class PMAnet(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMAnet, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        self.linear = nn.Sequential(nn.Linear(dim*2, dim*4), nn.ReLU(), nn.Linear(dim*4, dim*2), nn.GELU(), nn.Linear(dim*2, 1))
        self.activation = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def encode(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

    def forward(self, X, individual, label=None):
        """
        X.shape = (batch_size, num_examples, dim)
        individual.shape = (batch_size, num_individuals, dim)
        label.shape = (batch_size, num_individuals)
        """
        enc = self.encode(X)
        # reppeat enc to match the size of individual
        enc = enc.repeat(1, individual.shape[1], 1)
        enc = torch.cat([enc, individual], dim=2)
        out = self.activation(self.linear(enc)).squeeze()
        if label is not None:
            if out.ndim < 2:
                out = out.unsqueeze(0)
            loss = self.loss(out, label)
            return out, loss
        else:
            return out
