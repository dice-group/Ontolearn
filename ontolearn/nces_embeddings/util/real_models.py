import torch
import numpy as np
from torch.nn.init import xavier_normal_
import torch.nn as nn
from numpy.random import RandomState

torch.backends.cudnn.deterministic = True
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


class Distmult(torch.nn.Module):
    def __init__(self, param):
        super(Distmult, self).__init__()
        self.name = 'Distmult'
        self.param = param
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = self.param['num_entities']
        self.num_relations = self.param['num_relations']
        self.loss = torch.nn.BCELoss()
        # Real embeddings of entities
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        # Real embeddings of relations.
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        # Dropouts for quaternion embeddings of ALL entities.
        self.input_dp_ent_real = torch.nn.Dropout(self.param['input_dropout'])
        # Dropouts for quaternion embeddings of relations.
        self.input_dp_rel_real = torch.nn.Dropout(self.param['input_dropout'])
        # Batch normalization for quaternion embeddings of ALL entities.
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        # Batch normalization for quaternion embeddings of relations.
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim)

    def forward_head_batch(self, *, e1_idx, rel_idx):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        # (1.2) Real embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        real_score = torch.mm(emb_head_real * self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                              self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)).transpose(1, 0))
        score = real_score
        return torch.sigmoid(score)

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def init(self):
        xavier_normal_(self.emb_ent_real.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)

    def get_embeddings(self):
        return self.emb_ent_real.weight.data, self.emb_rel_real.weight.data

    def forward_triples(self, *, e1_idx, rel_idx, e2_idx):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head = self.emb_ent_real(e1_idx)
        # (1.2) Real embeddings of relations
        emb_rel = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        # (1.3) Real embeddings of tail entities
        emb_tail = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e2_idx)))
        # Compute multi-linear product embeddings
        return torch.sigmoid((emb_head * emb_rel * emb_tail).sum(dim=1))

    def forward_triples_and_loss(self, e1_idx, rel_idx, e2_idx, targets):
        scores = self.forward_triples(e1_idx=e1_idx, rel_idx=rel_idx, e2_idx=e2_idx)
        return self.loss(scores, targets)


class TransE(torch.nn.Module):
    """
    TransE trained with binary cross entropy
    """

    def __init__(self, param):
        super(TransE, self).__init__()
        self.name = 'TransE'
        self.param = param
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = self.param['num_entities']
        self.num_relations = self.param['num_relations']
        # Real embeddings of entities
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        # Real embeddings of relations.
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real

        self.gamma = nn.Parameter(
            torch.Tensor([self.param['gamma']]),
            requires_grad=False
        )

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_ent_real.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)

    def get_embeddings(self):
        return self.emb_ent_real.weight.data, self.emb_rel_real.weight.data

    def forward_triples(self, *, e1_idx, rel_idx, e2_idx):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head = self.emb_ent_real(e1_idx)
        # (1.2) Real embeddings of relations
        emb_rel = self.emb_rel_real(rel_idx)
        # (1.3) Real embeddings of tail entities
        emb_tail = self.emb_ent_real(e2_idx)
        distance = torch.norm((emb_head + emb_rel) - emb_tail, p=1, dim=1)
        score = self.gamma.item() - distance
        # If distance is very small , then score is very high, i.e. 1.0
        # If distance is very large, then score is very small, i.e. 0.0
        return torch.sigmoid(score)

    def forward_triples_and_loss(self, e1_idx, rel_idx, e2_idx, target):
        score = self.forward_triples(e1_idx=e1_idx, rel_idx=rel_idx, e2_idx=e2_idx)
        return self.loss(score, target)

    def forward_head_and_loss(self, *args, **kwargs):
        raise NotImplementedError('KvsAll is not implemented for TransE')

class Tucker(torch.nn.Module):
    def __init__(self, param):
        super(Tucker, self).__init__()
        self.name = 'Tucker'
        self.param = param
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = self.param['num_entities']
        self.num_relations = self.param['num_relations']

        self.E = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.R = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (self.embedding_dim, self.embedding_dim, self.embedding_dim)),
                         dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(self.param['input_dropout'])
        self.hidden_dropout1 = torch.nn.Dropout(self.param["hidden_dropout"])
        self.hidden_dropout2 = torch.nn.Dropout(self.param["hidden_dropout"])
        self.bn0 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward_head_batch(self, e1_idx, rel_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(rel_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def get_embeddings(self):
        return self.E.weight.data, self.R.weight.data

    def forward_triples(self, *args,**kwargs):
        raise NotImplementedError('Negative Sampling is not implemented for Tucker')

    def forward_triples_and_loss(self, *args,**kwargs):
        raise NotImplementedError('Negative Sampling is not implemented for Tucker')