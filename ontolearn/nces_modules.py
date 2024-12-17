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

"""NCES modules."""
# From https://github.com/juho-lee/set_transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    """MAB module."""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """SAB module."""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """ISAB module."""
    def __init__(self, dim_in, dim_out, num_heads, m, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, m, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """PMA module."""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# Convolutional Complex Knowledge Graph Embeddings
class ConEx(torch.nn.Module):
    """ Convolutional Complex Knowledge Graph Embeddings"""

    def __init__(self, embedding_dim, num_entities, num_relations, input_dropout, feature_map_dropout, kernel_size, num_of_output_channels):
        super(ConEx, self).__init__()
        self.name = 'ConEx'
        self.loss = torch.nn.BCELoss()
        self.embedding_dim = embedding_dim//2
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.input_dropout = input_dropout
        self.feature_map_dropout = feature_map_dropout
        self.kernel_size = kernel_size
        self.num_of_output_channels = num_of_output_channels

        # Embeddings.
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary i

        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout)
        self.input_dp_ent_i = torch.nn.Dropout(self.input_dropout)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout)
        self.input_dp_rel_i = torch.nn.Dropout(self.input_dropout)

        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(self.embedding_dim)

        # Convolution
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                     kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        # Formula for convolution output shape: (input_dim + 2* padding - kernel_size) / (stride) + 1
        self.fc_num_input = (self.embedding_dim+2-self.kernel_size+1) * (4+2-self.kernel_size+1) * self.num_of_output_channels 
        self.fc = torch.nn.Linear(self.fc_num_input, self.embedding_dim * 2)

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = torch.nn.BatchNorm1d(self.embedding_dim * 2)
        self.feature_dropout = torch.nn.Dropout2d(self.feature_map_dropout)

    def residual_convolution(self, C_1, C_2):
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim)], 2)
        
        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))
        x = self.feature_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.bn_conv2(self.fc(x)))
        return torch.chunk(x, 2, dim=1)

    def forward_head_batch(self, *, e1_idx, rel_idx):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \\in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.bn_ent_real(self.emb_ent_real(e1_idx))
        emb_head_i = self.bn_ent_i(self.emb_ent_i(e1_idx))
        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.bn_rel_real(self.emb_rel_real(rel_idx))
        emb_rel_i = self.bn_rel_i(self.emb_rel_i(rel_idx))

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_i),
                                        C_2=(emb_rel_real, emb_rel_i))
        a, b = C_3

        # (3) Apply dropout out on (1).
        emb_head_real = self.input_dp_ent_real(emb_head_real)
        emb_head_i = self.input_dp_ent_i(emb_head_i)
        emb_rel_real = self.input_dp_rel_real(emb_rel_real)
        emb_rel_i = self.input_dp_rel_i(emb_rel_i)
        # (4)
        # (4.1) Hadamard product of (2) and (1).
        # (4.2) Hermitian product of (4.1) and all entities.
        real_real_real = torch.mm(a * emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        real_imag_imag = torch.mm(a * emb_head_real * emb_rel_i, self.emb_ent_i.weight.transpose(1, 0))
        imag_real_imag = torch.mm(b * emb_head_i * emb_rel_real, self.emb_ent_i.weight.transpose(1, 0))
        imag_imag_real = torch.mm(b * emb_head_i * emb_rel_i, self.emb_ent_real.weight.transpose(1, 0))
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        return torch.sigmoid(score)

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def init(self):
        xavier_normal_(self.emb_ent_real.weight.data)
        xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_i.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb, rel_emb

    def forward_triples(self, *, e1_idx, rel_idx, e2_idx):
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        # (1.2) Complex embeddings of relations.
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)

        # (1.2) Complex embeddings of tail entities.
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_i),
                                        C_2=(emb_rel_real, emb_rel_i))
        a, b = C_3

        # (3) Apply dropout out on (1).
        emb_head_real = self.input_dp_ent_real(emb_head_real)
        emb_head_i = self.input_dp_ent_i(emb_head_i)
        emb_rel_real = self.input_dp_rel_real(emb_rel_real)
        emb_rel_i = self.input_dp_rel_i(emb_rel_i)
        # (4)
        # (4.1) Hadamard product of (2) and (1).
        # (4.2) Hermitian product of (4.1) and tail entities
        # Compute multi-linear product embeddings
        real_real_real = (a * emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (a * emb_head_real * emb_rel_i * emb_tail_i).sum(dim=1)
        imag_real_imag = (b * emb_head_i * emb_rel_real * emb_tail_i).sum(dim=1)
        imag_imag_real = (b * emb_head_i * emb_rel_i * emb_tail_real).sum(dim=1)
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        return torch.sigmoid(score)

    def forward_triples_and_loss(self, e1_idx, rel_idx, e2_idx, targets):
        scores = self.forward_triples(e1_idx=e1_idx, rel_idx=rel_idx, e2_idx=e2_idx)
        return self.loss(scores, targets)