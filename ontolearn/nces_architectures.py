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

"""NCES architectures."""
from ontolearn.nces_modules import *


class LSTM(nn.Module):
    """LSTM module."""
    def __init__(self, knowledge_base_path, vocab, inv_vocab, max_length, input_size, proj_dim, rnn_n_layers,
                 drop_prob):
        super().__init__()
        self.name = 'LSTM'
        self.max_len = max_length
        self.proj_dim = proj_dim
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.loss = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(input_size, proj_dim, rnn_n_layers, dropout=drop_prob, batch_first=True)
        self.bn = nn.BatchNorm1d(proj_dim)
        self.fc1 = nn.Linear(2*proj_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.fc3 = nn.Linear(proj_dim, len(self.vocab)*max_length)

    def forward(self, x1, x2, target_scores=None):
        seq1, _ = self.lstm(x1)
        seq2, _ = self.lstm(x2)
        out1 = seq1.sum(1).view(-1, self.proj_dim)
        out2 = seq2.sum(1).view(-1, self.proj_dim)
        x = torch.cat([out1, out2], 1)
        x = F.gelu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.bn(x)
        x = self.fc3(x)
        x = x.reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x


class GRU(nn.Module):
    """GRU module."""
    def __init__(self, knowledge_base_path, vocab, inv_vocab, max_length, input_size, proj_dim, rnn_n_layers,
                 drop_prob):
        super().__init__()
        self.name = 'GRU'
        self.max_len = max_length
        self.proj_dim = proj_dim
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.loss = nn.CrossEntropyLoss()
        self.gru = nn.GRU(input_size, proj_dim, rnn_n_layers, dropout=drop_prob, batch_first=True)
        self.bn = nn.BatchNorm1d(proj_dim)
        self.fc1 = nn.Linear(2*proj_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.fc3 = nn.Linear(proj_dim, len(self.vocab)*max_length)

    def forward(self, x1, x2, target_scores=None):
        seq1, _ = self.gru(x1)
        seq2, _ = self.gru(x2)
        out1 = seq1.sum(1).view(-1, self.proj_dim)
        out2 = seq2.sum(1).view(-1, self.proj_dim)
        x = torch.cat([out1, out2], 1)
        x = F.gelu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.bn(x)
        x = self.fc3(x)
        x = x.reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x


class SetTransformer(nn.Module):
    """SetTransformer module."""
    def __init__(self, knowledge_base_path, vocab, inv_vocab, max_length, input_size, proj_dim, num_heads, num_seeds,
                 m, ln):
        super(SetTransformer, self).__init__()
        self.name = 'SetTransformer'
        self.max_len = max_length
        self.m = m
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.loss = nn.CrossEntropyLoss()
        self.enc = nn.Sequential(
                ISAB(input_size, proj_dim, num_heads, m, ln=ln),
                ISAB(proj_dim, proj_dim, num_heads, m, ln=ln))
        self.dec = nn.Sequential(
                PMA(proj_dim, num_heads, num_seeds, ln=ln),
                nn.Linear(proj_dim, len(self.vocab)*max_length))

    def forward(self, x1, x2):
        x1 = self.enc(x1)
        x2 = self.enc(x2)
        x = torch.cat([x1, x2], -2)
        x = self.dec(x).reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x
