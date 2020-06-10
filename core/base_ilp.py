import torch
from torch.nn.init import xavier_normal_
from torch.nn import functional as F


class DCL(torch.nn.Module):
    """
    Differenciable Concept Learner
    """

    def __init__(self, args):
        super(DCL, self).__init__()

        self.embedding_dim = 50
        self.num_instances = args['num_instances']
        self.num_outputs = args['num_of_outputs']
        self.embedding = torch.nn.Embedding(args['num_instances'], self.embedding_dim, padding_idx=0)

        self.fc1 = torch.nn.Linear(self.embedding_dim * args['num_of_inputs_for_model'],
                                   self.embedding_dim * args['num_of_inputs_for_model'])

        self.fc2 = torch.nn.Linear(self.embedding_dim * args['num_of_inputs_for_model'],
                                   self.embedding_dim * args['num_of_inputs_for_model'])

        self.fc3 = torch.nn.Linear(self.embedding_dim * args['num_of_inputs_for_model'], args['num_of_outputs'])

        # (1) https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/5
        # (2) https://discuss.pytorch.org/t/userwarning-size-average-and-reduce-args-will-be-deprecated-please-use-reduction-sum-instead/24629/5

        self.loss = torch.nn.KLDivLoss(reduction='sum')

    def init(self):
        xavier_normal_(self.embedding.weight.data)

    def forward(self, X):
        X = self.embedding(X).flatten()
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return torch.softmax(X,
                             dim=0)  # https://discuss.pytorch.org/t/why-the-torch-nn-functional-log-softmax-returns-the-same-data/33355


class LengthClassifier(torch.nn.Module):
    def __init__(self, args):
        super(LengthClassifier, self).__init__()

        self.embedding_dim = args['num_dim']
        self.num_instances = args['num_instances']
        self.num_outputs = args['num_of_outputs']
        self.embedding = torch.nn.Embedding(args['num_instances'], self.embedding_dim, padding_idx=0)

        self.fc1 = torch.nn.Linear(self.embedding_dim * args['num_of_inputs_for_model'],
                                   self.embedding_dim * args['num_of_inputs_for_model'])

        self.fc2 = torch.nn.Linear(self.embedding_dim * args['num_of_inputs_for_model'],
                                   self.embedding_dim * args['num_of_inputs_for_model'])

        self.fc3 = torch.nn.Linear(self.embedding_dim * args['num_of_inputs_for_model'], args['num_of_outputs'])

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.embedding.weight.data)

    def forward(self, idx):
        emb_idx = self.embedding(idx)

        # reshape
        emb_idx = emb_idx.reshape(emb_idx.shape[0], emb_idx.shape[1] * emb_idx.shape[2])
        emb_idx = F.relu(self.fc1(emb_idx))
        emb_idx = F.relu(self.fc2(emb_idx))
        emb_idx = self.fc3(emb_idx)

        return torch.softmax(emb_idx, dim=1)


class DeepT(torch.nn.Module):
    """ Deep tunnelling for Refinement Operators"""

    def __init__(self, args):
        super(DeepT, self).__init__()

        self.embedding_dim = args['num_dim']
        self.num_instances = args['num_instances']
        self.num_outputs = args['num_of_outputs']
        self.embedding = torch.nn.Embedding(args['num_instances'], self.embedding_dim, padding_idx=0)
        self.bn1_emb = torch.nn.BatchNorm1d(1)

        self.fc1 = torch.nn.Linear(self.embedding_dim*2,
                                   self.embedding_dim * args['num_of_inputs_for_model'])
        self.bn1_h1 = torch.nn.BatchNorm1d(self.embedding_dim * args['num_of_inputs_for_model'])

        self.fc2 = torch.nn.Linear(self.embedding_dim * args['num_of_inputs_for_model'],
                                   self.embedding_dim * args['num_of_inputs_for_model'])
        self.bn1_h2 = torch.nn.BatchNorm1d(self.embedding_dim * args['num_of_inputs_for_model'])

        self.fc3 = torch.nn.Linear(self.embedding_dim * args['num_of_inputs_for_model'], args['num_of_outputs'])
        self.bn1_h3 = torch.nn.BatchNorm1d(args['num_of_outputs'])

        self.loss = torch.nn.KLDivLoss(reduction='sum')
        # self.loss=torch.nn.CrossEntropyLoss()

    def init(self):
        xavier_normal_(self.embedding.weight.data)

    def forward(self, idx):
        emb_idx = self.embedding(idx)
        # reshape
        emb_idx = emb_idx.reshape(emb_idx.shape[0], 1, emb_idx.shape[1] * emb_idx.shape[2])

        emb_idx = F.relu(self.fc1(emb_idx))
        emb_idx = self.bn1_emb(emb_idx)
        emb_idx = F.relu(self.fc2(emb_idx))
        # emb_idx = self.bn1_h2(emb_idx)
        emb_idx = self.fc3(emb_idx)
        # emb_idx = self.bn1_h3(emb_idx)

        emb_idx=emb_idx.squeeze()
        return torch.softmax(emb_idx, dim=1)