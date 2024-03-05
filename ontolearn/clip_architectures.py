import torch, torch.nn as nn
import random
from typing import List
from ontolearn.nces_modules import *    

class LengthLearner_LSTM(nn.Module):
    """LSTM architecture"""
    def __init__(self, input_size, output_size, proj_dim=256, rnn_n_layers=2, drop_prob=0.2):
        super().__init__()
        self.name = 'LSTM'
        self.loss = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(input_size, proj_dim, rnn_n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(2*proj_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.fc3 = nn.Linear(proj_dim, output_size)  
    
    def forward(self, x1, x2):
        ''' Forward pass through the network.'''
        x1, _ = self.lstm(x1)
        x1 = x1.sum(1).contiguous().view(x1.shape[0], -1)
        x2, _ = self.lstm(x2)
        x2 = x2.sum(1).contiguous().view(x2.shape[0], -1)
        x = torch.cat([x1, x2], dim=-1)
        x = self.fc1(x)
        x = torch.selu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + torch.tanh(x)
        x = self.fc3(x)
        return x

class LengthLearner_GRU(nn.Module):
    """GRU architecture"""
    def __init__(self, input_size, output_size, proj_dim=256, rnn_n_layers=2, drop_prob=0.2):
        super().__init__()
        self.name = 'GRU'
        self.loss = nn.CrossEntropyLoss()
        self.gru = nn.GRU(input_size, proj_dim, rnn_n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(2*proj_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.fc3 = nn.Linear(proj_dim, output_size)

    def forward(self, x1, x2):
        ''' Forward pass through the network.'''
        x1, _ = self.gru(x1)
        x1 = x1.sum(1).contiguous().view(x1.shape[0], -1)
        x2, _ = self.gru(x2)
        x2 = x2.sum(1).contiguous().view(x2.shape[0], -1)
        x = torch.cat([x1, x2], dim=-1)
        x = self.fc1(x)
        x = torch.selu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + torch.tanh(x)
        x = self.fc3(x)
        return x
        

class LengthLearner_CNN(nn.Module):
    """CNN architecture"""
    def __init__(self, input_size, output_size, num_examples, proj_dim=256, kernel_size: list=[[5,7], [5,7]], stride: list=[[3,3], [3,3]], drop_prob=0.2):
        super().__init__()
        assert isinstance(kernel_size, list) and isinstance(kernel_size[0], list), "kernel size and stride must be lists of lists, e.g., [[5,7], [5,7]]"
        self.name = 'CNN'
        self.loss = nn.CrossEntropyLoss()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(kernel_size[0][0], kernel_size[0][1]), stride=(stride[0][0], stride[0][1]), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(kernel_size[1][0], kernel_size[1][1]), stride=(stride[1][0], stride[1][1]), padding=(0,0))
        self.dropout1d = nn.Dropout(drop_prob)
        self.dropout2d = nn.Dropout2d(drop_prob)
        conv_out_dim = 3536
        self.fc1 = nn.Linear(conv_out_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.fc3 = nn.Linear(proj_dim, output_size)
        
    def forward(self, x1, x2):
        ''' Forward pass through the network.'''
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x = torch.cat([x1, x2], dim=-2)
        x = self.conv1(x)
        x = torch.selu(x)
        x = self.dropout2d(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.selu(x)
        x = self.dropout1d(x)
        x = self.fc2(x)
        x = x + torch.tanh(x)
        x = self.fc3(x)
        return x
    
        
class LengthLearner_SetTransformer(nn.Module):
    """SetTransformer architecture."""
    def __init__(self, input_size, output_size, proj_dim=256, num_heads=4, num_seeds=1, num_inds=32):
        super().__init__()
        self.name = 'SetTransformer'
        self.loss = nn.CrossEntropyLoss()
        self.enc = nn.Sequential(
                ISAB(input_size, proj_dim, num_heads, num_inds),
                ISAB(proj_dim, proj_dim, num_heads, num_inds))
        self.dec = nn.Sequential(
                PMA(proj_dim, num_heads, num_seeds),
                nn.Linear(proj_dim, output_size))

    def forward(self, x1, x2):
        ''' Forward pass through the network.'''
        x1 = self.enc(x1)
        x2 = self.enc(x2)
        x = torch.cat([x1, x2], dim=-2)
        x = self.dec(x).squeeze()
        return x