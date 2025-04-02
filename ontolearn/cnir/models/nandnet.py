import torch
import torch.nn as nn
from cnir.config import CNIRConfig
from transformers import AutoModel, PreTrainedModel

class NAND(PreTrainedModel):
    config_class = CNIRConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.linear1 = nn.Linear(config.input_size, config.individual_size)
        self.linear2 = nn.Linear(config.individual_size, config.individual_size)
        self.head = nn.Linear(config.individual_size * 2, config.output_size)
        self.activation = nn.ReLU()

    def encode(self, C, D):
        x1 = self.linear1(C)
        x1 = self.activation(x1)
        x2 = self.linear1(D)
        x2 = self.activation(x2)
        x = x1 + x2
        x = self.linear2(x)
        x = self.activation(x)
        return x

    def forward(self, C, D, individual):
        x = self.encode(C, D)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if individual.ndim == 1:
            individual = individual.unsqueeze(0)
        x = torch.cat((x.repeat(individual.shape[0], 1), individual), dim=-1)
        x = self.head(x)
        # use sigmoid to make sure the output is in [0,1]
        x = torch.sigmoid(x)
        return x.squeeze()
        
class Le(PreTrainedModel):
    config_class = CNIRConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.linear1 = nn.Linear(config.input_size, config.individual_size)
        self.linear2 = nn.Linear(config.input_size, config.individual_size)
        self.linear3 = nn.Linear(config.individual_size * 2, config.individual_size)
        self.head = nn.Linear(config.individual_size * 2, config.output_size)
        self.activation = nn.ReLU()

    def encode(self, r, C):
        x1 = self.linear1(C)
        x1 = self.activation(x1)
        x2 = self.linear2(r)
        x2 = self.activation(x2)
        x = torch.cat((x1, x2), dim=-1)

        x = self.linear3(x)
        x = self.activation(x)
        return x

    def forward(self, n, r, C, individual):
        x = self.encode(r, C)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if individual.ndim == 1:
            individual = individual.unsqueeze(0)
        x = torch.cat((x.repeat(individual.shape[0], 1), individual), dim=-1)
        # x = torch.sigmoid(torch.tensor([n]))+ self.head(x)
        x = torch.sigmoid(n) + self.head(x)
        # use sigmoid to make sure the output is in [0,1]
        x = torch.sigmoid(x)

        return x

class Inverse(PreTrainedModel):
    config_class = CNIRConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.linear1 = nn.Linear(config.input_size, config.individual_size)
        self.linear2 = nn.Linear(config.individual_size, config.individual_size)
        self.linear3 = nn.Linear(config.individual_size, config.individual_size)
        self.head = nn.Linear(config.individual_size, config.input_size)
        self.activation = nn.ReLU()

    def encode(self, r):
        # Pass through the first two linear layers with ReLU activation
        x1 = self.activation(self.linear1(r))
        x2 = self.activation(self.linear2(x1))
        x3 = self.activation(self.linear3(x2))
        output = self.head(x3)
        return torch.tanh(output)

    def forward(self, r):
        return self.encode(r)

class Self(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.linear1 = nn.Linear(config.individual_size, config.individual_size)
        self.head = nn.Linear(config.individual_size, config.output_size)
        self.activation = nn.ReLU()

    def forward(self, individual):
        x = self.linear1(individual)
        x = self.activation(x)
        x = self.head(x)
        # use sigmoid to make sure the output is in [0,1]
        x = torch.sigmoid(x)
        return x