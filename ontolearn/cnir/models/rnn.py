import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from cnir.config import CNIRConfig
from cnir.utils import gen_pe

class CNIRLSTM(PreTrainedModel):
    config_class = CNIRConfig
    def __init__(self, config, batch_training=True):
        super().__init__(config)
        self.config = config
        self.batch_training = batch_training
        self.loss_fn = nn.BCELoss()
        self.max_length = config.max_length
        self.embedding = nn.Embedding(config.vocab_size, config.individual_size, padding_idx=self.config.pad_token_id)
        
        self.lstm = nn.LSTM(input_size=config.individual_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_rnn_layers,
                            batch_first=True)

        self.fc = nn.Sequential(nn.Linear(config.hidden_size + config.individual_size, config.hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(config.hidden_size, config.output_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, label_features, labels=None, mask = None):
        individual_embeddings = label_features
        embeddings = self.embedding(input_ids)
        # hidden_state shape: [num_layers, batch_size, hidden_dim]
        seq, _ = self.lstm(embeddings)
        # [batch_size, hidden_dim]
        #print(seq.shape)
        final_hidden_state = seq.mean(1)

        if self.batch_training:
            batch_size = final_hidden_state.shape[0]
            n = individual_embeddings.shape[0]
            n_repeats = n // batch_size
            final_hidden_state = torch.repeat_interleave(final_hidden_state, repeats=n_repeats, dim=0)#final_hidden_state.unsqueeze(1).repeat(1, n, 1)
            combined = torch.cat((final_hidden_state, individual_embeddings), dim=1)  # [batch_size, hidden_dim + individual_embedding_dim]
            if mask is not None:
                mask = mask.to(self.config.device)
                inverted_mask = ~mask
                combined = combined.masked_fill(inverted_mask.unsqueeze(2), -1e9)
        else:
            n = individual_embeddings.shape[0]
            final_hidden_state = final_hidden_state.expand(n, -1)
            combined = torch.cat((final_hidden_state, individual_embeddings),
                                 dim=1)
        output = self.fc(combined)

        probabilities = self.sigmoid(output)

        if labels is not None:
            if n > 1:
                probabilities = probabilities.squeeze()
            else:
                probabilities = probabilities.squeeze(0)

            if self.batch_training and batch_size == 1:
                labels = labels.squeeze()

            if self.batch_training and mask is not None:
                probabilities = probabilities.masked_select(mask)
                labels = labels.masked_select(mask)

            loss = self.loss_fn(probabilities, labels)
            return probabilities, loss
        else:
            return probabilities
    
        
class CNIRGRU(PreTrainedModel):
    config_class = CNIRConfig

    def __init__(self, config, batch_training=True):
        super().__init__(config)
        self.config = config
        self.batch_training = batch_training
        self.loss_fn = nn.BCELoss()
        self.max_length = config.max_length
        self.embedding = nn.Embedding(config.vocab_size, config.individual_size, padding_idx=self.config.pad_token_id)

        self.gru = nn.GRU(input_size=config.individual_size,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_rnn_layers,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Linear(config.hidden_size + config.individual_size, config.hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(config.hidden_size, config.output_size))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, label_features, labels=None, mask = None):
        individual_embeddings = label_features
        embeddings = self.embedding(input_ids)
        # hidden_state shape: [num_layers, batch_size, hidden_dim]??
        seq, _ = self.gru(embeddings)
        # [batch_size, hidden_dim]
        final_hidden_state = seq.mean(1)
        if self.batch_training:
            batch_size = final_hidden_state.shape[0]
            n = individual_embeddings.shape[0]
            n_repeats = n // batch_size
            final_hidden_state = torch.repeat_interleave(final_hidden_state, repeats=n_repeats, dim=0)
            combined = torch.cat((final_hidden_state, individual_embeddings),
                                 dim=1)
            if mask is not None:
                mask = mask.to(self.config.device)
                inverted_mask = ~mask
                combined = combined.masked_fill(inverted_mask.unsqueeze(2), -1e9)
        else:
            n = individual_embeddings.shape[0]
            final_hidden_state = final_hidden_state.expand(n, -1)
            combined = torch.cat((final_hidden_state, individual_embeddings),
                                 dim=1)
        output = self.fc(combined)
        probabilities = self.sigmoid(output)
        if labels is not None:
            if n > 1:
                probabilities = probabilities.squeeze()
            else:
                probabilities = probabilities.squeeze(0)
            if self.batch_training and batch_size == 1:
                labels = labels.squeeze()
            if self.batch_training and mask is not None:
                probabilities = probabilities.masked_select(mask)
                labels = labels.masked_select(mask)
            loss = self.loss_fn(probabilities, labels)
            return probabilities, loss
        else:
            return probabilities