import torch
import torch.nn as nn
from transformers import PreTrainedModel
from cnir.config import CNIRConfig
from .utils import PMA, PositionalEncoding

class CNIRTransformer(PreTrainedModel):
    config_class = CNIRConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.loss_fn = nn.BCELoss()
        self.max_length = config.max_length
        # Embedding + Positional Encoding
        self.embedding = nn.Sequential(nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=self.config.pad_token_id),
                                       PositionalEncoding(d_model=config.embedding_dim, dropout=config.pe_dropout, max_length=config.max_length)
                                      )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.embedding_dim, nhead=config.num_attention_heads, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        
        # Fully connected
        self.fc = nn.Sequential(nn.Linear(2*config.embedding_dim, config.embedding_dim),
                                nn.GELU(),
                                nn.Dropout(0.1),
                                nn.Linear(config.embedding_dim, 1))
        # Activation function
        self.activation = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, label_features, labels=None):
        embeddings = self.embedding(input_ids)# + self.positional_encoding[:input_ids.size(1), :]
        transformer_output = self.encoder(embeddings,
                                          #src_key_padding_mask=(~attention_mask.bool())
                                                     ).mean(1)
        #transformer_output = self.decoder(transformer_output.mean(1))#.mean(dim=1)#.flatten(start_dim=1, end_dim=2)
        n = label_features.shape[0]
        n_repeats = n // transformer_output.shape[0]
        transformer_output = torch.repeat_interleave(transformer_output, repeats=n_repeats, dim=0)
        final_representation = torch.cat((transformer_output, label_features), dim=1)
        logits = self.fc(final_representation)
        probabilities = self.activation(logits).squeeze()
        
        if labels is not None:
            loss = self.loss_fn(probabilities, labels)
            return probabilities, loss
        else:
            return probabilities