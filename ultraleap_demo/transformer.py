import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1,0,2)

        return x + self.pe

class TransformerModel(nn.Module):
    def __init__(self, num_features, feature_ntokens, d_model, nhead, num_layers, max_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(ntokens, d_model) for ntokens in feature_ntokens
        ])
        
        self.encoder = nn.Linear(d_model * num_features, d_model)  # Update the input size
        self.decoder = nn.Linear(d_model, sum(feature_ntokens))
        self.d_model = d_model
        self.num_features = num_features
        self.feature_ntokens = feature_ntokens
        self.src_mask = None
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # returns a tensor of shape (sequence_length, batch_size, ntoken)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
            self.src_mask = mask

        # Apply embedding layers to input features
        embedded_features = []
        for i, embedding in enumerate(self.feature_embeddings):
            feature = src[:, :, i].long()
            embedded_feature = embedding(feature)
            embedded_features.append(embedded_feature)
        
        # Concatenate embedded features along the feature dimension
        input_data = torch.cat(embedded_features, dim=-1)
        
        input_data = self.encoder(input_data)
        input_data = self.pos_encoder(input_data)
        output = self.transformer_encoder(input_data, self.src_mask)
        output = self.decoder(output)
    
        # Split the output into separate features
        output_features = torch.split(output, self.feature_ntokens, dim=-1)
        
        return output_features
