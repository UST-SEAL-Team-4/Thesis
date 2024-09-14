import torch
import torch.nn as nn
import math

class RPNPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RPN(nn.Module):
    def __init__(self, input_dim, output_dim, nh, dim_ff):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nh, dim_feedforward=dim_ff)
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.posenc = RPNPositionalEncoding(d_model=input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.lrel = nn.ELU()

    def forward(self, x):

        x = self.posenc(x)
        x = self.trans_encoder(x)
        x = self.fc(x)
        x = self.lrel(x)

        return x
