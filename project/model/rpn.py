import torch
import torch.nn as nn

class RPN(nn.Module):
    def __init__(self, input_dim, output_dim, nh, dim_ff):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nh, dim_feedforward=dim_ff)
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, output_dim)
        self.lrel = nn.ELU()

    def forward(self, x):

        x = self.trans_encoder(x)
        x = self.fc(x)
        x = self.lrel(x)

        return x
