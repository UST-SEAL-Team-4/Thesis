import torch
import torch.nn as nn
import math
from torchvision.models import resnet18, ResNet18_Weights

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

class PretrainedEmbedder(nn.Module):
    def __init__(self, model, weights):
        super().__init__()
        pretrained_model = model(weights=weights)
        no_classifier = list(pretrained_model.children())[:-1]
        self.embedder = nn.Sequential(*(no_classifier))

    def forward(self, x):
        return self.embedder(x)

class RPN(nn.Module):
    def __init__(self, input_dim, output_dim, nh, dim_ff, embed_model=resnet18, embed_weights=(ResNet18_Weights.IMAGENET1K_V1)):
        super().__init__()

        self.embedder = PretrainedEmbedder(embed_model, embed_weights)
        input_dim=512
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nh, dim_feedforward=dim_ff)
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.posenc = RPNPositionalEncoding(d_model=input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.lrel = nn.ELU()

    def forward(self, x, i):

        slices = self.embedder(x)
        slices = slices.view(slices.shape[0], 1, -1)
        slices = self.posenc(slices)
        out = self.trans_encoder(slices)
        out = self.trans_encoder(slices[i])
        out = self.fc(out)
        out = self.lrel(out)

        return out
