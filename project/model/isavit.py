import torch
import torch.nn as nn
import numpy as np
import math

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, vit_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(patch_size*patch_size, vit_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        out = self.embed(x)
        return out

class PositionalEncoding(nn.Module):
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

class SegmentationHead(nn.Module):
    def __init__(self, d_model, patch_size, channels=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, patch_size*patch_size*channels)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out

class ISAVIT(nn.Module):
    def __init__(self, d_model, patch_size, dim_ff, global_context, n_heads=1, n_layers=1):
        super().__init__()

        self.config = dict(
            d_model=d_model,
            patch_size=patch_size,
            dim_ff=dim_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            global_context = global_context
        )

        self.global_context = global_context

        self.patchem = PatchEmbedding(patch_size=patch_size, vit_dim=d_model)
        self.posenc = PositionalEncoding(d_model=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff)
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.mlp_head = SegmentationHead(d_model, patch_size, 1)

    def forward(self, x, i):
        x = self.patchem(x)
        slices = self.posenc(x)
        if self.global_context == True:
            slices = torch.cat((slices, slices[i].unsqueeze(0)), dim=0)
            global_out = self.trans_encoder(slices)
            out = global_out[-1]
        else:
            out = self.trans_encoder(slices[i].unsqueeze(0))
        # linear output projection
        out = self.mlp_head(out)
        # patch to image
        return out
