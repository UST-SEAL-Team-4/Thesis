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
    
class ImagePatchEmbedding(nn.Module):
    def __init__(self, patch_size, vit_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(patch_size*patch_size, vit_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        out = self.embed(x)
        return out

class TraditionalPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, vit_dim, channels=1):
        super().__init__()

        # image to patches
        self.unflatter = nn.Unflatten(2, (image_size, image_size))
        self.i2p = nn.Unfold(kernel_size = patch_size, stride = patch_size)
        self.emb = nn.Linear(in_features = patch_size*patch_size*channels, out_features=vit_dim)

    def forward(self, x):
        assert len(x.shape) == 3, 'Input must have three dimensions'
        out = self.unflatter(x)
        out = self.i2p(out)
        out = out.permute(0, 2, 1)

        out = self.emb(out)
        return out

class ImageTraditionalPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, vit_dim, channels=1):
        super().__init__()

        # image to patches
        self.unflatter = nn.Unflatten(2, (image_size, image_size))
        self.i2p = nn.Unfold(kernel_size = patch_size, stride = patch_size)
        self.emb = nn.Linear(in_features = patch_size*patch_size*channels, out_features=vit_dim)

    def forward(self, x):
        assert len(x.shape) == 3, 'Input must have three dimensions'
        out = self.unflatter(x)
        out = self.i2p(out)
        out = out.permute(0, 2, 1)

        out = self.emb(out)
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
    def __init__(self, d_model, patch_size, global_context, patchpatch_size, channels=1):
        super().__init__()
        self.gc = global_context
        if global_context == True:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, patch_size*patch_size*channels)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, patchpatch_size*patchpatch_size*channels),
            )
            self.fold = nn.Fold(
                output_size = (patch_size, patch_size),
                kernel_size = patchpatch_size,
                stride = patchpatch_size
            )

    def forward(self, x):
        out = self.mlp(x)
        if self.gc == False:
            out = out.permute(0, 2, 1)
            out = self.fold(out)
            out = out.flatten(1)
        return out
    
class ImageSegmentationHead(nn.Module):
    def __init__(self, d_model, patch_size, global_context, patchpatch_size, channels=1):
        super().__init__()
        self.gc = global_context
        if global_context == True:
            self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # [1, 512] -> [64, 512]
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # [64, 512] -> [128, 512]
            nn.ReLU(),
            nn.Flatten(),  # Flatten the [128, 512] to [1, 128*512]
            nn.Linear(128 * 512, patch_size * patch_size)
            )
        else:
            self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # [1, 512] -> [64, 512]
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # [64, 512] -> [128, 512]
            nn.ReLU(),
            nn.Flatten(),  # Flatten the [128, 512] to [1, 128*512]
            nn.Linear(128 * 512, patch_size * patch_size)
            )
            self.fold = nn.Fold(
                output_size = (patch_size, patch_size),
                kernel_size = patchpatch_size,
                stride = patchpatch_size
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.mlp(x)
        if self.gc == False:
            out = out.permute(0, 2, 1)
            out = self.fold(out)
            out = out.flatten(1)
        return out

class ISAVIT(nn.Module):
    def __init__(self, d_model, patch_size, dim_ff, global_context, patchpatch_size=2, n_heads=1, n_layers=1):
        super().__init__()

        self.config = dict(
            d_model=d_model,
            patch_size=patch_size,
            dim_ff=dim_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            global_context = global_context,
            patchpatch_size = patchpatch_size
        )

        self.global_context = global_context

        if self.global_context == True:
            self.patchem = PatchEmbedding(patch_size=patch_size, vit_dim=d_model)
            self.image_patchem = ImagePatchEmbedding(patch_size=300, vit_dim=d_model)
        else:
            self.patchem = TraditionalPatchEmbedding(
                image_size=patch_size,
                patch_size=patchpatch_size,
                vit_dim=d_model
            )
            self.image_patchem = ImageTraditionalPatchEmbedding(
                image_size=300,
                patch_size=patchpatch_size,
                vit_dim=d_model
            )
        self.posenc = PositionalEncoding(d_model=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff)
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.mlp_head = SegmentationHead(
            d_model=d_model,
            patch_size=patch_size,
            global_context=global_context,
            patchpatch_size=patchpatch_size,
            channels=1,
        )
        self.image_mlp_head = SegmentationHead(
            d_model=d_model,
            patch_size=300,
            global_context=global_context,
            patchpatch_size=patchpatch_size,
            channels=1,
        )

    # def forward(self, x, i):
    #     x = self.patchem(x)
    #     print(f'AFTER PE: {x.shape}')
    #     slices = self.posenc(x)
    #     if self.global_context == True:
    #         out = self.trans_encoder(slices)
    #     out = self.trans_encoder(slices[i])
    #     # linear output projection
    #     out = self.mlp_head(out)
    #     # patch to image
    #     return out

    def forward(self, x, i):

        if x.shape[-1] == 1024:
            if self.global_context == True:
                x = self.patchem(x)
                slices = self.posenc(x)
                slices = torch.cat((slices, slices[i].unsqueeze(0)), dim=0)
                global_out = self.trans_encoder(slices)
                out = global_out[-1]
            else:
                slice = x[i].unsqueeze(0)
                out = self.patchem(slice)
                out = self.posenc(out)
                out = self.trans_encoder(out)
                
            # linear output projection
            out = self.mlp_head(out)
            # patch to image
            return out
        else:
            if self.global_context == True:
                x = self.image_patchem(x)
                slices = self.posenc(x)
                slices = torch.cat((slices, slices[i].unsqueeze(0)), dim=0)
                global_out = self.trans_encoder(slices)
                out = global_out[-1]
            else:
                slice = x[i].unsqueeze(0)
                out = self.image_patchem(slice)
                out = self.posenc(out)
                out = self.trans_encoder(out)

            # linear output projection
            out = self.image_mlp_head(out)
            # patch to image
            return out
