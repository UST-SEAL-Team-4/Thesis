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

class SliceEmbedding(nn.Module):
    def __init__(self, image_size, output_dim, in_channels=1, out_channels=1, kernel_size=2, stride=2):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2),
        )

        # Calculate shape after convolutions
        def out_d(d, k=kernel_size, p=0, s=stride):
            return ((d - k + 2*p)/s) + 1

        d = image_size
        for i in range(len(self.convs)):
            d = int(out_d(d))

        final_d = d*d
        print(final_d)

        self.mlp = nn.Sequential(
            nn.Linear(final_d, output_dim)
        )


    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.shape[0], 1, -1)
        return self.mlp(out)

class RPN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 image_size,
                 nh=5,
                 dim_ff=2500,
                 pretrained=False,
                 embed_model=resnet18,
                 embed_weights=(ResNet18_Weights.IMAGENET1K_V1),
                 *a,
                 **k
                 ):
        super().__init__()

        if pretrained is True:
            self.embedder = PretrainedEmbedder(embed_model, embed_weights)
        else:
            self.embedder = SliceEmbedding(image_size=image_size, output_dim=input_dim)

        # input_dim=512
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
