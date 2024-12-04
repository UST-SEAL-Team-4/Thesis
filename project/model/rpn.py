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
            nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=kernel_size , stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(2)
        )

        # Calculate shape after convolutions
        with torch.inference_mode():
            x = torch.zeros(1, in_channels, image_size, image_size)
            output = self.convs(x)
            final_d = output.numel()

        print(final_d)

        self.mlp = nn.Sequential(
            nn.Linear(final_d, output_dim)
        )


    def forward(self, x):
        out = self.convs(x)
        return self.mlp(out)

class RPN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 image_size,
                 global_context,
                 nh=5,
                 n_layers=1,
                 dim_ff=2500,
                 pretrained=False,
                 embed_model=resnet18,
                 embed_weights=(ResNet18_Weights.IMAGENET1K_V1),
                 *a,
                 **k
                 ):
        super().__init__()

        self.config = dict(
            input_dim = input_dim,
            output_dim = output_dim,
            image_size = image_size,
            nh = nh,
            n_layers = n_layers,
            dim_ff = dim_ff,
            pretrained = pretrained,
            global_context = global_context
        )

        self.global_context = global_context

        if pretrained is True:
            self.embedder = PretrainedEmbedder(embed_model, embed_weights)
        else:
            self.embedder = SliceEmbedding(image_size=image_size, output_dim=input_dim)

        # input_dim=512
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nh, dim_feedforward=dim_ff)
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.posenc = RPNPositionalEncoding(d_model=input_dim)

        self.output_features = 3
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim*3),
            nn.ReLU(),
            nn.Linear(input_dim*3, output_dim),
            nn.Conv1d(in_channels=1, out_channels=self.output_features, kernel_size=1, stride=1)
            # nn.Sigmoid(),
        )

    # def forward(self, x, i):
    #
    #     slices = self.embedder(x)
    #     slices = slices.view(slices.shape[0], 1, -1)
    #     slices = self.posenc(slices)
    #     if self.global_context == True:
    #         out = self.trans_encoder(slices)
    #     out = self.trans_encoder(slices[i])
    #     out = self.fc(out)
    #
    #     return out

    def forward(self, x, i):

        if self.global_context == True:
            slices = self.embedder(x)
            slices = slices.view(slices.shape[0], 1, -1)
            slices = self.posenc(slices)
            slices = torch.cat((slices, slices[i].unsqueeze(0)), dim=0)
            global_out = self.trans_encoder(slices)
            out = global_out[-1]
        else:
            slice = x[i].unsqueeze(0)
            slice = self.embedder(slice)
            slice = slice.view(slice.shape[0], 1, -1)
            slice = self.posenc(slice)
            out = self.trans_encoder(slice.squeeze(0))

        assert len(out.shape) == 2
        assert out.shape[0] == 1
        assert out.shape[1] == 512

        out = self.fc(out)

        assert len(out.shape) == 2
        assert out.shape[0] == self.output_features
        assert out.shape[1] == self.config['output_dim']

        out = out.permute(1, 0)

        return out
