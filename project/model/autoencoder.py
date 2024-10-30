import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

class Decoder(nn.Module):
    def __init__(self, image_size, input_dim, in_channels=1, out_channels=1, kernel_size=2, stride=2):
        super().__init__()
        self.upconvs = nn.Sequential(
            nn.Unflatten(2, (int(input_dim**.5), int(input_dim**.5))),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=24, kernel_size=kernel_size, stride=stride),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=24, out_channels=out_channels, kernel_size=kernel_size+1, stride=stride+1),
            nn.ELU(),
        )

    def forward(self, x):
        return self.upconvs(x)