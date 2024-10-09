import torch.nn as nn

def posemb():
    pass

class GCRPN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.trans = nn.Transformer(d_model=input_size)

    def forward(self, x):
        return self.trans(x)

class GCViT(nn.Module):
    def __init__(self):
        super().__init__()
        # use needed layers

class MainModel(nn.Module):
    def __init__(self, input_size_rpn):
        super().__init__()
        self.rpn = GCRPN(input_size=input_size_rpn)
        self.vit = GCViT()

    def forward(self, brain):
        # feed to rpn slice by slice
        # use final output as bbox
        # use the bbox for all slices
        # feed all bboxes to vit
        pass
