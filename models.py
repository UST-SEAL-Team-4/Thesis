import torch.nn

def posemb():
    pass

class GCRPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans = nn.Transformer()

    def forward(self, x):
        return self.trans(x)

class GCViT(nn.Module):
    def __init__(self):
        super().__init__()
        # use needed layers

class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rpn = GCRPN()
        self.vit = GCViT()

    def forward(self, brain):
        # feed to rpn slice by slice
        # use final output as bbox
        # use the bbox for all slices
        # feed all bboxes to vit
        pass
