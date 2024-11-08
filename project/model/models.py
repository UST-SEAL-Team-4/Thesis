import torch.nn as nn

def posemb():
    pass

class GCRPN(nn.Module):
    def __init__(self, rpn, feeder, image_size, patch_size):
        super().__init__()
        self.rpn = rpn
        self.feeder = feeder
        self.image_size = image_size
        self.patch_size = patch_size

    def forward(self, mri, mask, target):
        bbox = self.rpn(mri, target)
        bbox = bbox*self.image_size
        bbox = bbox.squeeze().int().tolist()
        cmri = self.feeder(mri, bbox, self.patch_size)
        cmask = self.feeder(mask, bbox, self.patch_size)

        return cmri, cmask

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
