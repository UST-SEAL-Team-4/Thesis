import torch
import torch.nn as nn

def posemb():
    pass

class GCRPN(nn.Module):
    def __init__(self, rpn, feeder, image_size, patch_size, model_b=False):
        super().__init__()
        self.rpn = rpn
        self.feeder = feeder
        self.image_size = image_size
        self.patch_size = patch_size
        self.model_b = model_b

    def forward(self, mri, mask, target):
        if self.model_b == False:
            bbox = self.rpn(mri, target)
            bbox = bbox*self.image_size
            bbox = bbox.squeeze().int().tolist()
            cmri = self.feeder(mri, bbox, self.patch_size)
            cmask = self.feeder(mask, bbox, self.patch_size)
            return cmri, mask
        else:
            y = self.rpn(mri, target)
            ts = y.argmax().tolist()

            cmri = []
            for i in range(mri.shape[0]):
                slc = self.feeder(mri[i].unsqueeze(0).float(), ts).unsqueeze(0)
                cmri.append(slc)

            cmask = []
            for i in range(mask.shape[0]):
                slc = self.feeder(mask[i].unsqueeze(0).float(), ts).unsqueeze(0)
                cmask.append(slc)

            return torch.stack(cmri), torch.stack(cmask)

        # return cmri, cmask

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
