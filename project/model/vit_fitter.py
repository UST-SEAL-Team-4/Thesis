from .fitter import Fitter
import torch

class ViTFitter(Fitter):
    def __init__(self, model, device, epochs, loss, optim, rpn):
        super().__init__(model, device, epochs, loss, optim)
        self.rpn = rpn # must have the pretrained weights

    def train_one_epoch(self, train_loader):
        self.model.train()
        # for all samples in train_loader
        # pass through rpn
        # acquire bounding box
        # acquire box from each slice
        # feed all slices to vit
        # query for the mask of all slices
        # calculate loss
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    def validation(self, val_loader):
        self.model.eval()
        with torch.inference_mode():
            # feed all samples
            # pass through rpn
            # acquire bounding box
            # acquire box from each slice
            # feed all slices to vit
            # query for the mask of all slices
            pass