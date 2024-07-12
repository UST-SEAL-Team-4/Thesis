from .fitter import Fitter
import torch

class RPNFitter(Fitter):
    def train_one_epoch(self, train_loader):
        # self.model.train()
        # for all samples in train_loader
        # feed each slice to rpn
        # requery rpn with 
        # calculate loss
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        pass
        
    def validation(self, val_loader):
        self.model.eval()
        with torch.inference_mode():
            # feed all samples
            # get prediction per slice
            # calculate loss
            pass