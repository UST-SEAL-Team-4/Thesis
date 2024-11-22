import torch 
import torch.nn as nn
import torchvision.ops as tvops


class l1_giou():
    def loss(pred, target, alpha = 1, beta = 0):
        l1 = nn.L1Loss()
        l1_loss = alpha * l1(pred, target)
        giou_loss = beta * (tvops.generalized_box_iou_loss(pred, target, reduction='mean').sigmoid())
        return l1_loss + giou_loss
       
    def __call__(self):
        return f'L1_GIOU: '
