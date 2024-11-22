import torch.nn as nn
import torchvision.ops as tvops


class SmoothL1GiouLoss():
    def loss(pred, target, alpha = 1, beta = 0):
        l1 = nn.SmoothL1Loss()
        l1_loss = alpha * l1(pred, target)
        giou_loss = beta * (tvops.generalized_box_iou_loss(pred, target, reduction='mean').sigmoid())
        
        return l1_loss + giou_loss
       
    def __call__(self):
        return f'SmoothL1GiouLoss: '
