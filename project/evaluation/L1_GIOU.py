import torch 
import torch.nn as nn
import torchvision.ops as tvops

class L1_GIOU():
    def loss(pred, target, alpha = 1, beta = 1):
        l1 = nn.L1Loss()
        return alpha * l1(pred, target) + beta * tvops.generalized_box_iou_loss(pred, target, reduction='mean')
    def __call__(self):
        return f'L1_GIOU: '
