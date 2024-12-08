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

class BoxesLoss():
    def loss(pred, target, alpha=1, beta=1):
        l1 = nn.SmoothL1Loss()
        bce = nn.BCEWithLogitsLoss()

        assert pred.ndim == 3, 'Predictions must have ndim of 3, SAMPLE X REGION X OUTPUTS'
        assert target.ndim == 3, 'Targets must have ndim of 3, SAMPLE X REGION X OUTPUTS'
        assert pred.shape[0] == target.shape[0], 'The number of outputs and ground truth must be the same'
        assert pred.shape[1] == target.shape[1], 'The predicted regions and the ground truth regions must be equal'
        assert len(target[:, :, 0].unique()) == 2, 'Ground truth classification feature should only be either 0 or 1'

        l1L = alpha * l1(pred[:, :, 1:], target[:, :, 1:])
        bceL = beta * bce(pred[:, :, :1], target[:, :, :1])

        return l1L + bceL
