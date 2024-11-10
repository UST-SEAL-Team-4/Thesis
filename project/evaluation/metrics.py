from torch import Tensor
import torch
from torchmetrics.functional.detection import intersection_over_union
from torchvision.ops import box_area
from typing import Tuple
import numpy as np

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    if total_sum == 0:
        return 0
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 4) #round up to 4 decimal places

def precision_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    if total_pixel_pred == 0:
        return 0
    precision = np.mean(intersect/total_pixel_pred)
    return round(precision, 4)

def recall_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    if total_pixel_truth == 0:
        return 0
    recall = np.mean(intersect/total_pixel_truth)
    return round(recall, 4)

def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    if (union + xor - intersect) == 0:
        return 0
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 4)

def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    if union == 0:
        return 0
    iou = np.mean(intersect/union)
    return round(iou, 4)

def isa_rpn_metric(image_size, target_bbox, predicted_bbox):
    iou_score = intersection_over_union(predicted_bbox * image_size, target_bbox * image_size)

    inter, union = _box_inter_union(predicted_bbox * image_size, target_bbox * image_size)

    precision_score = (inter / box_area(predicted_bbox * image_size))
    recall_score = (inter / box_area(target_bbox * image_size))

    if any([precision_score, recall_score]) == 0:
        f1_score = 0
    else:
        f1_score = (2 * (precision_score * recall_score)) / (precision_score + recall_score)
        f1_score = f1_score.detach().cpu().numpy()

    return iou_score, precision_score, recall_score, f1_score

def isa_vit_metric(predicted_segmentation, true_segmentation):
    dice_score = dice_coef(true_segmentation, predicted_segmentation)
    precision_score = precision_score_(true_segmentation, predicted_segmentation)
    recall_score = recall_score_(true_segmentation, predicted_segmentation)
    accuracy_score = accuracy(true_segmentation, predicted_segmentation)

    return dice_score, precision_score, recall_score, accuracy_score