from torch import Tensor
import torch
from torchmetrics.functional.detection import intersection_over_union
from torchvision.ops import box_area
from typing import Tuple
import numpy as np
from scipy.ndimage import label
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import euclidean

def compute_centroid_distance(pred, anno):
    pred_centroid = center_of_mass(pred)
    anno_centroid = center_of_mass(anno)
    return euclidean(pred_centroid, anno_centroid)

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
    return dice

def precision_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    if total_pixel_pred == 0:
        return 0
    precision = np.mean(intersect/total_pixel_pred)
    return precision

def recall_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    if total_pixel_truth == 0:
        return 0
    recall = np.mean(intersect/total_pixel_truth)
    return recall

def false_positive_rate(groundtruth_mask, pred_mask):
    false_positives = np.sum((pred_mask == 1) & (groundtruth_mask == 0))
    true_negatives = np.sum(groundtruth_mask == 0)
    if (false_positives + true_negatives) == 0:
        return 0
    fpr = false_positives / (false_positives + true_negatives)
    return fpr


def isa_rpn_metric(image_size, target_bbox, predicted_bbox):
    iou_score = intersection_over_union(predicted_bbox * image_size, target_bbox * image_size)

    inter, union = _box_inter_union(predicted_bbox * image_size, target_bbox * image_size)

    precision_score = (inter / box_area(predicted_bbox * image_size))
    recall_score = (inter / box_area(target_bbox * image_size))

    if any([precision_score, recall_score]) == 0:
        f1_score = torch.tensor(0)
    else:
        f1_score = (2 * (precision_score * recall_score)) / (precision_score + recall_score)
        # f1_score = f1_score.detach().cpu().numpy()

    return iou_score.tolist(), precision_score.squeeze().tolist(), recall_score.squeeze().tolist(), f1_score.squeeze().tolist()

def isa_vit_metric(TP, FP, FN, N):
    # true_segmentation = true_segmentation.astype(int)
    # predicted_segmentation = predicted_segmentation.astype(int)

    # dice_score = dice_coef(true_segmentation, predicted_segmentation)
    # precision_score = precision_score_(true_segmentation, predicted_segmentation)
    # recall_score = recall_score_(true_segmentation, predicted_segmentation)
    # fpr = false_positive_rate(true_segmentation, predicted_segmentation)
    
    # if any([precision_score, recall_score]) == 0:
    #     f1_score = 0
    # else:
    #     f1_score = (2 * (precision_score * recall_score)) / (precision_score + recall_score)

    precision_score =  TP/(TP+FP)
    recall_score =  TP/(TP+FN)
    f1_score =  (2*(TP/(TP+FN))*(TP/(TP+FP)))/((TP/(TP+FN))+(TP/(TP+FP)))
    fp_avg =  TP/N
    dice_score =  (2*TP)/(2*TP+FP+FN)
    return dice_score, precision_score, recall_score, f1_score, fp_avg


def count_fptpfn(pred_mask, anno_mask):
    TP = 0
    FP = 0
    FN = 0
    threshold = 0.5
    pred_mask = (pred_mask > threshold).int().detach().cpu().numpy()
    
    if all(np.equal(np.array([0, 1]), anno_mask.int().unique().detach().cpu().numpy())) != True:
        anno_mask = (anno_mask > min(anno_mask[0].unique().int().detach().cpu().numpy())).int().detach().cpu().numpy()
    else: 
        anno_mask = anno_mask.int().detach().cpu().numpy()
        
    labeled_pred, num_pred = label(pred_mask, structure=np.ones((3,3)))
    labeled_anno, num_anno = label(anno_mask, structure=np.ones((3,3)))

    used_predictions = []

    for anno_index in range(1, num_anno + 1):
        cur_anno_mask = (labeled_anno == anno_index)
        best_match = None
        best_value = None

        for pred_index in range(1, num_pred + 1):
            if pred_index in used_predictions:
                continue
                
            cur_pred_mask = (labeled_pred == pred_index)
            value = compute_centroid_distance(cur_pred_mask, cur_anno_mask)
            if value < 5.0:
                if best_value is None or value < best_value:
                    best_match = pred_index
                    best_value = value

        if best_match is not None:
            TP += 1
            used_predictions.append(best_match)
        else:
            FN += 1
    
    FP = num_pred - len(used_predictions)

    return TP, FP, FN