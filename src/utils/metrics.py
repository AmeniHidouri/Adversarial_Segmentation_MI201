"""
Evaluation metrics for segmentation attacks.
"""

import torch
import numpy as np


def compute_iou(pred, target):
    """
    Compute Intersection over Union (IoU) score.
    
    Args:
        pred: Predicted segmentation map
        target: Ground truth segmentation map
    
    Returns:
        float: IoU score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    intersection = np.logical_and(pred == target, target > 0).sum()
    union = np.logical_or(pred > 0, target > 0).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def compute_dice(pred, target):
    """
    Compute Dice coefficient.
    
    Args:
        pred: Predicted segmentation map
        target: Ground truth segmentation map
    
    Returns:
        float: Dice coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    intersection = np.logical_and(pred == target, target > 0).sum()
    total = (pred > 0).sum() + (target > 0).sum()
    
    if total == 0:
        return 0.0
    
    return float(2 * intersection / total)


def compute_pixel_accuracy(pred, target):
    """
    Compute pixel accuracy.
    
    Args:
        pred: Predicted segmentation map
        target: Ground truth segmentation map
    
    Returns:
        float: Pixel accuracy
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    correct = (pred == target).sum()
    total = pred.size
    
    return float(correct / total)
