"""
Model loading utilities for segmentation models.
"""

import torch
import torchvision.models.segmentation as models
import torchvision


def load_deeplabv3(pretrained=True, device=None):
    """
    Load DeepLabV3 with ResNet50 backbone.
    
    Args:
        pretrained (bool): Load pretrained weights
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        torch.nn.Module: DeepLabV3 model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.deeplabv3_resnet50(pretrained=pretrained).to(device)
    model.eval()
    return model


def load_maskrcnn(pretrained=True, device=None):
    """
    Load Mask R-CNN with ResNet50 backbone.
    
    Args:
        pretrained (bool): Load pretrained weights
        device (str): Device to load model on
    
    Returns:
        torch.nn.Module: Mask R-CNN model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained).to(device)
    model.eval()
    return model


def load_deit(pretrained=True, device=None):
    """
    Load DeiT model for segmentation.
    
    Args:
        pretrained (bool): Load pretrained weights
        device (str): Device to load model on
    
    Returns:
        Model: DeiT-based segmentation model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        import timm
        model = timm.create_model("deit_base_patch16_224", pretrained=pretrained, num_classes=0)
        model = model.to(device)
        model.eval()
        return model
    except ImportError:
        raise ImportError("Please install timm: pip install timm")
