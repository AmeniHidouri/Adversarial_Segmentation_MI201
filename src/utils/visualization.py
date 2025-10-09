"""
Visualization utilities for adversarial attacks.
"""

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np


def show_images(originals, adversarials, num=9):
    """
    Display original and adversarial images side by side.
    
    Args:
        originals: List of original image tensors
        adversarials: List of adversarial image tensors
        num: Number of images to display
    """
    fig, axs = plt.subplots(num, 2, figsize=(10, 5 * num))
    to_pil = transforms.ToPILImage()
    
    for i in range(min(num, len(originals))):
        orig_img = to_pil(originals[i].squeeze().cpu())
        adv_img = to_pil(adversarials[i].squeeze().cpu())
        
        axs[i, 0].imshow(orig_img)
        axs[i, 0].set_title(f"Original Image {i+1}")
        axs[i, 0].axis("off")
        
        axs[i, 1].imshow(adv_img)
        axs[i, 1].set_title(f"Adversarial Image {i+1}")
        axs[i, 1].axis("off")
    
    plt.tight_layout()
    plt.show()


def show_segmentation_comparison(originals, adversarials, model, num=9):
    """
    Display segmentation comparison between original and adversarial images.
    
    Args:
        originals: List of original image tensors
        adversarials: List of adversarial image tensors
        model: Segmentation model
        num: Number of images to display
    """
    from .metrics import compute_iou
    
    fig, axs = plt.subplots(num, 3, figsize=(15, 5 * num))
    device = next(model.parameters()).device
    
    for i in range(min(num, len(originals))):
        orig_tensor = originals[i].to(device)
        adv_tensor = adversarials[i].to(device)
        
        with torch.no_grad():
            orig_output = model(orig_tensor)["out"]
            adv_output = model(adv_tensor)["out"]
        
        orig_pred = torch.argmax(orig_output, dim=1).cpu().squeeze().numpy()
        adv_pred = torch.argmax(adv_output, dim=1).cpu().squeeze().numpy()
        
        iou_before = compute_iou(orig_pred, orig_pred)
        iou_after = compute_iou(adv_pred, orig_pred)
        
        axs[i, 0].imshow(orig_pred, cmap="viridis")
        axs[i, 0].set_title(f"Original Segmentation {i+1}\nIoU: {iou_before:.4f}")
        axs[i, 0].axis("off")
        
        axs[i, 1].imshow(adv_pred, cmap="viridis")
        axs[i, 1].set_title(f"Adversarial Segmentation {i+1}\nIoU: {iou_after:.4f}")
        axs[i, 1].axis("off")
        
        diff_map = (orig_pred != adv_pred)
        axs[i, 2].imshow(diff_map, cmap="gray")
        axs[i, 2].set_title(f"Difference Map {i+1}")
        axs[i, 2].axis("off")
        
        print(f"Image {i+1}: IoU Drop = {iou_before - iou_after:.4f}")
    
    plt.tight_layout()
    plt.show()


def plot_epsilon_vs_iou(epsilons, iou_scores):
    """
    Plot IoU scores vs epsilon values.
    
    Args:
        epsilons: List of epsilon values
        iou_scores: Corresponding IoU scores
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, iou_scores, marker="o", linestyle="-", linewidth=2)
    plt.xlabel("Epsilon (Attack Strength)", fontsize=12)
    plt.ylabel("IoU Score", fontsize=12)
    plt.title("IoU Drop vs. Attack Strength", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
