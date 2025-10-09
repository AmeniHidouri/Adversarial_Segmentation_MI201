"""
Run targeted adversarial attacks on segmentation models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from src.models.model_loader import load_deeplabv3
from src.attacks.pgd import PGDTargetedAttack
from src.utils.data_loader import load_coco_images
from src.utils.visualization import show_images, show_segmentation_comparison
from src.utils.metrics import compute_iou
import matplotlib.pyplot as plt


# MS-COCO class labels
COCO_CLASSES = {
    0: "background",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow"
}


def run_targeted_attack(model, images, target_class, epsilon=0.02, alpha=0.005, num_steps=10):
    """
    Run targeted PGD attack.
    
    Args:
        model: Segmentation model
        images: List of input images
        target_class: Target class to force predictions to
        epsilon: Maximum perturbation
        alpha: Step size
        num_steps: Number of iterations
    
    Returns:
        list: Adversarial images
    """
    target_name = COCO_CLASSES.get(target_class, f"Class {target_class}")
    
    print(f"\n{'='*60}")
    print(f"Running Targeted PGD Attack")
    print(f"Target Class: {target_class} ({target_name})")
    print(f"epsilon={epsilon}, alpha={alpha}, steps={num_steps}")
    print(f"{'='*60}")
    
    attack = PGDTargetedAttack(
        model, 
        target_class=target_class,
        epsilon=epsilon, 
        alpha=alpha, 
        num_steps=num_steps
    )
    adv_images = attack.generate(images)
    
    # Compute statistics
    device = next(model.parameters()).device
    iou_scores = []
    target_pixel_ratios = []
    
    for i, (orig, adv) in enumerate(zip(images, adv_images)):
        orig = orig.to(device)
        adv = adv.to(device)
        
        with torch.no_grad():
            orig_output = model(orig)["out"]
            adv_output = model(adv)["out"]
        
        orig_pred = torch.argmax(orig_output, dim=1).cpu().numpy()
        adv_pred = torch.argmax(adv_output, dim=1).cpu().numpy()
        
        # Compute IoU
        iou = compute_iou(adv_pred, orig_pred)
        iou_scores.append(iou)
        
        # Compute target class ratio
        target_ratio = (adv_pred == target_class).sum() / adv_pred.size
        target_pixel_ratios.append(target_ratio)
        
        print(f"Image {i+1}:")
        print(f"  IoU = {iou:.4f}")
        print(f"  Target class pixels: {target_ratio*100:.2f}%")
    
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_target_ratio = sum(target_pixel_ratios) / len(target_pixel_ratios)
    
    print(f"\n{'='*60}")
    print(f"Attack Results:")
    print(f"  Average IoU: {avg_iou:.4f}")
    print(f"  Average IoU Drop: {1.0 - avg_iou:.4f}")
    print(f"  Average Target Class Coverage: {avg_target_ratio*100:.2f}%")
    print(f"{'='*60}")
    
    return adv_images


def visualize_targeted_results(model, images, adv_images, target_class):
    """
    Visualize targeted attack results with detailed analysis.
    
    Args:
        model: Segmentation model
        images: Original images
        adv_images: Adversarial images
        target_class: Target class
    """
    device = next(model.parameters()).device
    num_images = len(images)
    
    fig, axs = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))
    
    for i in range(num_images):
        orig = images[i].to(device)
        adv = adv_images[i].to(device)
        
        with torch.no_grad():
            orig_output = model(orig)["out"]
            adv_output = model(adv)["out"]
        
        orig_pred = torch.argmax(orig_output, dim=1).cpu().squeeze().numpy()
        adv_pred = torch.argmax(adv_output, dim=1).cpu().squeeze().numpy()
        
        # Create target class mask
        target_mask = (adv_pred == target_class).astype(float)
        
        # Plot original segmentation
        axs[i, 0].imshow(orig_pred, cmap="viridis")
        axs[i, 0].set_title(f"Original Segmentation {i+1}")
        axs[i, 0].axis("off")
        
        # Plot adversarial segmentation
        axs[i, 1].imshow(adv_pred, cmap="viridis")
        axs[i, 1].set_title(f"Adversarial Segmentation {i+1}")
        axs[i, 1].axis("off")
        
        # Plot target class mask
        axs[i, 2].imshow(target_mask, cmap="Reds", vmin=0, vmax=1)
        target_ratio = target_mask.sum() / target_mask.size
        axs[i, 2].set_title(f"Target Class Mask ({target_ratio*100:.1f}%)")
        axs[i, 2].axis("off")
        
        # Plot difference map
        diff_map = (orig_pred != adv_pred).astype(float)
        axs[i, 3].imshow(diff_map, cmap="gray")
        axs[i, 3].set_title(f"Difference Map {i+1}")
        axs[i, 3].axis("off")
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run targeted adversarial attacks")
    parser.add_argument("--data_path", type=str, default="extracted_images",
                        help="Path to image directory")
    parser.add_argument("--num_images", type=int, default=9,
                        help="Number of images to process")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for attack (default: 1 = person)")
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="Maximum perturbation")
    parser.add_argument("--alpha", type=float, default=0.005,
                        help="Step size")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of PGD iterations")
    parser.add_argument("--visualize", action="store_true",
                        help="Show visualizations")
    parser.add_argument("--list_classes", action="store_true",
                        help="List available target classes")
    
    args = parser.parse_args()
    
    # List classes if requested
    if args.list_classes:
        print("\nAvailable MS-COCO classes:")
        print("="*60)
        for class_id, class_name in sorted(COCO_CLASSES.items()):
            print(f"  {class_id:2d}: {class_name}")
        print("="*60)
        return
    
    # Setup
    print("Loading model...")
    model = load_deeplabv3()
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    print(f"\nLoading {args.num_images} images from {args.data_path}...")
    images = load_coco_images(args.data_path, args.num_images)
    print(f"Loaded {len(images)} images")
    
    # Run targeted attack
    adv_images = run_targeted_attack(
        model, 
        images, 
        args.target_class,
        args.epsilon,
        args.alpha,
        args.steps
    )
    
    if args.visualize:
        print("\nVisualizing targeted attack results...")
        visualize_targeted_results(model, images, adv_images, args.target_class)
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
