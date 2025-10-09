"""
Run untargeted adversarial attacks (FGSM and PGD) on segmentation models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from src.models.model_loader import load_deeplabv3
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.utils.data_loader import load_coco_images
from src.utils.visualization import show_images, show_segmentation_comparison
from src.utils.metrics import compute_iou


def run_fgsm_attack(model, images, epsilon=0.01):
    """
    Run FGSM untargeted attack.
    
    Args:
        model: Segmentation model
        images: List of input images
        epsilon: Perturbation magnitude
    
    Returns:
        list: Adversarial images
    """
    print(f"\n{'='*60}")
    print(f"Running FGSM Attack with epsilon={epsilon}")
    print(f"{'='*60}")
    
    attack = FGSMAttack(model, epsilon=epsilon)
    adv_images = attack.generate(images)
    
    # Compute statistics
    device = next(model.parameters()).device
    iou_scores = []
    
    for i, (orig, adv) in enumerate(zip(images, adv_images)):
        orig = orig.to(device)
        adv = adv.to(device)
        
        with torch.no_grad():
            orig_output = model(orig)["out"]
            adv_output = model(adv)["out"]
        
        orig_pred = torch.argmax(orig_output, dim=1).cpu()
        adv_pred = torch.argmax(adv_output, dim=1).cpu()
        
        iou = compute_iou(adv_pred, orig_pred)
        iou_scores.append(iou)
        print(f"Image {i+1}: IoU = {iou:.4f}")
    
    avg_iou = sum(iou_scores) / len(iou_scores)
    print(f"\nAverage IoU: {avg_iou:.4f}")
    print(f"Average IoU Drop: {1.0 - avg_iou:.4f}")
    
    return adv_images


def run_pgd_attack(model, images, epsilon=0.02, alpha=0.005, num_steps=10):
    """
    Run PGD untargeted attack.
    
    Args:
        model: Segmentation model
        images: List of input images
        epsilon: Maximum perturbation
        alpha: Step size
        num_steps: Number of iterations
    
    Returns:
        list: Adversarial images
    """
    print(f"\n{'='*60}")
    print(f"Running PGD Attack")
    print(f"epsilon={epsilon}, alpha={alpha}, steps={num_steps}")
    print(f"{'='*60}")
    
    attack = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
    adv_images = attack.generate(images)
    
    # Compute statistics
    device = next(model.parameters()).device
    iou_scores = []
    
    for i, (orig, adv) in enumerate(zip(images, adv_images)):
        orig = orig.to(device)
        adv = adv.to(device)
        
        with torch.no_grad():
            orig_output = model(orig)["out"]
            adv_output = model(adv)["out"]
        
        orig_pred = torch.argmax(orig_output, dim=1).cpu()
        adv_pred = torch.argmax(adv_output, dim=1).cpu()
        
        iou = compute_iou(adv_pred, orig_pred)
        iou_scores.append(iou)
        print(f"Image {i+1}: IoU = {iou:.4f}")
    
    avg_iou = sum(iou_scores) / len(iou_scores)
    print(f"\nAverage IoU: {avg_iou:.4f}")
    print(f"Average IoU Drop: {1.0 - avg_iou:.4f}")
    
    return adv_images


def main():
    parser = argparse.ArgumentParser(description="Run untargeted adversarial attacks")
    parser.add_argument("--data_path", type=str, default="extracted_images",
                        help="Path to image directory")
    parser.add_argument("--num_images", type=int, default=9,
                        help="Number of images to process")
    parser.add_argument("--attack", type=str, choices=["fgsm", "pgd", "both"], default="both",
                        help="Attack type to run")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Perturbation magnitude for FGSM")
    parser.add_argument("--pgd_epsilon", type=float, default=0.02,
                        help="Maximum perturbation for PGD")
    parser.add_argument("--alpha", type=float, default=0.005,
                        help="Step size for PGD")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of PGD iterations")
    parser.add_argument("--visualize", action="store_true",
                        help="Show visualizations")
    
    args = parser.parse_args()
    
    # Setup
    print("Loading model...")
    model = load_deeplabv3()
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    print(f"\nLoading {args.num_images} images from {args.data_path}...")
    images = load_coco_images(args.data_path, args.num_images)
    print(f"Loaded {len(images)} images")
    
    # Run attacks
    if args.attack in ["fgsm", "both"]:
        fgsm_adv_images = run_fgsm_attack(model, images, args.epsilon)
        
        if args.visualize:
            print("\nVisualizing FGSM results...")
            show_images(images, fgsm_adv_images, len(images))
            show_segmentation_comparison(images, fgsm_adv_images, model, len(images))
    
    if args.attack in ["pgd", "both"]:
        pgd_adv_images = run_pgd_attack(model, images, args.pgd_epsilon, args.alpha, args.steps)
        
        if args.visualize:
            print("\nVisualizing PGD results...")
            show_images(images, pgd_adv_images, len(images))
            show_segmentation_comparison(images, pgd_adv_images, model, len(images))
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
