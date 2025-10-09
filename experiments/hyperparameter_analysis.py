"""
Analyze the effect of different hyperparameters on attack performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.models.model_loader import load_deeplabv3
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack, PGDTargetedAttack
from src.utils.data_loader import load_coco_images
from src.utils.metrics import compute_iou


def analyze_epsilon_fgsm(model, images, epsilon_values):
    """
    Analyze the effect of epsilon on FGSM attack performance.
    
    Args:
        model: Segmentation model
        images: List of input images
        epsilon_values: List of epsilon values to test
    
    Returns:
        dict: Results with epsilon values and IoU scores
    """
    print("\n" + "="*60)
    print("Analyzing FGSM Epsilon Values")
    print("="*60)
    
    results = {"epsilon": [], "mean_iou": [], "std_iou": []}
    device = next(model.parameters()).device
    
    for eps in epsilon_values:
        print(f"\nTesting epsilon = {eps:.4f}")
        attack = FGSMAttack(model, epsilon=eps)
        adv_images = attack.generate(images)
        
        iou_scores = []
        for orig, adv in zip(images, adv_images):
            orig = orig.to(device)
            adv = adv.to(device)
            
            with torch.no_grad():
                orig_output = model(orig)["out"]
                adv_output = model(adv)["out"]
            
            orig_pred = torch.argmax(orig_output, dim=1).cpu()
            adv_pred = torch.argmax(adv_output, dim=1).cpu()
            
            iou = compute_iou(adv_pred, orig_pred)
            iou_scores.append(iou)
        
        mean_iou = np.mean(iou_scores)
        std_iou = np.std(iou_scores)
        
        results["epsilon"].append(eps)
        results["mean_iou"].append(mean_iou)
        results["std_iou"].append(std_iou)
        
        print(f"  Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    
    return results


def analyze_pgd_hyperparameters(model, images, param_grid):
    """
    Analyze the effect of PGD hyperparameters.
    
    Args:
        model: Segmentation model
        images: List of input images
        param_grid: Dictionary with lists of parameters to test
    
    Returns:
        pd.DataFrame: Results dataframe
    """
    print("\n" + "="*60)
    print("Analyzing PGD Hyperparameters")
    print("="*60)
    
    results = []
    device = next(model.parameters()).device
    
    total_experiments = (len(param_grid["epsilon"]) * 
                         len(param_grid["alpha"]) * 
                         len(param_grid["num_steps"]))
    
    experiment_num = 0
    
    for epsilon in param_grid["epsilon"]:
        for alpha in param_grid["alpha"]:
            for num_steps in param_grid["num_steps"]:
                experiment_num += 1
                print(f"\nExperiment {experiment_num}/{total_experiments}")
                print(f"epsilon={epsilon}, alpha={alpha}, steps={num_steps}")
                
                attack = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
                adv_images = attack.generate(images)
                
                iou_scores = []
                for orig, adv in zip(images, adv_images):
                    orig = orig.to(device)
                    adv = adv.to(device)
                    
                    with torch.no_grad():
                        orig_output = model(orig)["out"]
                        adv_output = model(adv)["out"]
                    
                    orig_pred = torch.argmax(orig_output, dim=1).cpu()
                    adv_pred = torch.argmax(adv_output, dim=1).cpu()
                    
                    iou = compute_iou(adv_pred, orig_pred)
                    iou_scores.append(iou)
                
                mean_iou = np.mean(iou_scores)
                std_iou = np.std(iou_scores)
                
                results.append({
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "num_steps": num_steps,
                    "mean_iou": mean_iou,
                    "std_iou": std_iou,
                    "iou_drop": 1.0 - mean_iou
                })
                
                print(f"  Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
                print(f"  IoU Drop: {1.0 - mean_iou:.4f}")
    
    return pd.DataFrame(results)


def plot_epsilon_analysis(results, save_path=None):
    """
    Plot epsilon analysis results.
    
    Args:
        results: Dictionary with epsilon and IoU results
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    epsilons = results["epsilon"]
    mean_ious = results["mean_iou"]
    std_ious = results["std_iou"]
    
    plt.errorbar(epsilons, mean_ious, yerr=std_ious, 
                 marker="o", linestyle="-", linewidth=2, markersize=8,
                 capsize=5, capthick=2, label="FGSM Attack")
    
    plt.xlabel("Epsilon (Perturbation Magnitude)", fontsize=12)
    plt.ylabel("Mean IoU Score", fontsize=12)
    plt.title("Impact of Epsilon on FGSM Attack Performance", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_pgd_heatmaps(df, save_path=None):
    """
    Plot heatmaps for PGD hyperparameter analysis.
    
    Args:
        df: DataFrame with PGD results
        save_path: Optional path to save figure
    """
    num_steps_values = sorted(df["num_steps"].unique())
    
    fig, axes = plt.subplots(1, len(num_steps_values), figsize=(18, 5))
    
    for idx, num_steps in enumerate(num_steps_values):
        subset = df[df["num_steps"] == num_steps]
        pivot = subset.pivot(index="alpha", columns="epsilon", values="mean_iou")
        
        im = axes[idx].imshow(pivot, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        axes[idx].set_title(f"Steps = {num_steps}", fontsize=12)
        axes[idx].set_xlabel("Epsilon", fontsize=11)
        axes[idx].set_ylabel("Alpha", fontsize=11)
        
        # Set ticks
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels([f"{x:.3f}" for x in pivot.columns], rotation=45)
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels([f"{y:.3f}" for y in pivot.index])
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = axes[idx].text(j, i, f"{pivot.iloc[i, j]:.2f}",
                                     ha="center", va="center", color="black", fontsize=9)
    
    # Add colorbar
    fig.colorbar(im, ax=axes, label="Mean IoU", fraction=0.046, pad=0.04)
    fig.suptitle("PGD Hyperparameter Analysis: Mean IoU Scores", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze adversarial attack hyperparameters")
    parser.add_argument("--data_path", type=str, default="extracted_images",
                        help="Path to image directory")
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of images to process")
    parser.add_argument("--analysis", type=str, choices=["fgsm", "pgd", "both"], default="both",
                        help="Type of analysis to run")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    print("Loading model...")
    model = load_deeplabv3()
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    print(f"\nLoading {args.num_images} images...")
    images = load_coco_images(args.data_path, args.num_images)
    print(f"Loaded {len(images)} images")
    
    # Run FGSM analysis
    if args.analysis in ["fgsm", "both"]:
        epsilon_values = np.linspace(0.01, 0.1, 10)
        fgsm_results = analyze_epsilon_fgsm(model, images, epsilon_values)
        
        # Save results
        df_fgsm = pd.DataFrame(fgsm_results)
        csv_path = os.path.join(args.output_dir, "fgsm_epsilon_analysis.csv")
        df_fgsm.to_csv(csv_path, index=False)
        print(f"\nFGSM results saved to {csv_path}")
        
        # Plot results
        plot_epsilon_analysis(fgsm_results, 
                             save_path=os.path.join(args.output_dir, "fgsm_epsilon_plot.png"))
    
    # Run PGD analysis
    if args.analysis in ["pgd", "both"]:
        param_grid = {
            "epsilon": [0.02, 0.04, 0.06],
            "alpha": [0.005, 0.01, 0.015],
            "num_steps": [5, 10, 20]
        }
        
        pgd_results = analyze_pgd_hyperparameters(model, images, param_grid)
        
        # Save results
        csv_path = os.path.join(args.output_dir, "pgd_hyperparameter_analysis.csv")
        pgd_results.to_csv(csv_path, index=False)
        print(f"\nPGD results saved to {csv_path}")
        
        # Plot results
        plot_pgd_heatmaps(pgd_results,
                         save_path=os.path.join(args.output_dir, "pgd_heatmaps.png"))
        
        # Print best parameters
        best_attack = pgd_results.loc[pgd_results["iou_drop"].idxmax()]
        print("\n" + "="*60)
        print("Best PGD Configuration (Highest IoU Drop):")
        print(f"  Epsilon: {best_attack['epsilon']}")
        print(f"  Alpha: {best_attack['alpha']}")
        print(f"  Steps: {int(best_attack['num_steps'])}")
        print(f"  Mean IoU: {best_attack['mean_iou']:.4f}")
        print(f"  IoU Drop: {best_attack['iou_drop']:.4f}")
        print("="*60)
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
