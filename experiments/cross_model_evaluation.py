"""
Evaluate attack transferability across different segmentation models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model_loader import load_deeplabv3, load_maskrcnn
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.utils.data_loader import load_coco_images
from src.utils.metrics import compute_iou


def evaluate_transferability(source_model, target_models, images, attack_type="fgsm", **attack_params):
    """
    Evaluate attack transferability from source model to target models.
    
    Args:
        source_model: Model used to generate adversarial examples
        target_models: Dictionary of {name: model} to evaluate on
        images: Input images
        attack_type: Type of attack ("fgsm" or "pgd")
        **attack_params: Parameters for the attack
    
    Returns:
        pd.DataFrame: Transferability results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {attack_type.upper()} Attack Transferability")
    print(f"{'='*60}")
    
    # Generate adversarial examples on source model
    print("\nGenerating adversarial examples on source model...")
    if attack_type == "fgsm":
        attack = FGSMAttack(source_model, **attack_params)
    elif attack_type == "pgd":
        attack = PGDAttack(source_model, **attack_params)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    adv_images = attack.generate(images)
    print("Adversarial examples generated!")
    
    # Evaluate on all target models
    results = []
    device = next(source_model.parameters()).device
    
    for model_name, target_model in target_models.items():
        print(f"\nEvaluating on {model_name}...")
        target_model = target_model.to(device)
        target_model.eval()
        
        iou_scores = []
        for orig, adv in zip(images, adv_images):
            orig =
