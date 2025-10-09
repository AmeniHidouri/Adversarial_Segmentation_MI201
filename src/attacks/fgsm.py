"""
Fast Gradient Sign Method (FGSM) attack implementation.
"""

import torch
import torch.nn as nn


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) attack for segmentation models.
    
    Formula: x' = x + ε · sign(∇_x J(θ, x, y))
    """
    
    def __init__(self, model, epsilon=0.01, device=None):
        """
        Initialize FGSM attack.
        
        Args:
            model: Target segmentation model
            epsilon (float): Perturbation magnitude
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
    
    def generate(self, images):
        """
        Generate adversarial examples using FGSM.
        
        Args:
            images (torch.Tensor): Input images
        
        Returns:
            list: Adversarial images
        """
        adv_images = []
        
        for image in images:
            image = image.to(self.device)
            image.requires_grad = True
            
            # Forward pass
            output = self.model(image)["out"]
            target = torch.argmax(output, dim=1)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Generate adversarial image
            gradient = image.grad.data
            adv_image = self._fgsm_step(image, gradient)
            adv_images.append(adv_image.detach())
            
            # Free memory
            del image, output, loss, gradient
            torch.cuda.empty_cache()
        
        return adv_images
    
    def _fgsm_step(self, image, gradient):
        """
        Perform one FGSM step.
        
        Args:
            image: Input image tensor
            gradient: Gradient of loss w.r.t. image
        
        Returns:
            torch.Tensor: Perturbed image
        """
        sign_data_grad = gradient.sign()
        perturbed_image = image + self.epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
