"""
Projected Gradient Descent (PGD) attack implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    Projected Gradient Descent (PGD) untargeted attack.
    """
    
    def __init__(self, model, epsilon=0.02, alpha=0.005, num_steps=10, device=None):
        """
        Initialize PGD attack.
        
        Args:
            model: Target segmentation model
            epsilon (float): Maximum perturbation
            alpha (float): Step size
            num_steps (int): Number of iterations
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate(self, images):
        """
        Generate adversarial examples using PGD.
        
        Args:
            images (list): List of input image tensors
        
        Returns:
            list: Adversarial images
        """
        adv_images = []
        
        for image in images:
            image = image.to(self.device)
            
            # Get target (original prediction)
            with torch.no_grad():
                output = self.model(image)["out"]
            target = torch.argmax(output, dim=1)
            
            # Generate adversarial image
            adv_image = self._pgd_attack(image, target)
            adv_images.append(adv_image.detach())
            
            # Free memory
            del image, output, target
            torch.cuda.empty_cache()
        
        return adv_images
    
    def _pgd_attack(self, image, target):
        """
        Perform PGD attack.
        
        Args:
            image: Input image
            target: Target segmentation map
        
        Returns:
            torch.Tensor: Adversarial image
        """
        adv_image = image.clone().detach()
        adv_image.requires_grad = True
        
        for _ in range(self.num_steps):
            output = self.model(adv_image)["out"]
            loss = F.cross_entropy(output, target)
            
            self.model.zero_grad()
            loss.backward()
            
            # Take step in direction of gradient
            adv_image = adv_image + self.alpha * adv_image.grad.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(adv_image - image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(image + perturbation, 0, 1).detach()
            adv_image.requires_grad = True
        
        return adv_image


class PGDTargetedAttack(PGDAttack):
    """
    Targeted PGD attack for segmentation.
    """
    
    def __init__(self, model, target_class, epsilon=0.02, alpha=0.005, num_steps=10, device=None):
        """
        Initialize targeted PGD attack.
        
        Args:
            model: Target segmentation model
            target_class (int): Target class for attack
            epsilon (float): Maximum perturbation
            alpha (float): Step size
            num_steps (int): Number of iterations
            device: Device to run attack on
        """
        super().__init__(model, epsilon, alpha, num_steps, device)
        self.target_class = target_class
    
    def _pgd_attack(self, image, _):
        """
        Perform targeted PGD attack.
        
        Args:
            image: Input image
            _: Ignored (target is predetermined)
        
        Returns:
            torch.Tensor: Adversarial image
        """
        adv_image = image.clone().detach()
        adv_image.requires_grad = True
        
        # Create target label (force all pixels to target class)
        _, _, h, w = image.shape
        target_label = torch.full((1, h, w), self.target_class, dtype=torch.long).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.num_steps):
            self.model.zero_grad()
            output = self.model(adv_image)["out"]
            
            # Minimize loss to target class (targeted attack)
            loss = -criterion(output, target_label)
            loss.backward()
            
            # Apply perturbation
            adv_image.data = adv_image.data - self.alpha * adv_image.grad.data.sign()
            
            # Project to epsilon ball
            perturbation = torch.clamp(adv_image - image, -self.epsilon, self.epsilon)
            adv_image.data = torch.clamp(image + perturbation, 0, 1)
            
            adv_image.grad.zero_()
        
        return adv_image
