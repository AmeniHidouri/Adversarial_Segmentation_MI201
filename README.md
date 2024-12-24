# Adversarial_Segmentation_MI201
Projet de machine learning pour tester des attaques adversaires en segmentation
# Adversarial Segmentation

Ce projet explore les **attaques adversaires** dans le domaine de la segmentation d'images. Le but est d'ajouter des perturbations "invisibles" à des images pour tromper les modèles de segmentation pré-entraînés.

## Fonctionnalités
- Implémentation d'attaques adversaires :
  - **Untargeted** : Perturber pour fausser la prédiction, sans cible spécifique.
  - **Targeted** : Forcer la prédiction vers une classe ou un résultat spécifique.
- Modèles de segmentation pré-entraînés de `torchvision` (par ex. `fcn_resnet50`, `deeplabv3_resnet50`).
- Analyse de la robustesse des modèles face aux perturbations adversaires.
