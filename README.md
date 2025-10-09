# 🎯 Attaques Adversariales en Segmentation d'Images

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Auteurs**: Yassine Zanned, Seifeddine Ghozzi, Ameni Hidouri

Ce projet porte sur la conception et l'analyse d'attaques adversariales appliquées à des modèles de segmentation d'images. Une attaque adversariale consiste à introduire de légères perturbations dans une image, souvent invisibles pour un observateur humain, afin de tromper un modèle d'apprentissage automatique. L'objectif principal est de comprendre la vulnérabilité des modèles de segmentation et d'évaluer leur robustesse dans différents scénarios adversariaux.

![Project Banner](https://img.shields.io/badge/Deep_Learning-Adversarial_Attacks-red)
![Segmentation](https://img.shields.io/badge/Computer_Vision-Segmentation-green)

---

## 📋 Table des Matières

- [🎯 Objectifs du Projet](#-objectifs-du-projet)
- [🚀 Installation](#-installation)
- [📁 Structure du Projet](#-structure-du-projet)
- [💻 Utilisation](#-utilisation)
  - [Quick Start](#quick-start)
  - [Attaque FGSM Non Ciblée](#attaque-fgsm-non-ciblée)
  - [Attaque PGD Ciblée](#attaque-pgd-ciblée)
  - [Analyse des Hyperparamètres](#analyse-des-hyperparamètres)
  - [Évaluation Cross-Model](#évaluation-cross-model)
- [🔬 Méthodologie](#-méthodologie)
  - [Données Utilisées](#données-utilisées)
  - [Modèles de Segmentation](#modèles-de-segmentation)
  - [Attaques Implémentées](#attaques-implémentées)
  - [Métriques d'Évaluation](#métriques-dévaluation)
- [📊 Résultats](#-résultats)
  - [Impact de l'Epsilon (FGSM)](#impact-de-lepsilon-fgsm)
  - [Influence des Hyperparamètres PGD](#influence-des-hyperparamètres-pgd)
  - [Comparaison des Modèles](#comparaison-des-modèles)
- [🎓 Conclusion](#-conclusion)
- [📚 Références](#-références)
- [📝 License](#-license)
- [👥 Contributeurs](#-contributeurs)

---

## 🎯 Objectifs du Projet

Les objectifs principaux de ce projet sont les suivants :

| Question | Objectif | Méthode |
|----------|----------|---------|
| **Q1** | Développer une attaque non ciblée (untargeted) | FGSM pour produire des erreurs de prédiction |
| **Q2** | Concevoir une attaque ciblée (targeted) | PGD pour forcer une segmentation spécifique |
| **Q3** | Étudier l'impact de la taille des perturbations | Analyse de l'IoU pour différents ε |
| **Q4** | Évaluer la robustesse cross-model | Transférabilité des attaques |
| **Q5** | Explorer l'entraînement multi-réseaux | Attaques sur ensemble de modèles |

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- CUDA capable GPU (recommandé)
- 4 GB RAM minimum
- 2 GB d'espace disque

### Installation Rapide
```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/adversarial-segmentation.git
cd adversarial-segmentation

# 2. Créer un environnement virtuel
python -m venv venv

# Sur Linux/Mac
source venv/bin/activate

# Sur Windows
venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger les données MS-COCO
bash data/download_coco.sh
2. **Ensemble de modèles** 🤝
   - Combiner CNN + Transformer
   - Utiliser le vote majoritaire
   - Exploiter des architectures complémentaires

3. **Détection d'anomalies** 🔍
   - Monitorer les prédictions suspectes
   - Analyser la distribution des features
   - Rejeter les entrées avec forte incertitude

4. **Preprocessing défensif** 🎨
   - Compression JPEG
   - Débruitage
   - Quantification des pixels

### Limitations du Projet

- 📊 Dataset limité (50 images MS-COCO)
- 🔧 Uniquement 3 modèles testés
- ⚡ Attaques en boîte blanche (accès aux gradients)
- 🎯 Focus sur la segmentation sémantique

### Perspectives Futures

1. **Extension des attaques** :
   - C&W (Carlini & Wagner)
   - AutoAttack
   - Attaques en boîte noire

2. **Plus de modèles** :
   - SegFormer
   - Swin Transformer
   - SAM (Segment Anything Model)

3. **Défenses avancées** :
   - Certified defenses
   - Randomized smoothing
   - Neural network verification

4. **Applications réelles** :
   - Segmentation médicale
   - Conduite autonome
   - Surveillance

---

## 📚 Références

### Articles Fondamentaux

1. **FGSM** :
   - Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv:1412.6572
   - [📄 Paper](https://arxiv.org/abs/1412.6572)

2. **PGD** :
   - Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). *Towards deep learning models resistant to adversarial attacks*. arXiv:1706.06083
   - [📄 Paper](https://arxiv.org/abs/1706.06083)

3. **DeepLabV3** :
   - Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). *Rethinking atrous convolution for semantic image segmentation*. arXiv:1706.05587
   - [📄 Paper](https://arxiv.org/abs/1706.05587)

4. **Mask R-CNN** :
   - He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). *Mask R-CNN*. ICCV 2017
   - [📄 Paper](https://arxiv.org/abs/1703.06870)

5. **DeiT** :
   - Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). *Training data-efficient image transformers*. ICML 2021
   - [📄 Paper](https://arxiv.org/abs/2012.12877)

### Ressources Utiles

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [MS-COCO Dataset](https://cocodataset.org/)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [CleverHans](https://github.com/cleverhans-lab/cleverhans)

---

## 🔧 Commandes Utiles

### Exécution Rapide
```bash
