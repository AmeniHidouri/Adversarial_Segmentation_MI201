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
