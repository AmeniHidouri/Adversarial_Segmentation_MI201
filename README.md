# Attaque Adversaire en Segmentation d'Images

**Auteurs**: Yassine Zanned, Seifeddine Ghozzi, Ameni Hidouri

Ce projet porte sur la conception et l'analyse d'attaques adversariales appliquées à des modèles de segmentation d'images. L'objectif principal est de comprendre la vulnérabilité des modèles de segmentation et d'évaluer leur robustesse dans différents scénarios adversariaux.

## 📋 Table des Matières

- [Objectifs du Projet](#objectifs-du-projet)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)

## 🎯 Objectifs du Projet

Les objectifs principaux sont les suivants :

- **Q1**: Développer une attaque non ciblée (untargeted) qui pousse un modèle à produire des erreurs dans ses prédictions
- **Q2**: Concevoir une attaque ciblée (targeted) pour forcer le modèle à prédire une segmentation spécifique
- **Q3**: Étudier l'impact de la taille des perturbations sur la performance de l'attaque
- **Q4**: Évaluer la robustesse d'une attaque lorsqu'elle est appliquée à un modèle différent
- **Q5**: Explorer l'effet d'entraîner une attaque sur un ensemble de réseaux

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/votre-username/adversarial-segmentation.git
cd adversarial-segmentation

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données MS-COCO
bash data/download_coco.sh
```

## 📁 Structure du Projet

```
adversarial-segmentation/
├── README.md                       # Documentation du projet
├── requirements.txt                # Dépendances Python
├── data/                          # Scripts de téléchargement de données
│   └── download_coco.sh
├── src/                           # Code source principal
│   ├── models/                    # Chargeurs de modèles
│   │   └── model_loader.py
│   ├── attacks/                   # Implémentations des attaques
│   │   ├── fgsm.py               # Fast Gradient Sign Method
│   │   └── pgd.py                # Projected Gradient Descent
│   └── utils/                     # Utilitaires
│       ├── data_loader.py        # Chargement des données
│       ├── metrics.py            # Métriques d'évaluation
│       └── visualization.py      # Visualisation des résultats
├── experiments/                   # Scripts d'expériences
│   ├── run_untargeted_attack.py
│   ├── run_targeted_attack.py
│   └── hyperparameter_analysis.py
└── notebooks/                     # Notebooks Jupyter
    └── original_notebook.ipynb
```

## 💻 Utilisation

### Attaque FGSM Non Ciblée

```python
from src.models.model_loader import load_deeplabv3
from src.attacks.fgsm import generate_fgsm_adversa
