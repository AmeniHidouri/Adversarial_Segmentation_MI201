
# 🎯 Attaques Adversariales en Segmentation d'Images

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Auteurs**: Yassine Zanned, Seifeddine Ghozzi, Ameni Hidouri

## 📋 Table des Matières
- [🎯 Objectifs](#-objectifs)
- [🚀 Installation](#-installation)
- [📁 Structure](#-structure-du-projet)
- [💻 Utilisation](#-utilisation)
- [🔬 Méthodologie](#-méthodologie)
- [📊 Résultats](#-résultats)
- [📚 Références](#-références)

## 🎯 Objectifs

| Question | Objectif | Méthode |
|----------|----------|---------|
| **Q1** | Attaque non ciblée | FGSM |
| **Q2** | Attaque ciblée | PGD |
| **Q3** | Impact des perturbations | Analyse IoU |
| **Q4** | Robustesse cross-model | Transferabilité |
| **Q5** | Défenses | Entraînement adversarial |

## 🚀 Installation

```bash
git clone https://github.com/votre-username/adversarial-segmentation.git
cd adversarial-segmentation

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

pip install -r requirements.txt
bash data/download_coco.sh
```

## 📁 Structure du Projet
```
adversarial-segmentation/
├── data/
├── models/
├── attacks/
├── experiments/
├── utils/
├── requirements.txt
└── README.md
```

## 💻 Utilisation

### Quick Start
```bash
python experiments/run_untargeted_attack.py --attack both --num_images 5
```

### Attaques
```bash
# FGSM non ciblée
python experiments/run_untargeted_attack.py --attack fgsm --epsilon 0.03

# PGD ciblée
python experiments/run_targeted_attack.py --target_class 1 --steps 40
```

### Analyse
```bash
python experiments/hyperparameter_analysis.py --analysis both
python experiments/cross_model_evaluation.py --num_images 5
```

## 🔬 Méthodologie

### Modèles
- **DeepLabV3+**
- **Mask R-CNN**
- **DeiT**

### Attaques
- **FGSM**: `x' = x + ε * sign(∇ₓJ(θ,x,y))`
- **PGD**: `xₜ₊₁ = Πₓ₊ₛ(xₜ + α * sign(∇ₓJ(θ,x,y)))`

### Métriques
- mIoU
- Accuracy
- Success Rate

## 📊 Résultats

### Impact de ε (FGSM)
| ε | mIoU Orig | mIoU Adv | Drop |
|---|-----------|----------|------|
| 0.01 | 0.75 | 0.68 | -9% |
| 0.03 | 0.75 | 0.52 | -31% |
| 0.05 | 0.75 | 0.35 | -53% |

### PGD (40 steps)
| α | Success | PSNR |
|---|---------|------|
| 0.005 | 45% | 32.1 |
| 0.01 | 78% | 28.4 |
| 0.02 | 92% | 24.7 |

## 📚 Références
1. Goodfellow et al. - FGSM (2014)
2. Madry et al. - PGD (2017)
3. Chen et al. - DeepLabV3 (2017)
4. He et al. - Mask R-CNN (2017)

## 👥 Contributeurs
- Yassine Zanned
- Seifeddine Ghozzi  
- Ameni Hidouri

## 📝 License
MIT License - voir [LICENSE](LICENSE) pour détails.

