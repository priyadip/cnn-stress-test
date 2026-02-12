# Stress-Testing of Convolutional Neural Networks

**Robustness, Interpretability, and Failure Analysis of a Modified ResNet-18 on CIFAR-10**

> Deep Learning (DL) — Assignment 1  
> **Authors:** Priyadip Sau (M25CSA023), Saumya Pancholi (M25CSA027), Suparni Maitra (M25CSA029), Eshani Sawant (M25CSE012)

---

## Project Overview

This repository implements a complete stress-testing pipeline for a modified ResNet-18 trained from scratch on CIFAR-10. The focus is not merely on achieving high accuracy, but on understanding **where** and **why** the model fails, using Grad-CAM for visual explanations, and evaluating a single constrained improvement (Cutout regularization).

| Requirement | Implementation |
|---|---|
| Dataset | CIFAR-10 (official train/test split, 50K/10K) |
| Architecture | Modified ResNet-18 (adapted for 32×32 input) |
| Training | From scratch — no pretrained weights |
| Epochs | 50 |
| Random Seed | **42** (fixed for full reproducibility) |
| Failure Analysis | 398 high-confidence failures identified (threshold > 70%) |
| Explainability | Grad-CAM at `layer3` (8×8 spatial resolution) |
| Constrained Improvement | Cutout regularization (1 hole, 16×16) |
| Framework | PyTorch (exclusively) |

### Key Results

| Model | Test Accuracy | High-Confidence Failures |
|---|---|---|
| Baseline | **93.98%** | 398 |
| + Cutout | **94.32%** | 320 |

---

## Project Structure

```
cnn-stress-test/
├── config.py                 # Centralised hyperparameters and paths
├── main.py                   # Unified entry point for all pipeline modes
├── train.py                  # Training loop (baseline + Cutout)
├── evaluate.py               # Evaluation, failure discovery, confusion matrix
├── visualize.py              # Grad-CAM visualizations and comparison figures
├── analyze_disagreements.py  # Disagreement analysis between baseline and Cutout
├── run_stress_test.sh        # SLURM batch script for HPC execution
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── models/
│   ├── __init__.py
│   └── resnet.py             # Modified ResNet-18 for CIFAR-10 (32×32)
│
├── utils/
│   ├── __init__.py
│   ├── cutout.py             # Cutout augmentation implementation
│   ├── gradcam.py            # Grad-CAM implementation with hooks
│   └── data.py               # Data loading, transforms, denormalization
│
├── checkpoints/              # Saved model weights (generated during training)
├── results/                  # Evaluation JSONs and failure analysis
├── figures/                  # All generated plots and visualizations
└── data/                     # CIFAR-10 dataset (auto-downloaded)
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~2 GB disk space for dataset and checkpoints

### Setup

```bash
# Clone or navigate to the project directory
cd cnn-stress-test

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate       # Linux / Mac
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
Pillow>=9.5.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
pandas>=2.0.0
```

---

## Reproducing Results

### Option A: Run the Complete Pipeline

```bash
python main.py --mode all
```

This executes all phases sequentially: baseline training → evaluation → Grad-CAM → Cutout training → Cutout evaluation → comparison. Estimated time: ~2–3 hours on a single GPU.

### Option B: Step-by-Step Execution

```bash
# Phase 1: Train baseline model
python main.py --mode baseline

# Phase 2: Evaluate baseline and discover failure cases
python main.py --mode evaluate

# Phase 3: Generate Grad-CAM visualizations for baseline
python main.py --mode visualize --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth

# Phase 4: Train the Cutout-enhanced model
python main.py --mode cutout

# Phase 5: Evaluate Cutout model
python main.py --mode evaluate --use-cutout

# Phase 6: Generate comparison visualizations (Grad-CAM side-by-side)
python main.py --mode visualize \
    --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth \
    --cutout checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth

# Phase 7: Generate final comparative summary
python main.py --mode compare \
    --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth \
    --cutout checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth
```

### Option C: Using Individual Scripts

```bash
# Train baseline
python train.py

# Train with Cutout augmentation
python train.py --cutout

# Evaluate a specific checkpoint
python evaluate.py --checkpoint checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth

# Evaluate the Cutout model
python evaluate.py --checkpoint checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth --cutout

# Generate Grad-CAM for a single model
python visualize.py --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth

# Compare both models with Grad-CAM
python visualize.py \
    --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth \
    --cutout checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth

# Generate disagreement analysis
python analyze_disagreements.py
```

### Option D: HPC / SLURM Execution

```bash
sbatch run_stress_test.sh
```

The SLURM script (`run_stress_test.sh`) runs all seven phases sequentially on a GPU node. Edit the `USERNAME` variable and paths at the top of the script to match your environment.

---

## Experimental Configuration

All hyperparameters are centralised in `config.py` for easy modification.

### Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Batch Size | 128 | Standard for CIFAR-10 with ResNet |
| Epochs | 50 | Assignment constraint |
| Initial LR | 0.1 | Established default for SGD + ResNet |
| Momentum | 0.9 | Nesterov momentum for faster convergence |
| Weight Decay | 5×10⁻⁴ | L2 regularization |
| LR Schedule | MultiStepLR at [25, 40] | Decay by 0.1× at each milestone |
| Optimizer | SGD (Nesterov) | Gold standard for ResNet training |

### Cutout Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Number of Holes | 1 | Standard setting per the original paper |
| Hole Size | 16×16 | Half the image width (32/2) |

### Failure Analysis

| Parameter | Value |
|---|---|
| Confidence Threshold | 0.7 (70%) |
| Grad-CAM Target Layer | `layer3` (8×8 spatial maps) |

---

## Generated Outputs

After a full pipeline run, the following key files are produced:

```
figures/
├── ResNet-18-CIFAR_baseline_seed42_training_curves.png
├── ResNet-18-CIFAR_cutout_seed42_training_curves.png
├── ResNet-18-CIFAR_baseline_seed42_confusion_matrix.png
├── ResNet-18-CIFAR_cutout_seed42_confusion_matrix.png
├── final_comparison_summary.png
├── gradcam_baseline/
│   └── baseline_failure_*.png
├── gradcam_cutout/
│   └── cutout_failure_*.png
└── comparison/
    ├── comparison_case_1.png
    ├── comparison_case_2.png
    └── comparison_case_3.png

results/
├── ResNet-18-CIFAR_baseline_seed42_evaluation.json
├── ResNet-18-CIFAR_cutout_seed42_evaluation.json
├── ResNet-18-CIFAR_baseline_seed42_history.json
├── ResNet-18-CIFAR_cutout_seed42_history.json
└── ResNet-18-CIFAR_baseline_seed42_failures/
    ├── failure_analysis.json
    └── failure_cases_grid.png

checkpoints/
├── ResNet-18-CIFAR_baseline_seed42_final.pth
├── ResNet-18-CIFAR_baseline_seed42_best.pth
├── ResNet-18-CIFAR_cutout_seed42_final.pth
└── ResNet-18-CIFAR_cutout_seed42_best.pth
```

---

## Architecture Details

**Modified ResNet-18 for CIFAR-10 (32×32 input)**

Key differences from the standard ImageNet ResNet-18:

1. **Stem convolution:** 7×7 stride-2 → **3×3 stride-1** (preserves 32×32 spatial resolution)
2. **Max pooling:** **Removed** (prevents over-aggressive downsampling at low resolution)
3. **Feature map progression:** 32×32×64 → 16×16×128 → 8×8×256 → 4×4×512
4. **Total parameters:** ~11.2M
5. **Weight initialization:** Kaiming He (fan-out, ReLU)

The 4×4 final feature maps are sufficient for Grad-CAM interpretability; we target `layer3` (8×8) for finer spatial resolution in our visualizations.

---

## Reproducibility

**Random Seed: 42** — set for PyTorch, NumPy, Python's `random` module, CUDA RNG, and cuDNN deterministic mode.

```python
import config
config.set_seed(42)  # Must be called before any stochastic operations
```

All results reported in the accompanying PDF report are fully reproducible by running the pipeline as described above with the specified seed.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| CUDA out of memory | Reduce `BATCH_SIZE` in `config.py` to 64 or 32 |
| Slow training on CPU | Expect ~8–10 hours; consider Google Colab or reducing `NUM_EPOCHS` for testing |
| OpenCV import error | `pip install opencv-python-headless` |
| Missing `models/` or `utils/` imports | Ensure you run scripts from the project root directory |

---

## References

1. K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
2. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017.
3. T. DeVries and G. W. Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout," arXiv:1708.04552, 2017.
4. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, 2009.

---

## License

This code is provided for educational purposes as part of the Deep Learning course assignment.
