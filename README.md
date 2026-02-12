# CNN Stress-Testing Assignment

## Systematic Stress-Testing of Convolutional Architectures: Robustness, Interpretability, and Failure Analysis

This repository contains a complete implementation for the Deep Learning assignment on stress-testing Convolutional Neural Networks. The codebase implements a modified ResNet-18 architecture trained on CIFAR-10, with comprehensive failure analysis using Grad-CAM and Cutout regularization as a constrained improvement.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Experimental Configuration](#experimental-configuration)
7. [Expected Results](#expected-results)
8. [Report Generation](#report-generation)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Assignment Requirements Satisfied

| Requirement | Implementation |
|------------|----------------|
| Dataset | CIFAR-10 (official train/test split) |
| Architecture | Modified ResNet-18 (adapted for 32×32 images) |
| Training | From scratch, no pretrained weights |
| Epochs | 50 (configurable) |
| Random Seed | 42 (fixed for reproducibility) |
| Failure Cases | 3+ high-confidence failures identified |
| Explainability | Grad-CAM visualizations |
| Improvement | Cutout regularization (single modification) |
| Framework | PyTorch |

### Key Technical Contributions

1. **Modified ResNet-18 for CIFAR-10**: Standard ResNet-18 is designed for 224×224 ImageNet images. This implementation modifies the stem (3×3 conv with stride=1, no max pooling) to preserve spatial resolution for 32×32 images.

2. **Systematic Failure Discovery**: Automated pipeline to identify high-confidence (>90%) misclassifications and categorize them by failure type.

3. **Comprehensive Grad-CAM Analysis**: Visualizations comparing attention for predicted (wrong) class vs. true class, revealing spurious correlations.

4. **Cutout Regularization**: Data augmentation that randomly masks 16×16 patches, forcing the model to learn distributed features rather than shortcuts.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~2GB disk space for dataset and checkpoints

### Setup

```bash
# Clone or navigate to project directory
cd cnn-stress-test

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

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

## Project Structure

```
cnn-stress-test/
├── config.py              # All hyperparameters and settings
├── main.py                # Unified entry point
├── train.py               # Training loop
├── evaluate.py            # Evaluation and failure discovery
├── visualize.py           # Grad-CAM visualizations
├── requirements.txt       # Dependencies
├── README.md              # This file
│
├── models/
│   ├── __init__.py
│   └── resnet.py          # Modified ResNet-18 for CIFAR-10
│
├── utils/
│   ├── __init__.py
│   ├── cutout.py          # Cutout augmentation
│   ├── gradcam.py         # Grad-CAM implementation
│   └── data.py            # Data loading utilities
│
├── checkpoints/           # Saved model weights (created during training)
├── results/               # Evaluation results and failure analysis
├── figures/               # Generated visualizations
└── data/                  # CIFAR-10 dataset (auto-downloaded)
```

---

## Quick Start

### Run the Complete Pipeline

```bash
# This runs everything: train baseline, evaluate, visualize, train Cutout, compare
python main.py --mode all
```

Estimated time: ~2-3 hours on GPU, ~8-10 hours on CPU

### Step-by-Step Execution

```bash
# Step 1: Train baseline model
python main.py --mode baseline

# Step 2: Evaluate baseline and find failure cases
python main.py --mode evaluate

# Step 3: Generate Grad-CAM visualizations
python main.py --mode visualize --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth

# Step 4: Train Cutout-enhanced model
python main.py --mode cutout

# Step 5: Evaluate Cutout model
python main.py --mode evaluate --use-cutout

# Step 6: Generate comparison analysis
python main.py --mode compare
```

---

## Detailed Usage

### Training

```bash
# Train baseline model (no Cutout)
python train.py

# Train with Cutout augmentation
python train.py --cutout

# Resume training from checkpoint
python train.py --resume checkpoints/ResNet-18-CIFAR_baseline_seed42_epoch30.pth
```

### Evaluation

```bash
# Evaluate baseline model
python evaluate.py --checkpoint checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth

# Evaluate Cutout model
python evaluate.py --checkpoint checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth --cutout
```

### Visualization

```bash
# Generate Grad-CAM for baseline failures
python visualize.py --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth

# Compare baseline vs Cutout
python visualize.py \
    --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth \
    --cutout checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth
```

---

## Experimental Configuration

All hyperparameters are centralized in `config.py`:

### Training Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch Size | 128 | Stable gradient estimates |
| Epochs | 50 | Assignment constraint |
| Initial LR | 0.1 | Standard for ResNet on CIFAR |
| Momentum | 0.9 | Nesterov momentum |
| Weight Decay | 5×10⁻⁴ | L2 regularization |
| LR Schedule | MultiStepLR [25, 40] | Decay by 0.1× |

### Cutout Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| N Holes | 1 | Standard for CIFAR-10 |
| Length | 16 | Half image width (32/2) |

### Failure Analysis

| Parameter | Value |
|-----------|-------|
| Confidence Threshold | 0.9 (90%) |
| Target Layer (Grad-CAM) | layer4 |

---

## Expected Results

### Accuracy (with seed=42)

| Model | Test Accuracy | High-Conf Failures |
|-------|---------------|-------------------|
| Baseline | ~91-93% | ~15-25 |
| + Cutout | ~92-94% | ~10-18 |

### Common Failure Patterns

1. **Background Bias**: Birds classified as airplanes (blue sky)
2. **Texture Bias**: Frogs/deers confused due to green backgrounds
3. **Morphological Confusion**: Cat/dog misclassifications

### Generated Figures

After running the full pipeline, expect these figures in `figures/`:

```
figures/
├── ResNet-18-CIFAR_baseline_seed42_training_curves.png
├── ResNet-18-CIFAR_cutout_seed42_training_curves.png
├── ResNet-18-CIFAR_baseline_seed42_confusion_matrix.png
├── ResNet-18-CIFAR_baseline_seed42_per_class_accuracy.png
├── final_comparison_summary.png
├── gradcam_baseline/
│   ├── baseline_failure_1_gradcam.png
│   ├── baseline_failure_2_gradcam.png
│   └── ...
├── gradcam_cutout/
│   └── ...
└── comparison/
    ├── comparison_case_1.png
    └── ...
```

---

## Report Generation

### Required Figures for the Report

1. **Training Curves** (Section: Baseline Training)
   - `figures/*_training_curves.png`

2. **Confusion Matrix** (Section: Failure Case Discovery)
   - `figures/*_confusion_matrix.png`

3. **Failure Cases Grid** (Section: Failure Case Discovery)
   - `results/*_failures/failure_cases_grid.png`

4. **Grad-CAM Analysis** (Section: Explainability Analysis)
   - `figures/gradcam_baseline/baseline_failure_*_gradcam.png`

5. **Comparison** (Section: Constrained Improvement)
   - `figures/comparison/comparison_case_*.png`
   - `figures/final_comparison_summary.png`

### Failure Case Documentation

Each failure case is documented in:
- `results/ResNet-18-CIFAR_baseline_seed42_failures/failure_analysis.json`

Example entry:
```json
{
  "case_id": 1,
  "test_index": 3456,
  "true_class": "bird",
  "predicted_class": "airplane",
  "confidence": 0.97,
  "failure_category": "Background Bias (Blue/Sky)",
  "hypothesis": "Both 'bird' and 'airplane' often appear with blue backgrounds..."
}
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size in config.py
BATCH_SIZE = 64  # or 32
```

### Slow Training on CPU

Training 50 epochs on CPU takes ~8-10 hours. Consider:
- Using Google Colab (free GPU)
- Reducing epochs for testing: `NUM_EPOCHS = 10` in `config.py`

### OpenCV Import Error

```bash
pip install opencv-python-headless  # If running without display
```

### Reproducibility Issues

Ensure you're using the correct seed:
```python
import config
config.set_seed(42)  # Must be called before any random operations
```

---

## Citation

If you use this code for your assignment, please ensure you understand and can explain all components. This code is provided as a learning resource.

### Key References

1. He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
2. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
3. DeVries & Taylor, "Improved Regularization of CNNs with Cutout" (arXiv 2017)

---

## Random Seed Declaration

**Random Seed Used: 42**

This seed is set for:
- PyTorch random number generator
- NumPy random number generator
- Python's random module
- CUDA random number generator (if available)
- cuDNN deterministic mode enabled

---

## License

This code is provided for educational purposes as part of the Deep Learning course assignment.
