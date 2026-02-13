"""
All hyperparameters, paths, and experimental settings are here.
"""

import torch
import os


RANDOM_SEED = 42  # Fixed seed for all experiments 

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_NAME = "CIFAR-10"
NUM_CLASSES = 10
IMAGE_SIZE = 32
NUM_CHANNELS = 3

# CIFAR-10 normalization statistics (precomputed from training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NAME = "ResNet-18-CIFAR"  # Modified ResNet-18 for 32x32 input
NUM_FEATURES_FINAL = 512  # Feature dimension before classifier

# ============================================================================
# TRAINING HYPERPARAMETERS (Baseline)
# ============================================================================
BATCH_SIZE = 128
NUM_EPOCHS = 50
INITIAL_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NESTEROV = True

# Learning rate schedule (MultiStepLR)
LR_MILESTONES = [25, 40]  # Decay at epochs 25 and 40
LR_GAMMA = 0.1  # Multiply LR by 0.1 at each milestone

# ============================================================================
# CUTOUT CONFIGURATION 
# ============================================================================
CUTOUT_N_HOLES = 1  # Number of holes to cut
CUTOUT_LENGTH = 16  # Size of each hole (16x16 for CIFAR-10)

# ============================================================================
# GRAD-CAM CONFIGURATION
# ============================================================================
#TARGET_LAYER = 'layer4'  # Target layer for Grad-CAM (final conv block)
TARGET_LAYER = 'layer3'  # Alternative: 'layer3' for finer spatial resolution


# ============================================================================
# FAILURE ANALYSIS CONFIGURATION
# ============================================================================
HIGH_CONFIDENCE_THRESHOLD = 0.7  # Threshold for "high-confidence" failures
NUM_FAILURE_CASES = None   # Number of failure cases to analyze 

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = './data'
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'
FIGURES_DIR = './figures'


for dir_path in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# ============================================================================
# LOGGING
# ============================================================================
PRINT_FREQ = 100  # Print every N batches during training


def set_seed(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_experiment_name(use_cutout=False):
    """Generate experiment name for saving checkpoints."""
    suffix = "_cutout" if use_cutout else "_baseline"
    return f"{MODEL_NAME}{suffix}_seed{RANDOM_SEED}"


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Device: {DEVICE}")
    print("-" * 60)
    print("Training Hyperparameters:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Initial LR: {INITIAL_LR}")
    print(f"  Momentum: {MOMENTUM}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  LR Milestones: {LR_MILESTONES}")
    print("-" * 60)
    print("Cutout Configuration:")
    print(f"  N Holes: {CUTOUT_N_HOLES}")
    print(f"  Length: {CUTOUT_LENGTH}")
    print("=" * 60)
