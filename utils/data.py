"""
Data Loading Utilities for CIFAR-10

This module provides standardized data loading pipelines for:
1. Baseline training (standard augmentation)
2. Cutout-enhanced training (with cutout data augmentation)
3. Evaluation (no augmentation)

"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cutout import Cutout
import config


def get_train_transforms(use_cutout: bool = False) -> transforms.Compose:
    """
    Get training transforms for CIFAR-10.
    
    Baseline Augmentation:
    1. Random crop with padding: Forces translation invariance
    2. Random horizontal flip: Doubles effective dataset size
    3. Normalization: Essential for stable SGD convergence
    
    With Cutout :
    4. Cutout: Randomly masks 16x16 patches to prevent shortcut learning
    
    Args:
        use_cutout: Whether to include Cutout augmentation
        
    Returns:
        Composed transforms for training
    """
    transform_list = [
        # Padding + random crop: shift-invariance regularization
        transforms.RandomCrop(32, padding=4),
        
        # Horizontal flip: left-right symmetry for most CIFAR classes
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Convert to tensor (also scales to [0,1])
        transforms.ToTensor(),
        
        # Dataset-specific normalization (computed from CIFAR-10 training set)
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ]
    
    # Add Cutout AFTER normalization
    if use_cutout:
        transform_list.append(
            Cutout(n_holes=config.CUTOUT_N_HOLES, length=config.CUTOUT_LENGTH)
        )
    
    return transforms.Compose(transform_list)


def get_test_transforms() -> transforms.Compose:
    """
    Get test/validation transforms for CIFAR-10.
    
    No augmentation is applied during evaluation to ensure
    consistent and reproducible measurements.
    
    Returns:
        Composed transforms for testing
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])


def get_cifar10_loaders(use_cutout: bool = False,
                        batch_size: int = None,
                        num_workers: int = None,
                        data_dir: str = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 data loaders for training and testing.
    
    This function downloads CIFAR-10 if not present and creates
    DataLoaders with the appropriate transforms and settings.
    
    Args:
        use_cutout: Whether to use Cutout augmentation in training
        batch_size: Batch size (default: from config)
        num_workers: Number of data loading workers (default: from config)
        data_dir: Data directory (default: from config)
        
    Returns:
        Tuple of (train_loader, test_loader)

    """
    # Use defaults from config if not specified
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    data_dir = data_dir or config.DATA_DIR
    
    # Get transforms
    train_transform = get_train_transforms(use_cutout=use_cutout)
    test_transform = get_test_transforms()
    
    # Load datasets (downloads if not present)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Essential for SGD convergence
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True  # Avoid small final batch for BatchNorm stability
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic evaluation
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=False  # Evaluate ALL test samples
    )
    
    return train_loader, test_loader


def get_test_dataset_with_indices() -> torchvision.datasets.CIFAR10:
    """
    Get test dataset that also returns indices (for failure case tracking).
    
    Returns:
        CIFAR10 dataset with transform that yields (image, label, index)
    """
    test_transform = get_test_transforms()
    
    class CIFAR10WithIndex(torchvision.datasets.CIFAR10):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            return img, target, index
    
    test_dataset = CIFAR10WithIndex(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )
    
    return test_dataset


def get_raw_test_dataset() -> torchvision.datasets.CIFAR10:
    """
    Get raw test dataset WITHOUT normalization (for visualization).

    
    Returns:
        CIFAR10 dataset with only ToTensor transform
    """
    raw_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return torchvision.datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=raw_transform
    )


def denormalize(tensor: torch.Tensor,
                mean: Tuple[float, ...] = config.CIFAR10_MEAN,
                std: Tuple[float, ...] = config.CIFAR10_STD) -> torch.Tensor:
    """
    Reverse normalization for visualization purposes.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (N, C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:  # Batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def get_dataset_statistics():
    """
    Print dataset statistics.
    """
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True
    )
    
    print("=" * 50)
    print("CIFAR-10 Dataset Statistics")
    print("=" * 50)
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}x{config.NUM_CHANNELS}")
    print(f"Samples per class (train): {len(train_dataset) // config.NUM_CLASSES:,}")
    print(f"Samples per class (test): {len(test_dataset) // config.NUM_CLASSES:,}")
    print("-" * 50)
    print("Class names:")
    for idx, name in enumerate(config.CLASS_NAMES):
        print(f"  {idx}: {name}")
    print("-" * 50)
    print("Normalization parameters:")
    print(f"  Mean: {config.CIFAR10_MEAN}")
    print(f"  Std:  {config.CIFAR10_STD}")
    print("=" * 50)


if __name__ == "__main__":
    # Test data loading
    config.set_seed()
    
    print("Testing data loading utilities...")
    
    # Test baseline loaders
    train_loader, test_loader = get_cifar10_loaders(use_cutout=False)
    print(f"Baseline - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Test cutout loaders
    train_loader_cutout, _ = get_cifar10_loaders(use_cutout=True)
    print(f"Cutout - Train batches: {len(train_loader_cutout)}")
    
    # Verify batch shapes
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    
    # Print dataset statistics
    get_dataset_statistics()
    
    print("\nData loading test passed!")
