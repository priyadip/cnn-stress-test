"""
Training Script for CNN Stress-Testing Assignment

This script handles:
1. Baseline model training (modified ResNet-18 on CIFAR-10)
2. Cutout-enhanced training (constrained improvement)
3. Logging of training/validation metrics
4. Checkpoint saving

The training procedure follows established best practices:
- SGD with Nesterov momentum for robust convergence
- MultiStepLR learning rate scheduling
- Proper weight decay (L2 regularization)
- Fixed random seed for reproducibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import config
from models.resnet import resnet18_cifar, count_parameters
from utils.data import get_cifar10_loaders


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: The CNN model
        train_loader: Training data loader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: SGD optimizer
        device: CPU or CUDA device
        epoch: Current epoch number (for logging)
        
    Returns:
        Tuple of (average loss, accuracy percentage)
    """
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = 100.0 * correct / targets.size(0)
        
        # Update meters
        losses.update(loss.item(), targets.size(0))
        accuracies.update(accuracy, targets.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accuracies.avg:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.4f}'
        })
    
    return losses.avg, accuracies.avg


def validate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test/validation set.
    
    Args:
        model: The CNN model
        test_loader: Test data loader
        criterion: Loss function
        device: CPU or CUDA device
        
    Returns:
        Tuple of (average loss, accuracy percentage)
    """
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Validating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100.0 * correct / targets.size(0)
            
            # Update meters
            losses.update(loss.item(), targets.size(0))
            accuracies.update(accuracy, targets.size(0))
    
    return losses.avg, accuracies.avg


def save_checkpoint(model, optimizer, scheduler, epoch, train_history, 
                    test_history, filename):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_history': train_history,
        'test_history': test_history,
        'config': {
            'random_seed': config.RANDOM_SEED,
            'batch_size': config.BATCH_SIZE,
            'initial_lr': config.INITIAL_LR,
            'momentum': config.MOMENTUM,
            'weight_decay': config.WEIGHT_DECAY,
            'lr_milestones': config.LR_MILESTONES,
        }
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def plot_training_curves(train_history, test_history, save_path, title_suffix=""):
    """
    Plot and save training curves (loss and accuracy).
    
    These plots are required for the assignment report.
    """
    epochs = range(1, len(train_history['loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(epochs, train_history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, test_history['loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Loss Curves{title_suffix}', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Mark LR decay points
    for milestone in config.LR_MILESTONES:
        if milestone <= len(epochs):
            axes[0].axvline(x=milestone, color='gray', linestyle='--', alpha=0.5)
            axes[0].text(milestone, axes[0].get_ylim()[1], f'LR×0.1', 
                        rotation=90, va='top', fontsize=8)
    
    # Accuracy curves
    axes[1].plot(epochs, train_history['acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, test_history['acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Accuracy Curves{title_suffix}', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Mark LR decay points
    for milestone in config.LR_MILESTONES:
        if milestone <= len(epochs):
            axes[1].axvline(x=milestone, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def train(use_cutout=False, resume_from=None):
    """
    Main training function.
    
    Args:
        use_cutout: Whether to use Cutout augmentation (constrained improvement)
        resume_from: Path to checkpoint to resume from (optional)
        
    Returns:
        Tuple of (trained model, training history, test history)
    """
    # Print configuration
    config.print_config()
    
    # Set random seed for reproducibility
    config.set_seed()
    print(f"\nRandom seed set to: {config.RANDOM_SEED}")
    
    # Experiment name for saving
    exp_name = config.get_experiment_name(use_cutout)
    print(f"Experiment: {exp_name}")
    print(f"Cutout augmentation: {'Enabled' if use_cutout else 'Disabled'}")
    
    # Device setup
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Data loaders
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(use_cutout=use_cutout)
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Model
    print("\nInitializing model...")
    model = resnet18_cifar(num_classes=config.NUM_CLASSES)
    model = model.to(device)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: SGD with Nesterov momentum (gold standard for ResNets)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.INITIAL_LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=config.NESTEROV
    )
    
    # Learning rate scheduler: MultiStepLR with decay at epochs 25, 40
    scheduler = MultiStepLR(
        optimizer,
        milestones=config.LR_MILESTONES,
        gamma=config.LR_GAMMA
    )
    
    # Training history
    train_history = {'loss': [], 'acc': []}
    test_history = {'loss': [], 'acc': []}
    
    start_epoch = 0
    best_acc = 0.0
    
    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_history = checkpoint['train_history']
        test_history = checkpoint['test_history']
        best_acc = max(test_history['acc'])
        print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        test_history['loss'].append(test_loss)
        test_history['acc'].append(test_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(config.CHECKPOINT_DIR, f'{exp_name}_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  *** New best accuracy! Saved to {best_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.NUM_EPOCHS:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'{exp_name}_epoch{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, 
                          train_history, test_history, ckpt_path)
    
    total_time = time.time() - total_start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Final Test Accuracy: {test_history['acc'][-1]:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    # Save training curves
    curves_path = os.path.join(config.FIGURES_DIR, f'{exp_name}_training_curves.png')
    title_suffix = " (Baseline)" if not use_cutout else " (with Cutout)"
    plot_training_curves(train_history, test_history, curves_path, title_suffix)
    
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, f'{exp_name}_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    
    # Save training history as JSON
    history_path = os.path.join(config.RESULTS_DIR, f'{exp_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train': train_history,
            'test': test_history,
            'best_acc': best_acc,
            'final_acc': test_history['acc'][-1],
            'config': {
                'random_seed': config.RANDOM_SEED,
                'batch_size': config.BATCH_SIZE,
                'initial_lr': config.INITIAL_LR,
                'momentum': config.MOMENTUM,
                'weight_decay': config.WEIGHT_DECAY,
                'epochs': config.NUM_EPOCHS,
                'lr_milestones': config.LR_MILESTONES,
                'use_cutout': use_cutout,
            }
        }, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return model, train_history, test_history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--cutout', action='store_true',
                        help='Use Cutout augmentation (constrained improvement)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train(use_cutout=args.cutout, resume_from=args.resume)
