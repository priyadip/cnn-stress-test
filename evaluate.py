"""
Evaluation and Failure Case Discovery Script

This script performs:
1. Full evaluation on the test set
2. Identification of high-confidence failure cases
3. Confusion matrix generation
4. Per-class accuracy analysis
5. Saving failure cases for Grad-CAM analysis

The failure discovery process follows the assignment requirements:
- Find misclassifications with confidence > 70%
- Identify high-confidence misclassifications
- Track indices for reproducible analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os
import json
from collections import defaultdict

import config
from models.resnet import resnet18_cifar
from utils.data import get_cifar10_loaders, get_test_dataset_with_indices, denormalize


def evaluate_model(model, test_loader, device):
    """
    Comprehensive evaluation of the model on the test set.
    
    Returns detailed metrics including per-sample predictions,
    confidences, and correctness flags.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: CPU or CUDA device
        
    Returns:
        Dictionary containing all evaluation metrics and predictions
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    all_correct = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predictions and confidences
            confidences, predictions = probabilities.max(dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_correct.extend((predictions.cpu() == targets).numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)
    
    # Compute metrics
    accuracy = 100.0 * all_correct.sum() / len(all_correct)
    
    # Per-class accuracy
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    for target, correct in zip(all_targets, all_correct):
        per_class_total[target] += 1
        if correct:
            per_class_correct[target] += 1
    
    per_class_accuracy = {
        cls: 100.0 * per_class_correct[cls] / per_class_total[cls]
        for cls in range(config.NUM_CLASSES)
    }
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'confidences': all_confidences,
        'correct': all_correct,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
    }


def find_failure_cases(model, device, threshold=0.7, max_cases=None):
    """
    Find high-confidence failure cases for analysis.
    
    A failure case is defined as:
    - Model prediction != ground truth
    - Prediction confidence > threshold (default 90%)
    
    Args:
        model: Trained model
        device: CPU or CUDA device
        threshold: Minimum confidence for "high-confidence" failures
        max_cases: Maximum number of cases to return
        
    Returns:
        List of dictionaries containing failure case information
    """
    model.eval()
    
    # Use dataset that returns indices
    test_dataset = get_test_dataset_with_indices()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Process one at a time for clarity
        shuffle=False,
        num_workers=0
    )
    
    failure_cases = []
    
    print(f"\nSearching for high-confidence failures (confidence > {threshold:.0%})...")
    
    with torch.no_grad():
        for img, target, idx in tqdm(test_loader, desc='Finding failures'):
            img = img.to(device)
            
            # Forward pass
            output = model(img)
            probs = F.softmax(output, dim=1)
            confidence, prediction = probs.max(dim=1)
            
            confidence = confidence.item()
            prediction = prediction.item()
            target = target.item()
            idx = idx.item()
            
            # Check if this is a high-confidence failure
            if prediction != target and confidence > threshold:
                # Get all class probabilities for this sample
                all_probs = probs[0].cpu().numpy()
                
                failure_cases.append({
                    'index': idx,
                    'image': img.squeeze(0).cpu(),  # Store normalized tensor
                    'true_label': target,
                    'pred_label': prediction,
                    'confidence': confidence,
                    'true_class_prob': all_probs[target],
                    'all_probs': all_probs,
                })
    
    # Sort by confidence (most confident failures first)
    failure_cases.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"Found {len(failure_cases)} high-confidence failures")
    
    # Return top cases
    return failure_cases[:max_cases]


def analyze_failure_cases(failure_cases, save_dir):
    """
    Analyze and document failure cases for the report.
    
    Creates a detailed analysis file and visualizations for each failure case.
    
    Args:
        failure_cases: List of failure case dictionaries
        save_dir: Directory to save analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    analysis_results = []
    
    for i, case in enumerate(failure_cases):  # Analyze all cases for comprehensive report
        true_label = case['true_label']
        pred_label = case['pred_label']
        confidence = case['confidence']
        
        
        
        result = {
            'case_id': i + 1,
            'test_index': case['index'],
            'true_class': config.CLASS_NAMES[true_label],
            'true_label_idx': true_label,
            'predicted_class': config.CLASS_NAMES[pred_label],
            'pred_label_idx': pred_label,
            'confidence': float(confidence),
            'true_class_probability': float(case['true_class_prob']),

        }
        
        analysis_results.append(result)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Failure Case #{i+1}")
        print(f"{'='*60}")
        print(f"Test Index: {case['index']}")
        print(f"True Class: {result['true_class']} (idx: {true_label})")
        print(f"Predicted: {result['predicted_class']} (idx: {pred_label})")
        print(f"Confidence: {confidence:.2%}")
        print(f"True Class Prob: {case['true_class_prob']:.2%}")

    
    # Save analysis to JSON
    analysis_path = os.path.join(save_dir, 'failure_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nFailure analysis saved to {analysis_path}")
    
    return analysis_results


def plot_confusion_matrix(targets, predictions, save_path):
    """
    Generate and save confusion matrix visualization.
    
    The confusion matrix reveals systematic failure patterns,
    such as frequent confusion between similar classes.
    """
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - CIFAR-10 Test Set', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_per_class_accuracy(per_class_accuracy, save_path):
    """
    Visualize per-class accuracy as a bar chart.
    
    Identifies which classes are most challenging for the model.
    """
    classes = list(range(config.NUM_CLASSES))
    accuracies = [per_class_accuracy[c] for c in classes]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='steelblue', edgecolor='black')
    
    # Color bars below average differently
    avg_acc = np.mean(accuracies)
    for bar, acc in zip(bars, accuracies):
        if acc < avg_acc:
            bar.set_color('indianred')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy on CIFAR-10 Test Set', fontsize=14)
    plt.xticks(classes, config.CLASS_NAMES, rotation=45, ha='right')
    plt.axhline(y=avg_acc, color='gray', linestyle='--', 
                label=f'Average: {avg_acc:.1f}%')
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to {save_path}")
    plt.close()


def visualize_failure_cases(failure_cases, save_dir, num_cases=6):
    """
    Create a grid visualization of failure cases.
    
    Shows original images with true and predicted labels.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_cases = min(num_cases, len(failure_cases))
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, case in enumerate(failure_cases[:num_cases]):
        ax = axes[i]
        
        # Denormalize image for display
        img = denormalize(case['image'])
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        true_name = config.CLASS_NAMES[case['true_label']]
        pred_name = config.CLASS_NAMES[case['pred_label']]
        conf = case['confidence']
        
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf:.1%})",
            fontsize=10,
            color='red' if case['true_label'] != case['pred_label'] else 'green'
        )
        ax.axis('off')
    
    # Hide unused axes
    for j in range(num_cases, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('High-Confidence Failure Cases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'failure_cases_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Failure cases grid saved to {save_path}")
    plt.close()


def evaluate(checkpoint_path, use_cutout=False):
    """
    Main evaluation function.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        use_cutout: Whether this is the Cutout model (for naming)
        
    Returns:
        Dictionary of evaluation results and failure cases
    """
    # Setup
    config.set_seed()
    device = config.DEVICE
    exp_name = config.get_experiment_name(use_cutout)
    
    print("=" * 60)
    print(f"EVALUATING: {exp_name}")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = resnet18_cifar(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Get data loader
    _, test_loader = get_cifar10_loaders(use_cutout=False)  # No augmentation for eval
    
    # Full evaluation
    print("\nRunning full evaluation...")
    eval_results = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*40}")
    print(f"OVERALL ACCURACY: {eval_results['accuracy']:.2f}%")
    print(f"{'='*40}")
    
    print("\nPer-class accuracy:")
    for cls_idx, acc in eval_results['per_class_accuracy'].items():
        print(f"  {config.CLASS_NAMES[cls_idx]:12s}: {acc:.2f}%")
    
    # Save confusion matrix
    cm_path = os.path.join(config.FIGURES_DIR, f'{exp_name}_confusion_matrix.png')
    plot_confusion_matrix(eval_results['targets'], eval_results['predictions'], cm_path)
    
    # Save per-class accuracy
    pca_path = os.path.join(config.FIGURES_DIR, f'{exp_name}_per_class_accuracy.png')
    plot_per_class_accuracy(eval_results['per_class_accuracy'], pca_path)
    
    # Find failure cases
    failure_cases = find_failure_cases(
        model, device, 
        threshold=config.HIGH_CONFIDENCE_THRESHOLD,
        max_cases=None
    )
    
    # Analyze and document failures
    failure_dir = os.path.join(config.RESULTS_DIR, f'{exp_name}_failures')
    analysis_results = analyze_failure_cases(failure_cases, failure_dir)
    
    # Visualize failure cases
    visualize_failure_cases(failure_cases, failure_dir)
    
    # Save comprehensive results
    results = {
        'experiment': exp_name,
        'overall_accuracy': float(eval_results['accuracy']),
        'per_class_accuracy': {
            config.CLASS_NAMES[k]: float(v) 
            for k, v in eval_results['per_class_accuracy'].items()
        },
        'num_high_confidence_failures': len(failure_cases),
        'failure_threshold': config.HIGH_CONFIDENCE_THRESHOLD,
    }
    
    results_path = os.path.join(config.RESULTS_DIR, f'{exp_name}_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to {results_path}")
    
    return eval_results, failure_cases, model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--cutout', action='store_true',
                        help='Flag if this is the Cutout model')
    
    args = parser.parse_args()
    
    evaluate(args.checkpoint, use_cutout=args.cutout)
