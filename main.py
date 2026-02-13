"""
 CNN Stress-Testing 

This code run all components of the Project:

1. Train baseline model
2. Evaluate baseline and discover failure cases
3. Generate Grad-CAM visualizations
4. Train model with cutout
5. Compare baseline vs Cutout
6. Generate  figures

Usage:
    python main.py --mode all         # Run complete pipeline
    python main.py --mode baseline    # Train baseline only
    python main.py --mode cutout      # Train Cutout model only
    python main.py --mode evaluate    # Evaluate and find failures
    python main.py --mode visualize   # Generate Grad-CAM visualizations
    python main.py --mode compare     # Compare baseline vs Cutout
"""

import argparse
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import config
from train import train
from evaluate import evaluate, find_failure_cases, plot_confusion_matrix
from visualize import visualize, generate_gradcam_analysis, create_summary_visualization
from models.resnet import resnet18_cifar, count_parameters
from utils.data import get_dataset_statistics


def print_header(title):
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def run_baseline_training():
    """Train the baseline ResNet-18 model."""
    print_header("PHASE 1: BASELINE MODEL TRAINING")
    
    model, train_history, test_history = train(use_cutout=False)
    
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        f'{config.get_experiment_name(False)}_final.pth'
    )
    
    return checkpoint_path, test_history['acc'][-1]


def run_cutout_training():
    """Train the Cutout-enhanced model."""
    print_header("PHASE 4: CUTOUT MODEL TRAINING")
    
    model, train_history, test_history = train(use_cutout=True)
    
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        f'{config.get_experiment_name(True)}_final.pth'
    )
    
    return checkpoint_path, test_history['acc'][-1]


def run_evaluation(checkpoint_path, use_cutout=False):
    """Run evaluation and failure case discovery."""
    phase = "5" if use_cutout else "2"
    model_name = "CUTOUT" if use_cutout else "BASELINE"
    print_header(f"PHASE {phase}: {model_name} MODEL EVALUATION")
    
    eval_results, failure_cases, model = evaluate(checkpoint_path, use_cutout)
    
    return eval_results, failure_cases, model


def run_visualization(baseline_checkpoint, cutout_checkpoint=None):
    """Generate Grad-CAM visualizations."""
    print_header("PHASE 3/6: GRAD-CAM VISUALIZATION")
    
    visualize(baseline_checkpoint, cutout_checkpoint)


def run_comparison(baseline_checkpoint, cutout_checkpoint):
    """Generate comparison analysis between baseline and Cutout models."""
    print_header("PHASE 7: COMPARATIVE ANALYSIS")
    
    device = config.DEVICE
    
    # Load both models
    baseline_model = resnet18_cifar(num_classes=config.NUM_CLASSES)
    baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
    baseline_model = baseline_model.to(device).eval()
    
    cutout_model = resnet18_cifar(num_classes=config.NUM_CLASSES)
    cutout_model.load_state_dict(torch.load(cutout_checkpoint, map_location=device, weights_only=True))
    cutout_model = cutout_model.to(device).eval()
    
    # Load evaluation results
    baseline_exp = config.get_experiment_name(False)
    cutout_exp = config.get_experiment_name(True)
    
    baseline_results_path = os.path.join(config.RESULTS_DIR, f'{baseline_exp}_evaluation.json')
    cutout_results_path = os.path.join(config.RESULTS_DIR, f'{cutout_exp}_evaluation.json')
    
    with open(baseline_results_path, 'r') as f:
        baseline_results = json.load(f)
    
    with open(cutout_results_path, 'r') as f:
        cutout_results = json.load(f)
    
    # Add failure counts
    baseline_results['num_failures'] = baseline_results['num_high_confidence_failures']
    cutout_results['num_failures'] = cutout_results['num_high_confidence_failures']
    
    # Create summary visualization
    summary_path = os.path.join(config.FIGURES_DIR, 'final_comparison_summary.png')
    create_summary_visualization(baseline_results, cutout_results, summary_path)
    
    # Print comparison summary
    print("\n" + "=" * 50)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 50)
    
    print(f"\n{'Metric':<30} {'Baseline':>12} {'Cutout':>12} {'Delta':>10}")
    print("-" * 64)
    
    acc_delta = cutout_results['overall_accuracy'] - baseline_results['overall_accuracy']
    print(f"{'Test Accuracy (%)':<30} {baseline_results['overall_accuracy']:>12.2f} "
          f"{cutout_results['overall_accuracy']:>12.2f} {acc_delta:>+10.2f}")
    
    fail_delta = cutout_results['num_failures'] - baseline_results['num_failures']
    print(f"{'High-Conf Failures':<30} {baseline_results['num_failures']:>12} "
          f"{cutout_results['num_failures']:>12} {fail_delta:>+10}")
    
    # Per-class comparison
    print("\nPer-Class Accuracy Changes:")
    print("-" * 50)
    
    improvements = []
    for cls_name in config.CLASS_NAMES:
        baseline_acc = baseline_results['per_class_accuracy'][cls_name]
        cutout_acc = cutout_results['per_class_accuracy'][cls_name]
        delta = cutout_acc - baseline_acc
        improvements.append((cls_name, delta))
        
        symbol = "↑" if delta > 0.5 else "↓" if delta < -0.5 else "→"
        print(f"  {cls_name:<12}: {baseline_acc:>6.2f}% → {cutout_acc:>6.2f}% ({symbol} {delta:+.2f}%)")
    
    # Classes with most improvement
    improvements.sort(key=lambda x: x[1], reverse=True)
    print(f"\nMost improved class: {improvements[0][0]} ({improvements[0][1]:+.2f}%)")
    print(f"Most declined class: {improvements[-1][0]} ({improvements[-1][1]:+.2f}%)")
    
    return baseline_results, cutout_results


def generate_figures():
    """Generate all figures """
    print_header("GENERATING FIGURES")
    
    figures_generated = []
    
    # List all generated figures
    for root, dirs, files in os.walk(config.FIGURES_DIR):
        for file in files:
            if file.endswith('.png'):
                figures_generated.append(os.path.join(root, file))
    
    print(f"\nGenerated {len(figures_generated)} figures for visualize:")
    for fig in sorted(figures_generated):
        print(f"  • {fig}")
    
    return figures_generated


def run_full_pipeline():
    """Run the complete assignment pipeline."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "   CNN STRESS-TESTING - FULL PIPELINE".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    start_time = datetime.now()
    
    # Print configuration
    config.print_config()
    get_dataset_statistics()
    
    # Phase 1: Train baseline
    baseline_ckpt, baseline_acc = run_baseline_training()
    
    # Phase 2: Evaluate baseline
    baseline_eval, baseline_failures, baseline_model = run_evaluation(baseline_ckpt, False)
    
    # Phase 3: Visualize baseline failures
    run_visualization(baseline_ckpt, None)
    
    # Phase 4: Train Cutout model
    cutout_ckpt, cutout_acc = run_cutout_training()
    
    # Phase 5: Evaluate Cutout model
    cutout_eval, cutout_failures, cutout_model = run_evaluation(cutout_ckpt, True)
    
    # Phase 6: Visualize Cutout and comparison
    run_visualization(baseline_ckpt, cutout_ckpt)
    
    # Phase 7: Comparative analysis
    baseline_results, cutout_results = run_comparison(baseline_ckpt, cutout_ckpt)
    
    # Generate  figures summary
    figures = generate_figures()
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "#" * 70)
    print("#" + "   PIPELINE COMPLETE".center(68) + "#")
    print("#" * 70)
    
    print(f"\nTotal execution time: {duration}")
    print(f"\nKey Results:")
    print(f"   Baseline Test Accuracy: {baseline_acc:.2f}%")
    print(f"   Cutout Test Accuracy:   {cutout_acc:.2f}%")
    print(f"   Improvement:            {cutout_acc - baseline_acc:+.2f}%")
    print(f"\nOutput Directories:")
    print(f"   Checkpoints: {config.CHECKPOINT_DIR}")
    print(f"   Results:     {config.RESULTS_DIR}")
    print(f"   Figures:     {config.FIGURES_DIR}")
    print(f"\nTotal figures generated: {len(figures)}")


def main():
    """Main code with argument parsing."""
    parser = argparse.ArgumentParser(
        description='CNN Stress-Testing  Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode all              # Run complete pipeline
  python main.py --mode baseline         # Train baseline model only
  python main.py --mode cutout           # Train Cutout model only  
  python main.py --mode evaluate --checkpoint checkpoints/model.pth
  python main.py --mode visualize --baseline checkpoints/baseline.pth
  python main.py --mode compare --baseline checkpoints/baseline.pth --cutout checkpoints/cutout.pth
        """
    )
    
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'baseline', 'cutout', 'evaluate', 
                                 'visualize', 'compare', 'info'],
                        help='Pipeline mode to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path (for evaluate mode)')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Baseline checkpoint path')
    parser.add_argument('--cutout', type=str, default=None,
                        help='Cutout checkpoint path')
    parser.add_argument('--use-cutout', action='store_true',
                        help='Flag for Cutout model (for evaluate mode)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    config.set_seed()
    
    if args.mode == 'all':
        run_full_pipeline()
        
    elif args.mode == 'baseline':
        run_baseline_training()
        
    elif args.mode == 'cutout':
        run_cutout_training()
        
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            # Try to find checkpoint automatically
            exp_name = config.get_experiment_name(args.use_cutout)
            args.checkpoint = os.path.join(config.CHECKPOINT_DIR, f'{exp_name}_final.pth')
            
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            print("Please provide --checkpoint path or run training first.")
            return
            
        run_evaluation(args.checkpoint, args.use_cutout)
        
    elif args.mode == 'visualize':
        if args.baseline is None:
            args.baseline = os.path.join(
                config.CHECKPOINT_DIR,
                f'{config.get_experiment_name(False)}_final.pth'
            )
        
        if args.cutout is None:
            cutout_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'{config.get_experiment_name(True)}_final.pth'
            )
            if os.path.exists(cutout_path):
                args.cutout = cutout_path
                
        run_visualization(args.baseline, args.cutout)
        
    elif args.mode == 'compare':
        if args.baseline is None:
            args.baseline = os.path.join(
                config.CHECKPOINT_DIR,
                f'{config.get_experiment_name(False)}_final.pth'
            )
        if args.cutout is None:
            args.cutout = os.path.join(
                config.CHECKPOINT_DIR,
                f'{config.get_experiment_name(True)}_final.pth'
            )
            
        run_comparison(args.baseline, args.cutout)
        
    elif args.mode == 'info':
        print("\n" + "=" * 60)
        print("CNN STRESS-TESTING  INFO")
        print("=" * 60)
        config.print_config()
        get_dataset_statistics()
        
        # Model info
        model = resnet18_cifar(num_classes=config.NUM_CLASSES)
        print(f"\nModel Architecture: Modified ResNet-18 for CIFAR-10")
        print(f"Total Parameters: {count_parameters(model):,}")
        print(f"\nKey Modifications from Standard ResNet-18:")
        print("   Initial conv: 7x7 stride=2 → 3x3 stride=1")
        print("   Max pooling: Removed (preserves spatial resolution)")
        print("   Final feature map: 4x4x512 ")


if __name__ == "__main__":
    main()
