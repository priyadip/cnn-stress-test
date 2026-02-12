"""
Visualization  for Grad-CAM Analysis

Key Outputs:
1. Grad-CAM heatmaps for predicted (wrong) class
2. Grad-CAM heatmaps for true class
3. Comparative visualizations showing attention shift
4. Batch visualizations for multiple failure cases
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os
from tqdm import tqdm
import json

import config
from models.resnet import resnet18_cifar
from utils.gradcam import GradCAM, overlay_heatmap
from utils.data import get_test_dataset_with_indices, denormalize
from evaluate import find_failure_cases


def generate_gradcam_analysis(model, failure_cases, save_dir, model_name="baseline"):
    """
    Generate comprehensive Grad-CAM visualizations for failure cases.
    
    Creates detailed comparison figures showing:
    - Original image
    - Grad-CAM for predicted (wrong) class
    - Grad-CAM for true class
    - Analysis of attention differences
    
    Args:
        model: Trained model
        failure_cases: List of failure case dictionaries
        save_dir: Directory to save figures
        model_name: Name for file prefixes
    """
    os.makedirs(save_dir, exist_ok=True)
    device = config.DEVICE
    model = model.to(device)
    model.eval()
    
    # Initialize Grad-CAM with layer4 as target
    gradcam = GradCAM(model, model.layer4)
    
    print(f"\nGenerating Grad-CAM visualizations for {len(failure_cases)} cases...")
    
    for i, case in enumerate(tqdm(failure_cases[:6], desc='Creating visualizations')):
        # Get image tensor
        img_tensor = case['image'].unsqueeze(0).to(device)
        true_label = case['true_label']
        pred_label = case['pred_label']
        confidence = case['confidence']
        
        # Compute Grad-CAM for both classes
        heatmap_pred = gradcam(img_tensor.clone(), target_class=pred_label)
        heatmap_true = gradcam(img_tensor.clone(), target_class=true_label)
        
        # Denormalize image for visualization
        img_display = denormalize(case['image'])
        img_np = img_display.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Resize heatmaps to image size
        heatmap_pred_resized = cv2.resize(heatmap_pred, (32, 32))
        heatmap_true_resized = cv2.resize(heatmap_true, (32, 32))
        
        # Create overlays
        overlay_pred = overlay_heatmap(img_np, heatmap_pred, alpha=0.5)
        overlay_true = overlay_heatmap(img_np, heatmap_true, alpha=0.5)
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.5])
        
        # Title with case information
        true_name = config.CLASS_NAMES[true_label]
        pred_name = config.CLASS_NAMES[pred_label]
        
        fig.suptitle(
            f'Failure Case #{i+1}: True="{true_name}" But Predicted="{pred_name}" (Conf: {confidence:.1%})',
            fontsize=14, fontweight='bold', y=0.98
        )
        
        # Row 1: Original and analysis for PREDICTED class
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_np)
        ax1.set_title(f'Original Image\n(Ground Truth: {true_name})', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(heatmap_pred_resized, cmap='jet', vmin=0, vmax=1)
        ax2.set_title(f'Grad-CAM: "{pred_name}"\n(PREDICTED - Wrong)', fontsize=10, color='red')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(overlay_pred)
        ax3.set_title(f'Overlay: Predicted Class\nModel focuses HERE', fontsize=10)
        ax3.axis('off')
        
        # Difference heatmap
        ax4 = fig.add_subplot(gs[0, 3])
        diff_map = heatmap_pred_resized - heatmap_true_resized
        im4 = ax4.imshow(diff_map, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title('Attention Difference\n(Pred - True)', fontsize=10)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # Row 2: Analysis for TRUE class
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(img_np)
        ax5.set_title(f'Original Image\n(Test Index: {case["index"]})', fontsize=10)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(heatmap_true_resized, cmap='jet', vmin=0, vmax=1)
        ax6.set_title(f'Grad-CAM: "{true_name}"\n(TRUE Class)', fontsize=10, color='green')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(overlay_true)
        ax7.set_title(f'Overlay: True Class\nModel SHOULD focus here', fontsize=10)
        ax7.axis('off')
        
        # Probability distribution
        ax8 = fig.add_subplot(gs[1, 3])
        probs = case['all_probs']
        colors = ['green' if j == true_label else 'red' if j == pred_label else 'steelblue' 
                  for j in range(len(probs))]
        bars = ax8.bar(range(len(probs)), probs, color=colors)
        ax8.set_xticks(range(len(probs)))
        ax8.set_xticklabels(config.CLASS_NAMES, rotation=45, ha='right', fontsize=7)
        ax8.set_ylabel('Probability', fontsize=9)
        ax8.set_title('Class Probabilities', fontsize=10)
        ax8.set_ylim(0, 1)
        
        # Row 3: Analysis text
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis('off')
        
        # Compute attention statistics
        pred_focus = np.unravel_index(np.argmax(heatmap_pred_resized), heatmap_pred_resized.shape)
        true_focus = np.unravel_index(np.argmax(heatmap_true_resized), heatmap_true_resized.shape)
        
        # Determine if focus is on edges (potential background)
        def is_edge_focus(pos, size=32, margin=8):
            y, x = pos
            return y < margin or y > size - margin or x < margin or x > size - margin
        
        pred_edge = is_edge_focus(pred_focus)
        
        analysis_text = (
            f"ANALYSIS:\n"
            f"• Predicted class '{pred_name}' attention peak at position {pred_focus} "
            f"({'EDGE/BACKGROUND' if pred_edge else 'CENTER/OBJECT'} region)\n"
            f"• True class '{true_name}' attention peak at position {true_focus}\n"
            f"• Confidence in wrong prediction: {confidence:.1%} | "
            f"Probability for true class: {case['true_class_prob']:.1%}\n"

        )
        

        
        ax_text.text(0.02, 0.95, analysis_text, transform=ax_text.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, f'{model_name}_failure_{i+1}_gradcam.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    # Cleanup hooks
    gradcam.remove_hooks()
    print(f"\nGrad-CAM visualizations saved to {save_dir}")


def compare_baseline_vs_cutout(baseline_model, cutout_model, failure_cases_baseline,
                                save_dir):
    """
    Compare Grad-CAM attention between baseline and Cutout models.
    
    This visualization demonstrates how Cutout regularization changes
    the model's attention patterns, ideally showing more distributed
    attention on object features rather than background.
    
    Args:
        baseline_model: Trained baseline model
        cutout_model: Trained model with Cutout augmentation
        failure_cases_baseline: Failure cases from baseline model
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    device = config.DEVICE
    
    baseline_model = baseline_model.to(device).eval()
    cutout_model = cutout_model.to(device).eval()
    
    # Initialize Grad-CAM for both models
    gradcam_baseline = GradCAM(baseline_model, baseline_model.layer4)
    gradcam_cutout = GradCAM(cutout_model, cutout_model.layer4)
    
    print("\nComparing Baseline vs Cutout attention patterns...")
    
    for i, case in enumerate(tqdm(failure_cases_baseline[:3], desc='Comparing models')):
        img_tensor = case['image'].unsqueeze(0).to(device)
        true_label = case['true_label']
        pred_label_baseline = case['pred_label']
        
        # Get Cutout model's prediction for same image
        with torch.no_grad():
            output_cutout = cutout_model(img_tensor)
            probs_cutout = torch.softmax(output_cutout, dim=1)
            conf_cutout, pred_cutout = probs_cutout.max(dim=1)
            pred_label_cutout = pred_cutout.item()
            conf_cutout = conf_cutout.item()
        
        # Compute Grad-CAM for both models (for TRUE class)
        heatmap_baseline = gradcam_baseline(img_tensor.clone(), target_class=true_label)
        heatmap_cutout = gradcam_cutout(img_tensor.clone(), target_class=true_label)
        
        # Prepare image
        img_display = denormalize(case['image'])
        img_np = img_display.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # Resize heatmaps
        heatmap_baseline_resized = cv2.resize(heatmap_baseline, (32, 32))
        heatmap_cutout_resized = cv2.resize(heatmap_cutout, (32, 32))
        
        # Create overlays
        overlay_baseline = overlay_heatmap(img_np, heatmap_baseline, alpha=0.5)
        overlay_cutout = overlay_heatmap(img_np, heatmap_cutout, alpha=0.5)
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        true_name = config.CLASS_NAMES[true_label]
        pred_name_baseline = config.CLASS_NAMES[pred_label_baseline]
        pred_name_cutout = config.CLASS_NAMES[pred_label_cutout]
        
        fig.suptitle(
            f'Attention Comparison: Baseline vs Cutout Model\n'
            f'True Class: "{true_name}"',
            fontsize=14, fontweight='bold'
        )
        
        # Row 1: Baseline model
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f'Original\n(True: {true_name})', fontsize=10)
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(heatmap_baseline_resized, cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Baseline Grad-CAM\n(for {true_name})', fontsize=10)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(overlay_baseline)
        axes[0, 2].set_title('Baseline Overlay', fontsize=10)
        axes[0, 2].axis('off')
        
        # Baseline prediction info
        axes[0, 3].axis('off')
        baseline_text = (
            f"BASELINE MODEL\n"
            f"─────────────────\n"
            f"Predicted: {pred_name_baseline}\n"
            f"Confidence: {case['confidence']:.1%}\n"
            f"Correct: {'✓' if pred_label_baseline == true_label else '✗'}"
        )
        axes[0, 3].text(0.5, 0.5, baseline_text, transform=axes[0, 3].transAxes,
                        fontsize=11, ha='center', va='center',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcoral' if pred_label_baseline != true_label else 'lightgreen'))
        
        # Row 2: Cutout model
        axes[1, 0].imshow(img_np)
        axes[1, 0].set_title(f'Original\n(Test Idx: {case["index"]})', fontsize=10)
        axes[1, 0].axis('off')
        
        im2 = axes[1, 1].imshow(heatmap_cutout_resized, cmap='jet', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Cutout Grad-CAM\n(for {true_name})', fontsize=10)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(overlay_cutout)
        axes[1, 2].set_title('Cutout Overlay', fontsize=10)
        axes[1, 2].axis('off')
        
        # Cutout prediction info
        axes[1, 3].axis('off')
        cutout_text = (
            f"CUTOUT MODEL\n"
            f"─────────────────\n"
            f"Predicted: {pred_name_cutout}\n"
            f"Confidence: {conf_cutout:.1%}\n"
            f"Correct: {'✓' if pred_label_cutout == true_label else '✗'}"
        )
        axes[1, 3].text(0.5, 0.5, cutout_text, transform=axes[1, 3].transAxes,
                        fontsize=11, ha='center', va='center',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcoral' if pred_label_cutout != true_label else 'lightgreen'))
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(save_dir, f'comparison_case_{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved comparison: {save_path}")
    
    # Cleanup
    gradcam_baseline.remove_hooks()
    gradcam_cutout.remove_hooks()


def create_summary_visualization(baseline_results, cutout_results, save_path):
    """
    Create a summary comparison figure for the report.
    
    Shows key metrics side-by-side: accuracy, confidence on failures,
    and sample Grad-CAM comparisons.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy comparison
    models = ['Baseline', 'Cutout']
    accuracies = [baseline_results['overall_accuracy'], cutout_results['overall_accuracy']]
    
    bars = axes[0].bar(models, accuracies, color=['steelblue', 'seagreen'])
    axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[0].set_title('Overall Accuracy Comparison', fontsize=12)
    axes[0].set_ylim(80, 100)
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{acc:.2f}%', ha='center', fontsize=11)
    
    # Per-class accuracy comparison
    baseline_pca = list(baseline_results['per_class_accuracy'].values())
    cutout_pca = list(cutout_results['per_class_accuracy'].values())
    
    x = np.arange(config.NUM_CLASSES)
    width = 0.35
    
    axes[1].bar(x - width/2, baseline_pca, width, label='Baseline', color='steelblue')
    axes[1].bar(x + width/2, cutout_pca, width, label='Cutout', color='seagreen')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Per-Class Accuracy', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(config.CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(70, 100)
    
    # Summary statistics text
    axes[2].axis('off')
    summary_text = (
        "EXPERIMENTAL SUMMARY\n"
        "════════════════════════════════════\n\n"
        f"Dataset: CIFAR-10\n"
        f"Model: Modified ResNet-18\n"
        f"Training Epochs: {config.NUM_EPOCHS}\n"
        f"Random Seed: {config.RANDOM_SEED}\n\n"
        f"BASELINE MODEL:\n"
        f"   Test Accuracy: {baseline_results['overall_accuracy']:.2f}%\n"
        f"   High-conf failures: {baseline_results.get('num_failures', 'N/A')}\n\n"
        f"CUTOUT MODEL:\n"
        f"   Test Accuracy: {cutout_results['overall_accuracy']:.2f}%\n"
        f"   Cutout size: {config.CUTOUT_LENGTH}{config.CUTOUT_LENGTH}\n\n"
        f"IMPROVEMENT: {cutout_results['overall_accuracy'] - baseline_results['overall_accuracy']:+.2f}%"
    )
    
    axes[2].text(0.1, 0.95, summary_text, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Summary visualization saved to {save_path}")
    plt.close()


def visualize(baseline_checkpoint, cutout_checkpoint=None):
    """
    Main visualization function.
    
    Generates all required visualizations for the assignment report.
    
    Args:
        baseline_checkpoint: Path to baseline model checkpoint
        cutout_checkpoint: Path to Cutout model checkpoint (optional)
    """
    config.set_seed()
    device = config.DEVICE
    
    # Create output directories
    baseline_viz_dir = os.path.join(config.FIGURES_DIR, 'gradcam_baseline')
    comparison_dir = os.path.join(config.FIGURES_DIR, 'comparison')
    os.makedirs(baseline_viz_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Load baseline model
    print("Loading baseline model...")
    baseline_model = resnet18_cifar(num_classes=config.NUM_CLASSES)
    baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    
    # Find failure cases for baseline
    print("\nFinding failure cases for baseline model...")
    failure_cases_baseline = find_failure_cases(
        baseline_model, device,
        threshold=config.HIGH_CONFIDENCE_THRESHOLD,
        max_cases=10
    )
    
    # Generate Grad-CAM for baseline
    generate_gradcam_analysis(
        baseline_model, failure_cases_baseline,
        baseline_viz_dir, model_name='baseline'
    )
    
    # If Cutout model provided, do comparison
    if cutout_checkpoint and os.path.exists(cutout_checkpoint):
        print("\nLoading Cutout model for comparison...")
        cutout_model = resnet18_cifar(num_classes=config.NUM_CLASSES)
        cutout_model.load_state_dict(torch.load(cutout_checkpoint, map_location=device))
        cutout_model = cutout_model.to(device)
        cutout_model.eval()
        
        # Generate comparison visualizations
        compare_baseline_vs_cutout(
            baseline_model, cutout_model,
            failure_cases_baseline, comparison_dir
        )
        
        # Find failure cases for Cutout model
        print("\nFinding failure cases for Cutout model...")
        failure_cases_cutout = find_failure_cases(
            cutout_model, device,
            threshold=config.HIGH_CONFIDENCE_THRESHOLD,
            max_cases=10
        )
        
        # Generate Grad-CAM for Cutout model
        cutout_viz_dir = os.path.join(config.FIGURES_DIR, 'gradcam_cutout')
        generate_gradcam_analysis(
            cutout_model, failure_cases_cutout,
            cutout_viz_dir, model_name='cutout'
        )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Output directories:")
    print(f"   Baseline Grad-CAM: {baseline_viz_dir}")
    if cutout_checkpoint:
        print(f"   Cutout Grad-CAM: {cutout_viz_dir}")
        print(f"   Comparisons: {comparison_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--cutout', type=str, default=None,
                        help='Path to Cutout model checkpoint (optional)')
    
    args = parser.parse_args()
    
    visualize(args.baseline, args.cutout)
