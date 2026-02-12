"""
Gradient-weighted Class Activation Mapping (Grad-CAM)

Paper: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
       Selvaraju et al., ICCV 2017

Grad-CAM provides visual explanations for CNN decisions by producing a coarse
localization map highlighting important regions in the image for predicting a concept.

Mathematical Formulation:
    Given feature maps A^k from a target convolutional layer, and class score y^c,
    the importance weight for feature map k is:
    
        α_k^c = (1/Z) * Σ_i Σ_j (∂y^c / ∂A^k_ij)
    
    where Z is the spatial size of the feature map (global average pooling).
    
    The final Grad-CAM heatmap is:
    
        L^c_Grad-CAM = ReLU( Σ_k α_k^c * A^k )
    
    The ReLU suppresses negative contributions (features that argue against the class).
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple, List


class GradCAM:
    """
    Grad-CAM implementation for CNN interpretability.
    
    This class hooks into a specified layer of a CNN and computes
    gradient-weighted activation maps that highlight regions
    important for a given class prediction.
    
    Args:
        model: The trained CNN model
        target_layer: The layer to compute Grad-CAM from (e.g., model.layer4)
        
    Usage:
        gradcam = GradCAM(model, model.layer4)
        heatmap = gradcam(input_tensor, target_class=predicted_class)
        visualization = gradcam.overlay_heatmap(input_image, heatmap)
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        
        def forward_hook(module, input, output):
            # Store the activations (feature maps) from forward pass
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            # Store the gradients flowing back through the layer
            self.gradients = grad_output[0].detach()
        
        # Register the hooks
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
    def remove_hooks(self):
        """Remove the registered hooks (cleanup)."""
        self.forward_handle.remove()
        self.backward_handle.remove()
        
    def __call__(self, input_tensor: torch.Tensor, 
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given input and target class.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Class index to compute Grad-CAM for.
                         If None, uses the predicted class.
                         
        Returns:
            heatmap: Numpy array of shape (H, W) with values in [0, 1]
        """
        self.model.eval()
        
        # Ensure input requires gradients
        input_tensor = input_tensor.clone()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero existing gradients
        self.model.zero_grad()
        
        # Backward pass from the target class score
        # We use the logit (pre-softmax) score, not the probability
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get the stored activations and gradients
        activations = self.activations  # Shape: (1, C, H', W')
        gradients = self.gradients      # Shape: (1, C, H', W')
        
        # Compute importance weights via global average pooling of gradients
        # α_k^c = GAP(∂y^c / ∂A^k)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)
        
        # Weighted combination of feature maps
        # Σ_k α_k^c * A^k
        cam = (weights * activations).sum(dim=1, keepdim=True)  # Shape: (1, 1, H', W')
        
        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def generate_for_all_classes(self, input_tensor: torch.Tensor, 
                                  num_classes: int = 10) -> dict:
        """
        Generate Grad-CAM heatmaps for all classes.
        
        Useful for comparing what regions drive different class predictions.
        
        Args:
            input_tensor: Input image tensor
            num_classes: Number of classes
            
        Returns:
            Dictionary mapping class index to heatmap
        """
        heatmaps = {}
        for class_idx in range(num_classes):
            heatmaps[class_idx] = self(input_tensor.clone(), target_class=class_idx)
        return heatmaps


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, 
                    alpha: float = 0.5) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image: Original image as numpy array (H, W, C) in [0, 1] or [0, 255]
        heatmap: Grad-CAM heatmap (H', W') in [0, 1]
        alpha: Transparency for overlay (0 = only heatmap, 1 = only image)
        
    Returns:
        Blended visualization as numpy array (H, W, C)
    """
    # Ensure image is in [0, 255] uint8 format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to colormap (jet colormap: blue -> green -> yellow -> red)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend image and heatmap
    overlay = (alpha * image + (1 - alpha) * heatmap_colored).astype(np.uint8)
    
    return overlay


def visualize_gradcam(image_tensor: torch.Tensor,
                      heatmap: np.ndarray,
                      mean: Tuple[float, ...],
                      std: Tuple[float, ...],
                      true_label: str,
                      pred_label: str,
                      confidence: float,
                      title: str = "",
                      save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive Grad-CAM visualization figure.
    
    Args:
        image_tensor: Normalized input tensor (C, H, W)
        heatmap: Grad-CAM heatmap
        mean: Dataset normalization mean
        std: Dataset normalization std
        true_label: Ground truth class name
        pred_label: Predicted class name
        confidence: Prediction confidence (0-1)
        title: Optional title for the figure
        save_path: Path to save the figure
    """
    # Denormalize image
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    
    # Create overlay
    overlay = overlay_heatmap(img, heatmap, alpha=0.5)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f'Original\nTrue: {true_label}')
    axes[0].axis('off')
    
    # Heatmap
    heatmap_display = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    im = axes[1].imshow(heatmap_display, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'Grad-CAM\nPred: {pred_label}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nConf: {confidence:.2%}')
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def save_gradcam_comparison(image_tensor: torch.Tensor,
                            heatmap_pred: np.ndarray,
                            heatmap_true: np.ndarray,
                            mean: Tuple[float, ...],
                            std: Tuple[float, ...],
                            true_label: str,
                            pred_label: str,
                            confidence: float,
                            save_path: str) -> None:
    """
    Create a comparison visualization showing Grad-CAM for both
    predicted (wrong) class and true class.
    
    This is crucial for understanding WHY the model made a mistake:
    - Heatmap for predicted class shows what the model focused on
    - Heatmap for true class shows where it SHOULD have focused
    
    Args:
        image_tensor: Normalized input tensor (C, H, W)
        heatmap_pred: Grad-CAM for predicted (wrong) class
        heatmap_true: Grad-CAM for true class
        mean, std: Normalization parameters
        true_label, pred_label: Class names
        confidence: Prediction confidence
        save_path: Path to save figure
    """
    # Denormalize image
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    
    # Resize heatmaps
    heatmap_pred_resized = cv2.resize(heatmap_pred, (img.shape[1], img.shape[0]))
    heatmap_true_resized = cv2.resize(heatmap_true, (img.shape[1], img.shape[0]))
    
    # Create overlays
    overlay_pred = overlay_heatmap(img, heatmap_pred, alpha=0.5)
    overlay_true = overlay_heatmap(img, heatmap_true, alpha=0.5)
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: Analysis for PREDICTED (wrong) class
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=10)
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(heatmap_pred_resized, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Grad-CAM: "{pred_label}"\n(Predicted - WRONG)', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(overlay_pred)
    axes[0, 2].set_title(f'Overlay\nConfidence: {confidence:.2%}', fontsize=10)
    axes[0, 2].axis('off')
    
    # Row 2: Analysis for TRUE class
    axes[1, 0].imshow(img)
    axes[1, 0].set_title(f'Ground Truth: "{true_label}"', fontsize=10)
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(heatmap_true_resized, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Grad-CAM: "{true_label}"\n(True Class)', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(overlay_true)
    axes[1, 2].set_title('Overlay (True Class)', fontsize=10)
    axes[1, 2].axis('off')
    
    # Add colorbars
    fig.colorbar(im1, ax=axes[0, :], shrink=0.5, location='right', pad=0.02)
    
    fig.suptitle(
        f'Failure Analysis: True="{true_label}" → Predicted="{pred_label}"',
        fontsize=12, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison visualization to {save_path}")
    plt.close()


def batch_gradcam_analysis(model: torch.nn.Module,
                           target_layer: torch.nn.Module,
                           failure_cases: List[dict],
                           class_names: List[str],
                           mean: Tuple[float, ...],
                           std: Tuple[float, ...],
                           output_dir: str) -> None:
    """
    Perform Grad-CAM analysis on multiple failure cases.
    
    Args:
        model: Trained model
        target_layer: Layer for Grad-CAM
        failure_cases: List of dicts with 'image', 'true_label', 'pred_label', 'confidence'
        class_names: List of class names
        mean, std: Normalization parameters
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    gradcam = GradCAM(model, target_layer)
    
    for idx, case in enumerate(failure_cases):
        image_tensor = case['image'].unsqueeze(0)
        true_idx = case['true_label']
        pred_idx = case['pred_label']
        confidence = case['confidence']
        
        # Compute Grad-CAM for predicted (wrong) class
        heatmap_pred = gradcam(image_tensor, target_class=pred_idx)
        
        # Compute Grad-CAM for true class
        heatmap_true = gradcam(image_tensor, target_class=true_idx)
        
        # Save comparison visualization
        save_path = os.path.join(output_dir, f'failure_case_{idx+1}.png')
        save_gradcam_comparison(
            case['image'],
            heatmap_pred,
            heatmap_true,
            mean, std,
            class_names[true_idx],
            class_names[pred_idx],
            confidence,
            save_path
        )
    
    gradcam.remove_hooks()
    print(f"Generated {len(failure_cases)} Grad-CAM visualizations in {output_dir}")


if __name__ == "__main__":
    # Test Grad-CAM with a dummy model
    import sys
    sys.path.append('..')
    from models.resnet import resnet18_cifar
    
    print("Testing Grad-CAM implementation...")
    
    # Create model and dummy input
    model = resnet18_cifar(num_classes=10)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, model.layer4)
    
    # Compute heatmap
    heatmap = gradcam(dummy_input, target_class=0)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Cleanup
    gradcam.remove_hooks()
    
    print("Grad-CAM test passed!")
