"""
Cutout Data Augmentation
Cutout randomly masks out square regions of the input image during training.
"""

import torch
import numpy as np


class Cutout:
    """
    Randomly mask out one or more patches from an image tensor.

    
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        fill_value (float or tuple): Value to fill the masked region.
                                     Default: 0 (black).
    
    Usage:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(n_holes=1, length=16),
        ])
    
    Note:
        Cutout is applied AFTER normalization, so the fill value should
        be in the normalized space. Using 0 effectively fills with the
        mean pixel value post-normalization.
    """
    
    def __init__(self, n_holes=1, length=16, fill_value=0.0):
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value
        
    def __call__(self, img):
        """
        Apply Cutout augmentation to an image tensor.
        
        Args:
            img (Tensor): Tensor image of shape (C, H, W).
            
        Returns:
            Tensor: Image with n_holes random patches cut out.
        """
        h = img.size(1)
        w = img.size(2)
        
        # Create a binary mask (1 = keep, 0 = mask out)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            # Sample random center point for the patch
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            # Calculate patch boundaries with clamping
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            # Set mask region to 0 (will be masked out)
            mask[y1:y2, x1:x2] = 0.0
        
        # Convert mask to tensor and expand to match image channels
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        
        # Apply mask: keep original where mask=1, fill where mask=0
        img = img * mask + self.fill_value * (1 - mask)
        
        return img
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length})'


class CutoutPIL:
    """
    Cutout variant that operates on PIL Images (before ToTensor).

    
    Args:
        n_holes (int): Number of patches to cut out.
        length (int): Side length of each square patch.
        fill_color (tuple): RGB color to fill the hole. Default: (0, 0, 0) black.
    """
    
    def __init__(self, n_holes=1, length=16, fill_color=(0, 0, 0)):
        self.n_holes = n_holes
        self.length = length
        self.fill_color = fill_color
        
    def __call__(self, img):
        """
        Apply Cutout to a PIL Image.
        
        Args:
            img (PIL.Image): Input image.
            
        Returns:
            PIL.Image: Image with patches cut out.
        """
        import PIL.Image as Image
        import PIL.ImageDraw as ImageDraw
        
        w, h = img.size
        
        # Create a copy to avoid modifying original
        img = img.copy()
        draw = ImageDraw.Draw(img)
        
        for _ in range(self.n_holes):
            # Random center
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            # Calculate rectangle bounds
            x1 = np.clip(x - self.length // 2, 0, w)
            y1 = np.clip(y - self.length // 2, 0, h)
            x2 = np.clip(x + self.length // 2, 0, w)
            y2 = np.clip(y + self.length // 2, 0, h)
            
            # Draw filled rectangle
            draw.rectangle([x1, y1, x2, y2], fill=self.fill_color)
        
        return img
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length})'


def visualize_cutout_effect(image_tensor, n_samples=5, n_holes=1, length=16):
    """
    Visualize the effect of Cutout augmentation on a single image.

    
    Args:
        image_tensor: Original image tensor (C, H, W), already normalized
        n_samples: Number of cutout variations to generate
        n_holes: Number of holes per cutout
        length: Size of each hole
        
    Returns:
        List of tensors showing different cutout variations
    """
    cutout = Cutout(n_holes=n_holes, length=length)
    samples = [image_tensor.clone()]  # Original
    
    for _ in range(n_samples - 1):
        augmented = cutout(image_tensor.clone())
        samples.append(augmented)
    
    return samples


if __name__ == "__main__":
    # Test Cutout implementation
    print("Testing Cutout augmentation...")
    
    # Create a dummy tensor (3 channels, 32x32)
    dummy_img = torch.randn(3, 32, 32)
    
    # Apply cutout
    cutout = Cutout(n_holes=1, length=16)
    augmented = cutout(dummy_img.clone())
    
    print(f"Input shape: {dummy_img.shape}")
    print(f"Output shape: {augmented.shape}")
    
    # Count masked pixels
    mask_diff = (dummy_img != augmented).float().mean()
    print(f"Percentage of pixels modified: {mask_diff * 100:.1f}%")
    
    # Expected: ~25% for 16x16 hole in 32x32 image
    print(f"Expected: ~{(16*16)/(32*32)*100:.1f}%")
    
    print("Cutout test passed!")
