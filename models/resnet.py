"""
Modified ResNet-18 for CIFAR-10 (32x32 images)

Key Modifications from Standard ImageNet ResNet-18:
1. Initial conv: 7x7 stride 2 -> 3x3 stride 1 (preserves spatial resolution)
2. Remove initial max pooling layer (prevents excessive downsampling)
3. Final feature map: 4x4 spatial resolution (For Grad-CAM)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.
    
    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU

    """
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolution with potential downsampling
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second convolution maintains spatial dimensions
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Downsample layer for matching dimensions when stride > 1
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection with optional downsampling
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Residual addition: y = F(x) + x
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 modified for CIFAR-10 (32x32 RGB images).
    
    Architecture Overview:
        Input: 32x32x3
        Stem: Conv3x3 stride=1 -> 32x32x64
        Layer1: 2 BasicBlocks -> 32x32x64
        Layer2: 2 BasicBlocks (stride=2) -> 16x16x128
        Layer3: 2 BasicBlocks (stride=2) -> 8x8x256
        Layer4: 2 BasicBlocks (stride=2) -> 4x4x512
        GAP: Global Average Pool -> 1x1x512
        FC: Linear -> 10 classes
    
    Total parameters: ~11.2M 
    
    Spatial Resolution Preservation:
        Unlike standard ResNet-18 which reduces 224x224 -> 7x7,
        this variant preserves more spatial info: 32x32 -> 4x4
    """
    
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR, self).__init__()
        
        self.in_planes = 64
        
        # MODIFIED STEM: 3x3 conv with stride 1 (instead of 7x7 stride 2)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        # NO MAX POOLING - Removed to prevent aggressive downsampling
        # Standard ResNet uses: self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # We replace it with identity operation 
        
        # Residual layers with progressive downsampling
        self.layer1 = self._make_layer(64, 2, stride=1)   # 32x32 -> 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 32x32 -> 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 16x16 -> 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 8x8 -> 4x4
        
        # Global Average Pooling: 4x4x512 -> 1x1x512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classifier
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Weight initialization using Kaiming He initialization
        self._initialize_weights()
        
    def _make_layer(self, planes, num_blocks, stride):
        """
        Create a residual layer with multiple BasicBlocks.
        
        Args:
            planes: Number of output channels
            num_blocks: Number of BasicBlocks in this layer
            stride: Stride for the first block (1 or 2 for downsampling)
        """
        downsample = None
        
        # Create downsample layer if dimensions change
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion)
            )
        
        layers = []
        
        # First block may have stride > 1 for downsampling
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * BasicBlock.expansion
        
        # Remaining blocks have stride 1
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize weights using Kaiming He initialization.

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Input shape: (N, 3, 32, 32)
        Output shape: (N, num_classes)
        """
        # Stem (modified for CIFAR-10)
        x = self.conv1(x)          # (N, 64, 32, 32)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        # Note: No maxpool here!
        
        # Residual layers
        x = self.layer1(x)  # (N, 64, 32, 32)
        x = self.layer2(x)  # (N, 128, 16, 16)
        x = self.layer3(x)  # (N, 256, 8, 8)
        x = self.layer4(x)  # (N, 512, 4, 4) <- Target for Grad-CAM
        
        # Classification head
        x = self.avgpool(x)  # (N, 512, 1, 1)
        x = torch.flatten(x, 1)  # (N, 512)
        x = self.fc(x)  # (N, num_classes)
        
        return x
    
    def get_feature_maps(self, x, layer_name='layer4'):
        """
        Extract intermediate feature maps for Grad-CAM.
        
        Args:
            x: Input tensor (N, 3, 32, 32)
            layer_name: Which layer to extract ('layer1', 'layer2', 'layer3', 'layer4')
            
        Returns:
            Feature maps from specified layer
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.layer1(x)
        if layer_name == 'layer1':
            return x
            
        x = self.layer2(x)
        if layer_name == 'layer2':
            return x
            
        x = self.layer3(x)
        if layer_name == 'layer3':
            return x
            
        x = self.layer4(x)
        return x


def resnet18_cifar(num_classes=10, pretrained=False):
    """
    Function for creating ResNet-18 for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        pretrained: False 
        
    Returns:
        ResNet18CIFAR model instance
    """
    if pretrained:
        raise ValueError(
            "Pretrained weights are not used."
        )
    
    return ResNet18CIFAR(num_classes=num_classes)


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model architecture
    model = resnet18_cifar(num_classes=10)
    print(f"Model: ResNet-18 (CIFAR-10 variant)")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction for Grad-CAM
    # features = model.get_feature_maps(dummy_input, 'layer3')
    # print(f"Layer3 feature map shape: {features.shape}")  

    features = model.get_feature_maps(dummy_input, 'layer4')
    print(f"Layer4 feature map shape: {features.shape}")  
