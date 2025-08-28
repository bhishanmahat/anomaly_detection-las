"""
Local Anomaly Synthesis (LAS)

Three-Step Process:
1. Step I: Anomaly Mask Generation (using Perlin noise and foreground mask)
2. Step II: Anomaly Texture Generation (DTD textures with augmentations)
3. Step III: Overlay Fusion (transparency-based blending)
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import DTD
import numpy as np
import random
import math


def generate_fast_perlin_mask(height, width, device, scale=6.0, octaves=4):
    """Fast vectorized Perlin-like noise generation"""
    # Create coordinate meshgrid
    x = torch.linspace(0, scale, width, device=device)
    y = torch.linspace(0, scale, height, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Multi-octave noise using trigonometric functions
    noise = torch.zeros_like(X)
    frequency = 1.0
    amplitude = 1.0
    
    for _ in range(octaves):
        # Use multiple trigonometric combinations for organic-looking noise
        layer_noise = (
            torch.sin(X * frequency) * torch.cos(Y * frequency) +
            0.5 * torch.sin(X * frequency * 2) * torch.sin(Y * frequency * 2) +
            0.25 * torch.cos(X * frequency * 3) * torch.cos(Y * frequency * 3)
        )
        noise += layer_noise * amplitude
        frequency *= 2.0
        amplitude *= 0.5
    
    # Normalize and binarize
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return (noise > 0.5).float()


class AnomalyMaskGeneration:
    """Step I: Generate anomaly masks using Perlin noise and foreground constraints"""
    
    def __init__(self, alpha=1/3):
        """
        Args:
            alpha: Parameter for mask combination strategy (paper default: 1/3)
        """
        self.alpha = alpha
        
    def generate_perlin_mask(self, height, width, device, scale=6.0, octaves=4):
        """Generate binary mask using fast Perlin-like noise"""
        return generate_fast_perlin_mask(height, width, device, scale, octaves)
    
    def generate_foreground_mask(self, image, device):
        """
        Generate foreground mask through binarization
        """
        if image.dim() == 4:  # Batch dimension
            image = image[0]  # Take first image
        
        # Convert to grayscale if RGB
        if image.shape[0] == 3:
            grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            grayscale = image[0]
        
        # Simple Otsu-like thresholding for foreground detection
        threshold = torch.mean(grayscale)
        foreground_mask = (grayscale > threshold * 0.8).float()
        
        return foreground_mask
    
    def execute(self, image, device):
        """
        Execute Step I following Equation 3 from the paper:
        
        mi = {
            (m1 ∧ m2) ∧ mf     if 0 ≤ pm ≤ α
            (m1 ∨ m2) ∧ mf     if α < pm ≤ 2α  
            m1 ∧ mf            if 2α < pm ≤ 1
        }
        """
        if image.dim() == 4:
            _, _, height, width = image.shape
        else:
            _, height, width = image.shape
        
        # Generate two Perlin noise masks
        m1 = self.generate_perlin_mask(height, width, device)
        m2 = self.generate_perlin_mask(height, width, device)
        
        # Generate foreground mask
        mf = self.generate_foreground_mask(image, device)
        
        # Random probability for mask combination strategy
        pm = random.random()
        
        # Apply three mask combination strategies
        if 0 <= pm <= self.alpha:
            # Intersection: (m1 ∧ m2) ∧ mf
            anomaly_mask = m1 * m2 * mf
        elif self.alpha < pm <= 2 * self.alpha:
            # Union: (m1 ∨ m2) ∧ mf  
            union_mask = torch.clamp(m1 + m2, 0, 1)
            anomaly_mask = union_mask * mf
        else:
            # Single: m1 ∧ mf
            anomaly_mask = m1 * mf
        
        return anomaly_mask


class AnomalyTextureGeneration:
    """Step II: Generate anomaly textures using DTD dataset with augmentations"""
    
    def __init__(self, dtd_root="./data"):
        """Initialize with DTD dataset"""
        self.dtd_dataset = self._load_dtd_dataset(dtd_root)
        self.augmentation_methods = self._setup_augmentation_methods()
    
    def _load_dtd_dataset(self, dtd_root):
        """Load DTD (Describable Textures Dataset)"""
        try:
            dtd_dataset = DTD(
                root=dtd_root,
                download=True,
                transform=transforms.ToTensor()
            )
            print(f"DTD dataset loaded: {len(dtd_dataset)} texture images")
            return dtd_dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load DTD dataset: {e}")
    
    def _setup_augmentation_methods(self):
        """
        Setup K=9 augmentation methods
        """
        return [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # T1
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),                        # T2
            transforms.RandomRotation(degrees=45),                                           # T3
            transforms.RandomHorizontalFlip(p=1.0),                                         # T4
            transforms.RandomVerticalFlip(p=1.0),                                           # T5
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),                     # T6
            transforms.RandomAffine(degrees=0, scale=(0.7, 1.3)),                           # T7
            transforms.RandomAdjustSharpness(sharpness_factor=2.5),                         # T8
            transforms.RandomAutocontrast(p=1.0),                                           # T9
        ]
    
    def select_random_texture(self, height, width, device):
        """Randomly select and resize texture from DTD dataset"""
        # Select random texture from DTD
        idx = random.randint(0, len(self.dtd_dataset) - 1)
        texture, _ = self.dtd_dataset[idx]  # DTD returns (image, label)
        
        # Resize to target dimensions
        texture = F.interpolate(
            texture.unsqueeze(0),
            size=(height, width), 
            mode='bilinear',
            align_corners=False
        ).squeeze(0).to(device)
        
        return texture
    
    def apply_random_augmentations(self, texture, device):
        """
        Apply 3 random augmentations from the pool of K=9 methods
        
        Args:
            texture: Input texture (3, H, W)
            device: PyTorch device
            
        Returns:
            torch.Tensor: Augmented texture x''i = TR(x'i)
        """
        # Randomly select 3 methods from K=9
        selected_methods = random.sample(self.augmentation_methods, 3)
        
        augmented_texture = texture
        
        for method in selected_methods:
            try:
                # Convert to PIL for augmentation
                texture_pil = transforms.ToPILImage()(augmented_texture.cpu())
                texture_pil = method(texture_pil)
                augmented_texture = transforms.ToTensor()(texture_pil).to(device)
            except Exception:
                # Skip failed augmentations
                continue
        
        return augmented_texture
    
    def execute(self, height, width, device):
        """
        Execute Step II: Generate augmented anomaly texture
        
        Process:
        1. Randomly select texture x'i from DTD
        2. Apply 3 random augmentations: x''i = TR(x'i)
        
        Returns:
            torch.Tensor: Augmented anomaly texture x''i (3, H, W)
        """
        # Step II.1: Select random texture x'i
        texture = self.select_random_texture(height, width, device)
        
        # Step II.2: Apply random augmentations TR(x'i)  
        augmented_texture = self.apply_random_augmentations(texture, device)
        
        return augmented_texture


class OverlayFusion:
    """Step III: Overlay fusion with transparency coefficient"""
    
    def __init__(self, beta_mean=0.5, beta_std=0.1):
        """
        Args:
            beta_mean: Mean of transparency coefficient β
            beta_std: Standard deviation of transparency coefficient β
        """
        self.beta_mean = beta_mean
        self.beta_std = beta_std
    
    def sample_transparency_coefficient(self, device):
        """
        Sample transparency coefficient β∼N(μm,σ²m)
        """
        beta = torch.normal(
            torch.tensor(self.beta_mean, device=device),
            torch.tensor(self.beta_std, device=device) 
        ).clamp(0.2, 0.8).item()  # Clamp to reasonable range
        
        return beta
    
    def execute(self, normal_image, anomaly_texture, anomaly_mask, device):
        """
        Execute Step III following Equation 4 from the paper:
        
        xi+ = xi ⊙ m̄i + (1-β)x''i ⊙ mi + βxi ⊙ mi
        
        Where:
        - xi: normal training image
        - x''i: augmented anomaly texture
        - mi: anomaly mask  
        - m̄i: inverted anomaly mask (complement)
        - β: transparency coefficient
        """
        # Sample transparency coefficient β
        beta = self.sample_transparency_coefficient(device)
        
        # Ensure mask has channel dimension for broadcasting
        if anomaly_mask.dim() == 2:
            anomaly_mask = anomaly_mask.unsqueeze(0)  # (1, H, W)
        
        # Create inverted mask m̄i (complement)
        inverted_mask = 1 - anomaly_mask
        
        # Apply Equation 4: Overlay Fusion
        synthetic_image = (
            normal_image * inverted_mask +                    # xi ⊙ m̄i (preserve normal regions)
            (1 - beta) * anomaly_texture * anomaly_mask +     # (1-β)x''i ⊙ mi (texture overlay)
            beta * normal_image * anomaly_mask                # βxi ⊙ mi (blended regions)
        )
        
        return synthetic_image


class LocalAnomalySynthesis:
    """
    Complete Local Anomaly Synthesis (LAS) Implementation
    
    Three-step workflow following GLASS paper:
    Step I:   Anomaly Mask Generation  
    Step II:  Anomaly Texture Generation
    Step III: Overlay Fusion
    """
    
    def __init__(self, alpha=1/3, beta_mean=0.5, beta_std=0.1, dtd_root="./data"):
        """
        Initialize LAS with three-step workflow
        
        Args:
            alpha: Parameter for mask combination (Step I)
            beta_mean: Mean for transparency coefficient (Step III)
            beta_std: Standard deviation for transparency coefficient (Step III) 
            dtd_root: Root directory for DTD dataset (Step II)
        """
        self.step_i = AnomalyMaskGeneration(alpha=alpha)
        self.step_ii = AnomalyTextureGeneration(dtd_root=dtd_root)
        self.step_iii = OverlayFusion(beta_mean=beta_mean, beta_std=beta_std)
    
    def synthesize(self, normal_images):
        """
        Complete LAS synthesis pipeline
        
        Args:
            normal_images: Batch of normal images (B, C, H, W)
            
        Returns:
            torch.Tensor: Batch of synthetic anomaly images (B, C, H, W)
        """
        B, C, H, W = normal_images.shape
        device = normal_images.device
        synthetic_images = []
        
        for b in range(B):
            # Get current normal image
            normal_image = normal_images[b]  # (3, H, W)
            
            # Step I: Anomaly Mask Generation
            anomaly_mask = self.step_i.execute(normal_image.unsqueeze(0), device)
            
            # Step II: Anomaly Texture Generation
            anomaly_texture = self.step_ii.execute(H, W, device)
            
            # Step III: Overlay Fusion
            synthetic_image = self.step_iii.execute(
                normal_image, anomaly_texture, anomaly_mask, device
            )
            
            synthetic_images.append(synthetic_image)
        
        return torch.stack(synthetic_images)