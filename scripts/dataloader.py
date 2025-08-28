"""
Metal Parts Defect Detection (MPDD) Dataset

This module provides a PyTorch Dataset implementation for anomaly detection
with mask support, designed for training and evaluation of defect detection models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MPDD_Dataset(Dataset):
    """Dataset for metal parts defect detection with mask support."""
    
    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self, 
        root_path: str | Path, 
        category: str, 
        split: str = 'train', 
        image_size: int = 288
    ):
        self.root = Path(root_path) / category
        self.split = split
        self.image_size = image_size
        
        self.image_transform = self._create_image_transform()
        self.mask_transform = self._create_mask_transform()
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples from '{category}'")
    
    def _create_image_transform(self) -> transforms.Compose:
        """Create image preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        ])
    
    def _create_mask_transform(self) -> transforms.Compose:
        """Create mask preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size), 
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])
    
    def _load_samples(self) -> List[Tuple[Path, int, Optional[Path]]]:
        """Load image paths with labels and optional mask paths."""
        samples = []
        
        if self.split == 'train':
            samples.extend(self._load_train_samples())
        else:
            samples.extend(self._load_test_samples())
        
        return samples
    
    def _load_train_samples(self) -> List[Tuple[Path, int, None]]:
        """Load training samples (normal images only)."""
        good_path = self.root / 'train' / 'good'
        return [(img_path, 0, None) for img_path in good_path.glob('*.png')]
    
    def _load_test_samples(self) -> List[Tuple[Path, int, Optional[Path]]]:
        """Load test samples (normal and anomalous images)."""
        test_samples = []
        test_path = self.root / 'test'
        
        # Normal test images
        good_path = test_path / 'good'
        test_samples.extend([(img_path, 0, None) for img_path in good_path.glob('*.png')])
        
        # Anomalous images with masks
        for defect_dir in test_path.iterdir():
            if defect_dir.is_dir() and defect_dir.name != 'good':
                test_samples.extend(self._load_defect_samples(defect_dir))
        
        return test_samples
    
    def _load_defect_samples(self, defect_dir: Path) -> List[Tuple[Path, int, Optional[Path]]]:
        """Load samples from a specific defect directory."""
        defect_samples = []
        mask_dir = self.root / 'ground_truth' / defect_dir.name
        
        for img_path in defect_dir.glob('*.png'):
            mask_path = mask_dir / f"{img_path.stem}_mask{img_path.suffix}"
            mask_path = mask_path if mask_path.exists() else None
            defect_samples.append((img_path, 1, mask_path))
        
        return defect_samples
    
    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Load and preprocess an image."""
        image = Image.open(img_path).convert("RGB")
        return self.image_transform(image)
    
    def _load_mask(self, mask_path: Optional[Path]) -> torch.Tensor:
        """Load and preprocess a mask, or create empty mask if None."""
        if mask_path and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            return (self.mask_transform(mask) > 0.5).float()
        return torch.zeros(1, self.image_size, self.image_size)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample containing image, label, and mask."""
        img_path, label, mask_path = self.samples[idx]
        
        return {
            "image": self._load_image(img_path),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": self._load_mask(mask_path)
        }


def create_dataloaders(
    dataset_path: str | Path,
    category: str,
    batch_size: int = 8,
    image_size: int = 288,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for the MPDD dataset.
    
    Args:
        dataset_path: Path to the dataset root directory
        category: Category/class name to load
        batch_size: Batch size for dataloaders
        image_size: Size to resize images to
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = MPDD_Dataset(dataset_path, category, 'train', image_size)
    test_dataset = MPDD_Dataset(dataset_path, category, 'test', image_size)
    
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    
    return train_loader, test_loader