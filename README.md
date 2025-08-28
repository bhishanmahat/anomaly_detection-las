# MPDD Anomaly Detection with LAS

Implementation of Local Anomaly Synthesis (LAS) for metal parts defect detection using the MPDD dataset.

## 🎯 Project Overview

This project implements an anomaly detection system for industrial quality control, specifically targeting metal parts defect detection. The implementation uses Local Anomaly Synthesis (LAS) with image-level texture overlay to distinguish between good and defective parts.

## 📊 Dataset

**MPDD (Metal Parts Defect Detection)**: https://drive.google.com/file/d/1b3dcRqTXR7LZkOEkVQ9qO_EcKzzC2EEI/view
- **Categories**: Bracket (brown, white, black), Metal plate
- **Current Implementation**: Supports all categories
- **Structure**: Train/Test splits with good/bad part classification

## 🏗️ Architecture

### Core Components
1. **Feature Extractor**: WideResNet50 backbone (frozen)
2. **Feature Adaptor**: Linear layer for domain adaptation  
3. **Discriminator**: MLP for anomaly classification
4. **LAS Module**: Local Anomaly Synthesis with three-step workflow

## 🚀 How to Run

### Installation
```bash
# Clone the repo
git clone https://github.com/bhishanmahat/anomaly_detection-las.git
cd anomaly_detection-las

# Create & activate a virtualenv
conda create -n env_anom-detect python=3.13
conda activate env_anom-detect

# Install PyTorch based on your hardware; for NVIDIA CUDA 12.8 on Linux, use 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install the packages
pip install -r requirements.txt

```

### Dataset Setup
1. Download MPDD dataset from: https://drive.google.com/file/d/1b3dcRqTXR7LZkOEkVQ9qO_EcKzzC2EEI/view
2. In your project root, create `data/`
3. Unzip the dataset into `data/` to match the following structure:
```
├── data/
    ├── anomaly_dataset/
        ├── bracket_white/
        │   ├── ground_truth/
        |   |   ├── defective_painting/
        |   |   ├── scratches/ 
        │   ├── test
        |   |   ├── defective_painting/
        |   |   ├── good/
        |   |   ├── scratches/ 
        │   └── train/good/
        ├── bracket_brown/
        ├── bracket_black/
        └── metal_plate/
```

### DTD Dataset Setup
LAS requires the DTD (Describable Textures Dataset) for anomaly texture generation:
```bash
# The DTD dataset will be automatically downloaded on first run
# Default location: ./data/dtd/
```

### Training
```bash
python scripts/main.py
```

### Configuration
Modify settings in `main.py`:
```python
DATASET_PATH = "./data/anomaly_dataset"
CATEGORY = "bracket_white"  # Choose: bracket_white, bracket_brown, bracket_black, metal_plate
RESULTS_DIR = "./results/las"
BATCH_SIZE = 8
IMAGE_SIZE = 288
EPOCHS = 50
```

## 📁 File Structure

```
├── data/anomaly_dataset/
├── scripts/
|   ├── main.py              # Entry point and configuration
|   ├── trainer.py           # Training loop and model management
|   ├── las.py               # Local Anomaly Synthesis implementation
|   ├── models.py            # Neural network architectures
|   ├── datloader.py         # Data loading and preprocessing
|   ├── visualize.py         # Plotting and visualization
├── results/                 # Output directory for models and plots
├── requirements.txt         # Required packages
```

## 🔧 Model Details

### Hyperparameters
- **Learning Rate**: 0.0001 (Feature Adaptor), 0.0002 (Discriminator)
- **LAS Parameters**:
  - `alpha = 1/3` (mask combination parameter)
  - `beta ~ N(0.5, 0.1²)` (transparency coefficient)
- **Image Size**: 288×288
- **Feature Dimensions**: 1536 (WideResNet50 layer2 + layer3)

## 🔬 Technical Implementation

### Local Anomaly Synthesis (GAS)
1. **Step I: Anomaly Mask Generation**
  - Generate two Perlin noise masks (m1, m2)
  - Create foreground mask (mf) through binarization
  - Apply mask combination strategy based on random probability:
    - Intersection: `(m1 ∧ m2) ∧ mf`
    - Union: `(m1 ∨ m2) ∧ mf`
    - Single: `m1 ∧ mf`
2. **Step II: Anomaly Texture Generation**
  - Randomly select texture from DTD dataset
  - Apply 3 random augmentations from 9 methods:
    - Color jittering, Gaussian blur, rotation
    - Horizontal/vertical flips, translation, scaling
    - Sharpness adjustment, auto-contrast
3. **Step III: Overlay Fusion**
  - Blend normal image with anomaly texture using transparency coefficient β
  - Formula: `x+ = x ⊙ m̄ + (1-β)x'' ⊙ m + βx ⊙ m`

### Key Features
- **Fast Vectorized Operations**: Optimized Perlin noise generation
- **Comprehensive Timing**: Per-epoch and total training time tracking
- **Best Model Saving**: Automatically saves highest AUC checkpoint
- **Focal Loss**: Handles class imbalance in anomaly regions
- **Memory Efficient**: GPU-accelerated tensor operations

## 📈 Output

After training, the following files will be generated in `results/las`:
- `training_curves_{category}.png` - Loss and AUC plots
- `roc_curve_{category}.png` - ROC curve with AUC score
- `best_model_{category}.pth` - Best performing model checkpoint
- `training_log_{category}_LAS_{timestamp}.txt` - Detailed training log with timing

## 📚 References

- **GLASS Paper**: "A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization" - https://arxiv.org/abs/2407.09359
- **MPDD Dataset**: https://github.com/stepanje/MPDD
- **DTD Dataset**: Describing Textures in the Wild" - https://www.robots.ox.ac.uk/~vgg/data/dtd/
- **WideResNet**: "Wide Residual Networks" (Zagoruyko & Komodakis, 2016)
