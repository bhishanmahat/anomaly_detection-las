import os
import torch

# Import modules
from models import FeatureExtractor, FeatureAdaptor, Discriminator
from dataloader import create_dataloaders
from las import LocalAnomalySynthesis
from trainer import GLASSTrainer
from visualize import visualize_results


def main():
    """Main training script for LAS-only anomaly detection"""
    
    # Configuration
    DATASET_PATH = "./data/anomaly_dataset"
    CATEGORY = "bracket_white"
    RESULTS_DIR = "./results/las"
    BATCH_SIZE = 8
    IMAGE_SIZE = 288
    EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    print(f"Category: {CATEGORY}")
    print("Training mode: LAS-only")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        DATASET_PATH, CATEGORY, BATCH_SIZE, IMAGE_SIZE
    )
    
    # Initialize model components
    feature_extractor = FeatureExtractor()
    feature_adaptor = FeatureAdaptor()
    discriminator = Discriminator()
    
    # Initialize LAS
    las = LocalAnomalySynthesis(
        alpha=1/3, beta_mean=0.5, beta_std=0.1
    )
    print("LAS initialized")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize trainer
    trainer = GLASSTrainer(
        feature_extractor=feature_extractor,
        feature_adaptor=feature_adaptor,
        discriminator=discriminator,
        las=las,
        device=DEVICE
    )
    
    # Train model
    train_losses, normal_losses, las_losses, val_aucs = trainer.train(
        train_loader, test_loader, epochs=EPOCHS, 
        save_dir=RESULTS_DIR, category=CATEGORY
    )
    
    # Load best model for visualization
    if not trainer.load_model(RESULTS_DIR, CATEGORY):
        print("Warning: No saved model found, using final epoch weights")
    
    # Generate visualizations
    visualize_results(
        trainer, test_loader, train_losses, normal_losses, 
        las_losses, val_aucs, RESULTS_DIR, CATEGORY
    )
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"Mode: LAS-only")
    print(f"Best AUC: {trainer.best_auc:.4f} at epoch {trainer.best_epoch}")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()