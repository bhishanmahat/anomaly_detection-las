import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_training_curves(train_losses, normal_losses, las_losses, val_aucs, save_path):
    """Plot training curves for LAS-only training"""
    epochs = np.arange(1, len(train_losses) + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2)
    ax1.set_title('Total Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Individual losses
    ax2.plot(epochs, normal_losses, 'g-', linewidth=2, label='Normal')
    ax2.plot(epochs, las_losses, 'm-', linewidth=2, label='LAS')
    ax2.set_title('Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Validation AUC
    if val_aucs:
        ax3.plot(epochs, val_aucs, 'purple', linewidth=2, marker='o', markersize=3)
        ax3.set_title('Validation ROC AUC')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ROC AUC')
        ax3.set_ylim([0.5, 1.0])
        ax3.grid(True, alpha=0.3)
        
        # Best AUC annotation
        best_epoch = np.argmax(val_aucs)
        best_auc = val_aucs[best_epoch]
        ax3.annotate(f'Best: {best_auc:.4f}', 
                    xy=(best_epoch + 1, best_auc), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


def plot_roc_curve(trainer, test_loader, save_path):
    """Plot ROC curve for final model"""
    trainer.feature_adaptor.eval()
    trainer.discriminator.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(trainer.device)
            labels = batch['label'].cpu().numpy()
            
            features = trainer.extract_features(images)
            scores = trainer.discriminator(features)
            scores_np = scores.view(scores.size(0), -1).max(dim=1)[0].cpu().numpy()
            
            all_scores.extend(scores_np)
            all_labels.extend(labels)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'LAS (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - LAS Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved: {save_path}")


def visualize_results(trainer, test_loader, train_losses, normal_losses, 
                     las_losses, val_aucs, results_dir, category):
    """Generate all visualizations for LAS training"""
    os.makedirs(results_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Training curves
    curves_path = os.path.join(results_dir, f"training_curves_{category}.png")
    plot_training_curves(train_losses, normal_losses, las_losses, val_aucs, curves_path)
    
    # ROC curve
    roc_path = os.path.join(results_dir, f"roc_curve_{category}.png")
    plot_roc_curve(trainer, test_loader, roc_path)
    
    print("Visualizations completed!")