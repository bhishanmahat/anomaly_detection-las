import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import datetime
import time


class GLASSTrainer:
    """LAS-only trainer for anomaly detection"""
    
    def __init__(self, feature_extractor, feature_adaptor, discriminator, las, device='cuda'):
        self.device = device
        self.feature_extractor = feature_extractor.to(device).eval()
        self.feature_adaptor = feature_adaptor.to(device)
        self.discriminator = discriminator.to(device)
        self.las = las
        
        # Optimizers
        self.adaptor_optimizer = optim.Adam(self.feature_adaptor.parameters(), lr=0.0001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.focal_loss = self._create_focal_loss()
        
        # Tracking
        self.best_auc = 0.0
        self.best_epoch = 0
        self.log_file = None
    
    def _create_focal_loss(self, alpha=1.0, gamma=2.0):
        """Focal Loss for handling imbalanced data"""
        def focal_loss(inputs, targets):
            bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-bce)
            return (alpha * (1 - pt) ** gamma * bce).mean()
        return focal_loss
    
    def extract_features(self, images):
        """Extract and adapt features from images"""
        with torch.no_grad():
            raw_features = self.feature_extractor(images)
        return self.feature_adaptor(raw_features)
    
    def train_epoch(self, train_loader):
        """Train for one epoch with LAS only"""
        self.feature_adaptor.train()
        self.discriminator.train()
        
        epoch_losses = {'total': 0, 'normal': 0, 'las': 0}
        num_batches = 0
        
        for batch in train_loader:
            images = batch['image'].to(self.device)
            
            # Normal branch
            normal_features = self.extract_features(images)
            normal_scores = self.discriminator(normal_features)
            normal_labels = torch.zeros_like(normal_scores.view(-1))
            normal_loss = self.bce_loss(normal_scores.view(-1), normal_labels)
            
            # LAS branch
            las_images = self.las.synthesize(images)
            las_features = self.extract_features(las_images)
            las_scores = self.discriminator(las_features)
            las_labels = torch.ones_like(las_scores.view(-1))
            las_loss = self.focal_loss(las_scores.view(-1), las_labels)
            
            # Total loss: L = L_n + L_las
            total_loss = normal_loss + las_loss
            
            # Optimization step
            self.adaptor_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            total_loss.backward()
            self.adaptor_optimizer.step()
            self.discriminator_optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['normal'] += normal_loss.item()
            epoch_losses['las'] += las_loss.item()
            num_batches += 1
        
        return {k: v/num_batches for k, v in epoch_losses.items()}
    
    def evaluate(self, test_loader):
        """Evaluate model and return ROC AUC"""
        self.feature_adaptor.eval()
        self.discriminator.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                
                features = self.extract_features(images)
                scores = self.discriminator(features)
                scores_np = scores.view(scores.size(0), -1).max(dim=1)[0].cpu().numpy()
                
                all_scores.extend(scores_np)
                all_labels.extend(labels)
        
        return roc_auc_score(all_labels, all_scores)
    
    def _setup_logging(self, save_dir, category):
        """Setup logging"""
        log_filename = f"training_log_{category}_LAS_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_file = os.path.join(save_dir, log_filename)
        
        with open(self.log_file, 'w') as f:
            f.write(f"LAS Training Log - {category}\n")
            f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
            f.write("Epoch,Total_Loss,Normal_Loss,LAS_Loss,AUC,Epoch_Time(s)\n")
    
    def train(self, train_loader, test_loader, epochs=100, save_dir="./results", category=""):
        """Complete training loop with timing"""
        os.makedirs(save_dir, exist_ok=True)
        self._setup_logging(save_dir, category)
        
        print(f"Training LAS for {epochs} epochs...")
        
        # Start total training timer
        total_start_time = time.time()
        
        train_losses, normal_losses, las_losses, val_aucs = [], [], [], []
        
        for epoch in range(epochs):
            # Time individual epoch
            epoch_start_time = time.time()
            
            # Train and evaluate
            losses = self.train_epoch(train_loader)
            auc = self.evaluate(test_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # Store metrics
            train_losses.append(losses['total'])
            normal_losses.append(losses['normal'])
            las_losses.append(losses['las'])
            val_aucs.append(auc)
            
            # Print progress with timing
            print(f"Epoch {epoch+1}/{epochs} - Total: {losses['total']:.4f}, "
                  f"Normal: {losses['normal']:.4f}, LAS: {losses['las']:.4f}, "
                  f"AUC: {auc:.4f}, Time: {epoch_time:.1f}s")
            
            # Log to file with timing
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch+1},{losses['total']:.4f},{losses['normal']:.4f},"
                       f"{losses['las']:.4f},{auc:.4f},{epoch_time:.1f}\n")
            
            # Save best model
            if auc > self.best_auc:
                self.best_auc = auc
                self.best_epoch = epoch + 1
                self.save_model(save_dir, category)
        
        # Calculate and display total training time
        total_time = time.time() - total_start_time
        avg_epoch_time = total_time / epochs
        
        # Write final summary to log file
        with open(self.log_file, 'a') as f:
            f.write("-" * 50 + "\n")
            f.write(f"Training completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
            f.write(f"Average time per epoch: {avg_epoch_time:.1f}s\n")
            f.write(f"Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch}\n")
        
        print(f"\nTraining completed!")
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average time per epoch: {avg_epoch_time:.1f}s")
        print(f"Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch}")
        
        return train_losses, normal_losses, las_losses, val_aucs
    
    def save_model(self, save_dir, category):
        """Save model checkpoint"""
        model_path = os.path.join(save_dir, f"best_model_{category}.pth")
        torch.save({
            'feature_adaptor_state_dict': self.feature_adaptor.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
        }, model_path)
    
    def load_model(self, save_dir, category):
        """Load model checkpoint"""
        model_path = os.path.join(save_dir, f"best_model_{category}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.feature_adaptor.load_state_dict(checkpoint['feature_adaptor_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            print(f"Loaded model from epoch {checkpoint['best_epoch']} (AUC: {checkpoint['best_auc']:.4f})")
            return True
        return False