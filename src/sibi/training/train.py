""" SIBI Model Trainer
This module provides functionality to train a Multi-Layer Perceptron (MLP) model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
from typing import Tuple
from ..utils.visualization import plot_training_history

class SIBITrainer:
    """SIBI Model Trainer"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.scaler = StandardScaler()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        print(f"🔧 Using device: {self.device}")
    
    def load_data(self, data_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load processed training data"""
        print("📂 Loading processed data...")
        
        try:
            X_train = np.load(os.path.join(data_dir, "train_landmarks.npy"))
            y_train = np.load(os.path.join(data_dir, "train_labels.npy"))
            X_val = np.load(os.path.join(data_dir, "val_landmarks.npy"))
            y_val = np.load(os.path.join(data_dir, "val_labels.npy"))
            
            print(f"   ✅ Training: {X_train.shape[0]} samples")
            print(f"   ✅ Validation: {X_val.shape[0]} samples")
            print(f"   ✅ Features: {X_train.shape[1]}")
            
            return X_train, y_train, X_val, y_val
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ Data not found: {e}\n   Please process dataset first (option 1)")
    
    def prepare_data(self, X_train, y_train, X_val, y_val, batch_size: int = 32):
        """Prepare data loaders"""
        print("🔄 Preparing data loaders...")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.LongTensor(y_val)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"   ✅ Batch size: {batch_size}")
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(self.train_loader), 100 * correct / total
    
    def validate(self, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(self.val_loader), 100 * correct / total
    
    def train(self, data_dir: str = "data/processed", num_epochs: int = 100, 
              learning_rate: float = 0.001, batch_size: int = 32) -> float:
        """Main training loop"""
        
        # Load data
        X_train, y_train, X_val, y_val = self.load_data(data_dir)
        
        # Prepare data
        self.prepare_data(X_train, y_train, X_val, y_val, batch_size)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        best_val_acc = 0.0
        
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"   📊 Learning rate: {learning_rate}")
        print(f"   📦 Batch size: {batch_size}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            val_loss, val_acc = self.validate(criterion)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                      f'Train: {train_acc:5.1f}% ({train_loss:.4f}) | '
                      f'Val: {val_acc:5.1f}% ({val_loss:.4f})')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
                
                if (epoch + 1) % 10 == 0:
                    print(f'   🎉 New best model! Validation accuracy: {val_acc:.2f}%')
            
            scheduler.step()
        
        print("-" * 60)
        print(f'🎯 Training completed! Best validation accuracy: {best_val_acc:.2f}%')
        
        # Plot results
        plot_training_history(self.history)
        
        return best_val_acc
    
    def save_model(self, model_dir: str = "data/models"):
        """Save model and scaler separately (PyTorch 2.6 compatible)"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model checkpoint (weights_only compatible)
        model_path = os.path.join(model_dir, "sibi_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'history': self.history
        }, model_path)
        
        # Save scaler separately
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # print(f"💾 Model saved: {model_path}")
        # print(f"💾 Scaler saved: {scaler_path}")