import matplotlib.pyplot as plt
import json
import os

def plot_training_history(history, save_path="training_history.png"):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2, color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2, color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2, color='red')
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training history plot saved: {save_path}")

def show_dataset_statistics(data_dir="data/processed"):
    """Show dataset statistics"""
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print("❌ Dataset statistics not found. Please process dataset first.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("📊 Dataset Statistics")
    print("=" * 50)
    print(f"📚 Training samples:   {metadata['train_samples']:,}")
    print(f"📊 Validation samples: {metadata['val_samples']:,}")
    print(f"🔢 Total samples:      {metadata['train_samples'] + metadata['val_samples']:,}")
    print(f"🎯 Number of classes:  {metadata['num_classes']}")
    print(f"📐 Feature dimension:  {metadata['feature_dim']}")
    print(f"🔤 Letters: {', '.join(metadata['letters'])}")
    print("=" * 50)