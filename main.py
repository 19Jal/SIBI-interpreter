"""
SIBI Recognition System v2 - Interactive Main Interface
"""

import sys
import os

# Add src to Python path so we can import our package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.sibi import SIBIDatasetProcessor, SIBITrainer, SIBIRealTimeRecognizer, SIBIBasicMLP
from src.sibi.utils.visualization import show_dataset_statistics

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print(" 🇮🇩  SIBI Recognition System v2.0  🤖")
    print("    Indonesian Sign Language Recognition")
    print("=" * 70)

def print_menu():
    """Print main menu options"""
    print("\nWhat would you like to do?")
    print("-" * 40)
    print("1️. Process Kaggle dataset (extract landmarks)")
    print("2️. Train deep learning model") 
    print("3️. Run real-time recognition")
    print("4️. Show dataset statistics")
    print("Press 'q' to quit")
    print("-" * 40)

def option_1_process_data():
    """Option 1: Process Kaggle dataset"""
    print("\nOption 1: Process Kaggle Dataset")
    print("=" * 50)
    
    dataset_root = input("Enter path to dataset root (with train/ and val/ folders): ").strip()
    
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset path not found: {dataset_root}")
        return
    
    # Check for train and val folders
    train_path = os.path.join(dataset_root, "train")
    val_path = os.path.join(dataset_root, "val")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: train/ and val/ folders not found in dataset root")
        print(f"   Expected structure: {dataset_root}/train/ and {dataset_root}/val/")
        return
    
    try:
        print(f"Processing dataset from: {dataset_root}")
        processor = SIBIDatasetProcessor(dataset_root, output_dir="data/processed")
        processor.process_dataset()
        print("Dataset processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")

def option_2_train_model():
    """Option 2: Train deep learning model"""
    print("\nOption 2: Train Deep Learning Model")
    print("=" * 50)
    
    try:
        # Get training parameters
        print("Configure training parameters (press Enter for defaults):")
        
        epochs_input = input(f"    Number of epochs (default: 100): ").strip()
        epochs = int(epochs_input) if epochs_input else 100
        
        lr_input = input(f"    Learning rate (default: 0.001): ").strip()
        lr = float(lr_input) if lr_input else 0.001
        
        batch_input = input(f"    Batch size (default: 32): ").strip()
        batch_size = int(batch_input) if batch_input else 32
        
        print(f"\n Training configuration:")
        print(f"    Epochs: {epochs}")
        print(f"    Learning rate: {lr}")
        print(f"    Batch size: {batch_size}")
        
        # Initialize model and trainer
        print("\n Initializing model...")
        model = SIBIBasicMLP(input_size=63, num_classes=26)
        trainer = SIBITrainer(model)
        
        # Train model
        print(" Starting training...")
        best_acc = trainer.train(
            data_dir="data/processed",
            num_epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size
        )
        
        print(f"\n Training completed!")
        print(f"    Best validation accuracy: {best_acc:.2f}%")
        print(f"    Model saved to: data/models/")
        
    except FileNotFoundError as e:
        print(f" {e}")
    except ValueError as e:
        print(f" Invalid input: {e}")
    except Exception as e:
        print(f" Error during training: {e}")

def option_3_run_recognition():
    """Option 3: Run real-time recognition"""
    print("\n Option 3: Real-time Recognition")
    print("=" * 50)
    
    try:
        print(" Initializing recognition system...")
        recognizer = SIBIRealTimeRecognizer(model_dir="data/models")
        
        print(" System ready!")
        input(" Press Enter to start camera (make sure your camera is working)...")
        
        recognizer.run()
        
    except FileNotFoundError as e:
        print(f" {e}")
    except Exception as e:
        print(f" Error during recognition: {e}")

def option_4_show_stats():
    """Option 4: Show dataset statistics"""
    print("\n Option 4: Dataset Statistics")
    print("=" * 50)
    
    show_dataset_statistics("data/processed")

def main():
    """Main interactive loop"""
    print_banner()
    
    while True:
        print_menu()
        
        choice = input("👉 Enter your choice (1-4 or 'q'): ").strip().lower()
        
        if choice == '1':
            option_1_process_data()
            
        elif choice == '2':
            option_2_train_model()
            
        elif choice == '3':
            option_3_run_recognition()
            
        elif choice == '4':   
            option_4_show_stats()
            
        elif choice == 'q' or choice == 'quit':
            print("\n👋 Thank you for using SIBI Recognition System v2!")
            print("Goodbye!")
            break
            
        else:
            print(" Invalid choice. Please enter 1, 2, 3, 4, or 'q'")
        
        # Wait before showing menu again
        input("\n  Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Program interrupted by user")
        print(" Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        print(" Please report this issue")
        sys.exit(1)