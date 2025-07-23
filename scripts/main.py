import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import json
from collections import deque, Counter
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm

class SIBIBasicModel(nn.Module):
    """
    Simple Multi-Layer Perceptron for SIBI letter recognition
    Input: 63 features (21 hand landmarks × 3 coordinates)
    Output: 26 classes (A-Z)
    """
    
    def __init__(self, input_size=63, num_classes=26, hidden_sizes=[128, 64, 32]):
        super(SIBIBasicModel, self).__init__()
        
        # Create layers dynamically
        layers = []
        prev_size = input_size
        
        # Hidden layers with ReLU activation and dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class HandLandmarkExtractor:
    """Extract and normalize hand landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # Important: True for static images
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lower threshold for static images
            min_tracking_confidence=0.5
        )
        
    def extract_landmarks_from_image(self, image_path):
        """
        Extract normalized hand landmarks from image file
        Returns: numpy array of shape (63,) or None if no hand detected
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return None
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                # Extract coordinates and normalize
                coords = []
                for landmark in landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                return np.array(coords, dtype=np.float32)
            
            return None
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def extract_landmarks_from_frame(self, frame):
        """
        Extract normalized hand landmarks from video frame (for real-time use)
        Returns: numpy array of shape (63,) or None if no hand detected
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use different settings for real-time
            hands_realtime = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.6
            )
            
            results = hands_realtime.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                coords = []
                for landmark in landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                return np.array(coords, dtype=np.float32)
            
            return None
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

class SIBIImageDatasetProcessor:
    """Process Kaggle SIBI image dataset to extract landmarks"""
    
    def __init__(self, dataset_root, output_dir="processed_sibi_data"):
        """
        Initialize dataset processor
        
        Args:
            dataset_root: Path to dataset root containing train/ and val/ folders
            output_dir: Directory to save processed landmarks
        """
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.landmark_extractor = HandLandmarkExtractor()
        
        # SIBI letters A-Z (assuming folders are named A, B, C, ...)
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.letters)}
        
    def process_dataset(self):
        """Process the entire dataset and extract landmarks"""
        print("Processing SIBI image dataset...")
        
        # Process training data
        print("Processing training data...")
        train_landmarks, train_labels = self._process_split("train")
        
        # Process validation data
        print("Processing validation data...")
        val_landmarks, val_labels = self._process_split("val")
        
        if len(train_landmarks) == 0 or len(val_landmarks) == 0:
            raise ValueError("No landmarks extracted! Check your dataset structure and image quality.")
        
        # Save processed data
        self._save_processed_data(train_landmarks, train_labels, val_landmarks, val_labels)
        
        print(f"Dataset processing completed!")
        print(f"Training samples: {len(train_landmarks)}")
        print(f"Validation samples: {len(val_landmarks)}")
        
        return train_landmarks, train_labels, val_landmarks, val_labels
    
    def _process_split(self, split_name):
        """Process a single split (train or val)"""
        split_path = os.path.join(self.dataset_root, split_name)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        
        landmarks_list = []
        labels_list = []
        
        # Process each letter folder
        for letter in self.letters:
            letter_path = os.path.join(split_path, letter)
            
            if not os.path.exists(letter_path):
                print(f"Warning: Letter directory not found: {letter_path}")
                continue
            
            # Get all image files in the letter directory
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(letter_path, ext)))
                image_files.extend(glob.glob(os.path.join(letter_path, ext.upper())))
            
            print(f"Processing {len(image_files)} images for letter {letter}...")
            
            # Extract landmarks from each image
            successful_extractions = 0
            for image_path in tqdm(image_files, desc=f"Letter {letter}"):
                landmarks = self.landmark_extractor.extract_landmarks_from_image(image_path)
                
                if landmarks is not None:
                    landmarks_list.append(landmarks)
                    labels_list.append(self.letter_to_idx[letter])
                    successful_extractions += 1
            
            print(f"Successfully extracted landmarks from {successful_extractions}/{len(image_files)} images for letter {letter}")
        
        return np.array(landmarks_list), np.array(labels_list)
    
    def _save_processed_data(self, train_X, train_y, val_X, val_y):
        """Save processed landmarks to files"""
        # Save training data
        np.save(os.path.join(self.output_dir, "train_landmarks.npy"), train_X)
        np.save(os.path.join(self.output_dir, "train_labels.npy"), train_y)
        
        # Save validation data
        np.save(os.path.join(self.output_dir, "val_landmarks.npy"), val_X)
        np.save(os.path.join(self.output_dir, "val_labels.npy"), val_y)
        
        # Save letter mapping
        letter_mapping = {i: letter for i, letter in enumerate(self.letters)}
        with open(os.path.join(self.output_dir, "letter_mapping.json"), 'w') as f:
            json.dump(letter_mapping, f)
        
        # Save dataset statistics
        stats = {
            "total_train_samples": len(train_X),
            "total_val_samples": len(val_X),
            "num_classes": len(self.letters),
            "feature_dim": 63,
            "letters": self.letters
        }
        
        with open(os.path.join(self.output_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Processed data saved to {self.output_dir}/")

class SIBITrainer:
    """Train the basic SIBI model using processed landmarks"""
    
    def __init__(self, data_dir="processed_sibi_data", model_save_path="sibi_basic_model.pth"):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load processed data
        self.load_data()
        
        # Initialize model
        self.model = SIBIBasicModel(input_size=63, num_classes=26)
        self.model.to(self.device)
        
        # Initialize scaler for normalization
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load processed training data"""
        train_landmarks_path = os.path.join(self.data_dir, "train_landmarks.npy")
        train_labels_path = os.path.join(self.data_dir, "train_labels.npy")
        val_landmarks_path = os.path.join(self.data_dir, "val_landmarks.npy")
        val_labels_path = os.path.join(self.data_dir, "val_labels.npy")
        
        if not all(os.path.exists(p) for p in [train_landmarks_path, train_labels_path, 
                                               val_landmarks_path, val_labels_path]):
            raise FileNotFoundError("Processed data not found. Please process the dataset first.")
        
        self.X_train = np.load(train_landmarks_path)
        self.y_train = np.load(train_labels_path)
        self.X_val = np.load(val_landmarks_path)
        self.y_val = np.load(val_labels_path)
        
        print(f"Loaded training data: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"Loaded validation data: {self.X_val.shape[0]} samples, {self.X_val.shape[1]} features")
        print(f"Training classes: {np.unique(self.y_train)}")
        print(f"Validation classes: {np.unique(self.y_val)}")
    
    def prepare_data(self, batch_size=32):
        """Prepare training and validation data loaders"""
        # Normalize features using training data statistics
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train_scaled)
        y_train_tensor = torch.LongTensor(self.y_train)
        X_val_tensor = torch.FloatTensor(self.X_val_scaled)
        y_val_tensor = torch.LongTensor(self.y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Prepared data loaders with batch size: {batch_size}")
    
    def train(self, num_epochs=100, learning_rate=0.001, batch_size=32):
        """Train the model"""
        self.prepare_data(batch_size)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0.0
        
        print("Starting training...")
        print(f"Training for {num_epochs} epochs with learning rate {learning_rate}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            
            # Store history
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
                print('-' * 50)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
                if (epoch + 1) % 10 == 0:
                    print(f'New best model saved! Validation accuracy: {val_acc:.2f}%')
            
            scheduler.step()
        
        print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return best_val_acc
    
    def save_model(self):
        """Save model and scaler"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'model_config': {
                'input_size': 63,
                'num_classes': 26,
                'hidden_sizes': [128, 64, 32]
            }
        }, self.model_save_path)
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Training Loss', linewidth=2)
        ax1.plot(val_losses, label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(train_accs, label='Training Accuracy', linewidth=2)
        ax2.plot(val_accs, label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sibi_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Training history plot saved as 'sibi_training_history.png'")

class SIBIRealTimeRecognizer:
    """Real-time SIBI recognition using trained model"""
    
    def __init__(self, model_path="sibi_basic_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.load_model(model_path)
        
        # Initialize landmark extractor
        self.landmark_extractor = HandLandmarkExtractor()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # SIBI letters
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        
    def load_model(self, model_path):
        """Load model with secure globals allowlist"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # FIX: Add safe globals for StandardScaler
        torch.serialization.add_safe_globals([
            'sklearn.preprocessing._data.StandardScaler',
            'sklearn.preprocessing.StandardScaler'  # For older sklearn versions
        ])
        
        try:
            # Now we can load with weights_only=True (secure)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Secure loading failed: {e}")
            print("Falling back to weights_only=False")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Rest of the loading code...
        config = checkpoint['model_config']
        self.model = SIBIBasicModel(
            input_size=config['input_size'],
            num_classes=config['num_classes'],
            hidden_sizes=config['hidden_sizes']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.scaler = checkpoint['scaler']
        print("Model loaded successfully!")
    
    def predict(self, landmarks):
        """Predict SIBI letter from landmarks"""
        if landmarks is None:
            return None, 0.0
        
        try:
            # Normalize landmarks
            landmarks_normalized = self.scaler.transform(landmarks.reshape(1, -1))
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(landmarks_normalized).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_letter = self.letters[predicted.item()]
            confidence_score = confidence.item()
            
            return predicted_letter, confidence_score
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def get_stable_prediction(self, letter, confidence):
        """Get stable prediction using history"""
        if confidence > 0.6:
            self.prediction_history.append(letter)
        
        if len(self.prediction_history) < 3:
            return "Detecting...", 0.0
        
        # Get most common prediction
        most_common = Counter(self.prediction_history).most_common(1)[0]
        
        if most_common[1] >= 2:  # Appears at least 2 times
            return most_common[0], confidence
        else:
            return f"{most_common[0]}?", confidence
    
    def run(self):
        """Run real-time recognition"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("SIBI Basic Deep Learning Recognition Started!")
        print("Trained on Kaggle SIBI dataset")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks = self.landmark_extractor.extract_landmarks_from_frame(frame)
                
                # Draw hand landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands_display = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.6
                )
                results = hands_display.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    
                    # Predict letter
                    predicted_letter, confidence = self.predict(landmarks)
                    
                    if predicted_letter:
                        stable_letter, stable_confidence = self.get_stable_prediction(
                            predicted_letter, confidence)
                        
                        # Display prediction
                        color = (0, 255, 0) if stable_confidence > 0.7 else (0, 255, 255)
                        cv2.putText(frame, f"SIBI: {stable_letter}", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                        cv2.putText(frame, f"Confidence: {stable_confidence:.2f}", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Processing...", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.prediction_history.clear()
                
                # Instructions
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('SIBI Basic Deep Learning Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function to run different modes"""
    print("SIBI Basic Deep Learning System - Kaggle Dataset Version")
    print("=" * 50)
    print("1. Process Kaggle dataset (extract landmarks from images)")
    print("2. Train model on processed data")
    print("3. Run real-time recognition")
    print("4. Show dataset statistics")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        # Process Kaggle dataset
        dataset_root = input("Enter path to dataset root (containing train/ and val/ folders): ").strip()
        
        if not os.path.exists(dataset_root):
            print(f"Error: Dataset path not found: {dataset_root}")
            return
        
        try:
            processor = SIBIImageDatasetProcessor(dataset_root)
            processor.process_dataset()
            print("Dataset processing completed successfully!")
        except Exception as e:
            print(f"Error processing dataset: {e}")
        
    elif choice == "2":
        # Train model
        try:
            trainer = SIBITrainer()
            
            # Get training parameters
            epochs = input("Enter number of epochs (default: 100): ").strip()
            epochs = int(epochs) if epochs else 100
            
            lr = input("Enter learning rate (default: 0.001): ").strip()
            lr = float(lr) if lr else 0.001
            
            batch_size = input("Enter batch size (default: 32): ").strip()
            batch_size = int(batch_size) if batch_size else 32
            
            print(f"Training with: {epochs} epochs, LR: {lr}, Batch size: {batch_size}")
            
            best_acc = trainer.train(num_epochs=epochs, learning_rate=lr, batch_size=batch_size)
            print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please process the dataset first (option 1)")
        except Exception as e:
            print(f"Error during training: {e}")
    
    elif choice == "3":
        # Run real-time recognition
        try:
            recognizer = SIBIRealTimeRecognizer()
            recognizer.run()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please train the model first (option 2)")
        except Exception as e:
            print(f"Error during recognition: {e}")
    
    elif choice == "4":
        # Show dataset statistics
        stats_path = "processed_sibi_data/dataset_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            print("Dataset Statistics:")
            print(f"Total training samples: {stats['total_train_samples']}")
            print(f"Total validation samples: {stats['total_val_samples']}")
            print(f"Number of classes: {stats['num_classes']}")
            print(f"Feature dimension: {stats['feature_dim']}")
            print(f"Letters: {', '.join(stats['letters'])}")
        else:
            print("No dataset statistics found. Please process the dataset first.")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    # Required packages:
    # pip install torch torchvision opencv-python mediapipe numpy scikit-learn matplotlib pillow tqdm
    
    main()