""" SIBI Real-Time Recognizer
This module provides functionality to recognize SIBI letters in real-time using a pre-trained model"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import os
import time
import gc
from collections import deque, Counter
from ..models.mlp import SIBIBasicMLP
from ..utils.landmarks import HandLandmarkExtractor
import mediapipe as mp

class FPSCounter:
    """Simple FPS counter with moving average"""
    
    def __init__(self, history_size=30):
        self.history_size = history_size
        self.frame_times = deque(maxlen=history_size)
        self.last_time = time.time()
        
    def update(self):
        """Update FPS calculation"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.last_time = current_time
        
    def get_fps(self):
        """Get current FPS (moving average)"""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
class SIBIRealTimeRecognizer:
    """Real-time SIBI recognition - PyTorch 2.6 Compatible"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        
        # Load model and scaler
        self.load_model()
        
        # Initialize landmark extractor
        self.landmark_extractor = HandLandmarkExtractor(static_mode=False)
        self.mp_drawing = mp.solutions.drawing_utils

        # Create MediaPipe Hands instance ONCE
        self.hands_display = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        )
        
        # SIBI letters
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        
        # FPS counter
        self.fps_counter = FPSCounter()
        self.frame_count = 0
        
        print(f"🔧 Recognition system ready! Using device: {self.device}")
    
    def load_model(self):
        """Load model and scaler from separate files"""
        model_path = os.path.join(self.model_dir, "sibi_model.pth")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}\n   Please train the model first (option 2)")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"❌ Scaler not found: {scaler_path}\n   Please train the model first (option 2)")
        
        # Load model checkpoint (secure - weights only)
        print("🔄 Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Initialize and load model
        config = checkpoint['model_config']
        self.model = SIBIBasicMLP(**config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        print("🔄 Loading scaler...")
        self.scaler = joblib.load(scaler_path)
        
        print("✅ Model and scaler loaded successfully!")
    
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
            print(f"❌ Prediction error: {e}")
            return None, 0.0
    
    def get_stable_prediction(self, letter, confidence):
        """Get stable prediction using history"""
        if confidence > 0.6:
            self.prediction_history.append(letter)
        
        if len(self.prediction_history) < 3:
            return "Detecting...", 0.0
        
        # Get most common prediction
        most_common = Counter(self.prediction_history).most_common(1)[0]
        
        if most_common[1] >= 2:
            return most_common[0], confidence
        else:
            return f"{most_common[0]}?", confidence
    
    def run(self):
        """Run real-time recognition"""
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # Reduce buffer to 1 frame
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # Lower resolution  
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🎥 SIBI Real-time Recognition Started!")
        print("📹 Camera feed active - show SIBI letters to the camera")
        print("❌ Press 'q' to quit")
        print("-" * 60)
        
        self.frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self.fps_counter.update()

                frame = cv2.flip(frame, 1)
                
                
                # Extract landmarks
                landmarks = self.landmark_extractor.extract_from_frame(frame)
                
                # Draw hand landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands_display.process(rgb_frame)
                
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
                
                current_fps = self.fps_counter.get_fps()
                height, width = frame.shape[:2]

                 # FPS display (top-right corner)
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(frame, fps_text, (width - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Frame counter (top-right, below FPS)
                frame_text = f"Frame: {self.frame_count}"
                cv2.putText(frame, frame_text, (width - 140, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('SIBI v2 Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("📹 Camera released")