"""Hand Landmark Extraction Module
This module provides functionality to extract hand landmarks from images or video frames using MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Union

class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self, static_mode: bool = True):
        self.mp_hands = mp.solutions.hands
        self.static_mode = static_mode
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=1,
            min_detection_confidence=0.5 if static_mode else 0.8,
            min_tracking_confidence=0.5 if static_mode else 0.6
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """Extract landmarks from image file"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                coords = []
                for landmark in landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                return np.array(coords, dtype=np.float32)
            
            return None
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract landmarks from video frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
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