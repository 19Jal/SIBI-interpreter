""" SIBI Dataset Processor
This module processes the SIBI image dataset to extract hand landmarks and save them"""

import os
import numpy as np
import glob
import json
from tqdm import tqdm
from typing import Tuple
from ..utils.landmarks import HandLandmarkExtractor

class SIBIDatasetProcessor:
    """Process SIBI image dataset to extract landmarks"""
    
    def __init__(self, dataset_root: str, output_dir: str = "data/processed"):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.landmark_extractor = HandLandmarkExtractor(static_mode=True)
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.letters)}
        
    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process the entire dataset"""
        print("🔄 Processing SIBI image dataset...")
        
        # Process training data
        print("📚 Processing training split...")
        train_landmarks, train_labels = self._process_split("train")
        
        # Process validation data  
        print("📊 Processing validation split...")
        val_landmarks, val_labels = self._process_split("val")
        
        if len(train_landmarks) == 0 or len(val_landmarks) == 0:
            raise ValueError("❌ No landmarks extracted! Check dataset structure.")
        
        # Save processed data
        self._save_processed_data(train_landmarks, train_labels, val_landmarks, val_labels)
        
        print(f"✅ Dataset processing completed!")
        print(f"   📚 Training samples: {len(train_landmarks)}")
        print(f"   📊 Validation samples: {len(val_landmarks)}")
        
        return train_landmarks, train_labels, val_landmarks, val_labels
    
    def _process_split(self, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single split"""
        split_path = os.path.join(self.dataset_root, split_name)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        
        landmarks_list = []
        labels_list = []
        
        # Process each letter folder
        for letter in self.letters:
            letter_path = os.path.join(split_path, letter)
            
            if not os.path.exists(letter_path):
                print(f"⚠️  Warning: Letter directory not found: {letter}")
                continue
            
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(letter_path, ext)))
                image_files.extend(glob.glob(os.path.join(letter_path, ext.upper())))
            
            if len(image_files) == 0:
                print(f"⚠️  No images found for letter {letter}")
                continue
                
            print(f"   Processing {len(image_files)} images for letter {letter}...")
            
            # Extract landmarks from each image
            successful_extractions = 0
            for image_path in tqdm(image_files, desc=f"Letter {letter}", leave=False):
                landmarks = self.landmark_extractor.extract_from_image(image_path)
                
                if landmarks is not None:
                    landmarks_list.append(landmarks)
                    labels_list.append(self.letter_to_idx[letter])
                    successful_extractions += 1
            
            success_rate = (successful_extractions / len(image_files)) * 100
            print(f"   ✅ {successful_extractions}/{len(image_files)} ({success_rate:.1f}%) successful for {letter}")
        
        return np.array(landmarks_list), np.array(labels_list)
    
    def _save_processed_data(self, train_X, train_y, val_X, val_y):
        """Save processed landmarks to files"""
        # Save data arrays
        np.save(os.path.join(self.output_dir, "train_landmarks.npy"), train_X)
        np.save(os.path.join(self.output_dir, "train_labels.npy"), train_y)
        np.save(os.path.join(self.output_dir, "val_landmarks.npy"), val_X)
        np.save(os.path.join(self.output_dir, "val_labels.npy"), val_y)
        
        # Save metadata
        metadata = {
            "letters": self.letters,
            "letter_to_idx": self.letter_to_idx,
            "train_samples": len(train_X),
            "val_samples": len(val_X),
            "feature_dim": 63,
            "num_classes": 26
        }
        
        with open(os.path.join(self.output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Processed data saved to {self.output_dir}/")