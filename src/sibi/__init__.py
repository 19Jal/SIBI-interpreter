
"""SIBI Recognition System v2"""

__version__ = "2.0.0"

# Import main classes for easy access
from .models.mlp import SIBIBasicMLP
from .training.train import SIBITrainer
from .inference.recognizer import SIBIRealTimeRecognizer
from .dataprocessing.processor import SIBIDatasetProcessor
from .utils.landmarks import HandLandmarkExtractor

__all__ = [
    "SIBIBasicMLP",
    "SIBITrainer",
    "SIBIRealTimeRecognizer", 
    "SIBIDatasetProcessor",
    "HandLandmarkExtractor",
]