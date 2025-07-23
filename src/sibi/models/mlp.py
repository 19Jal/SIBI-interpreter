""" SIBI Basic MLP Model
This module defines a basic Multi-Layer Perceptron (MLP) model for recognizing SIBI Letters"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

class SIBIBasicMLP(nn.Module):
    """Basic MLP for SIBI letter recognition"""
    
    def __init__(
        self, 
        input_size: int = 63, 
        num_classes: int = 26, 
        hidden_sizes: List[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
            
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'hidden_sizes': self.hidden_sizes,
            'dropout': self.dropout
        }