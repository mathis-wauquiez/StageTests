from abc import ABC, abstractmethod
import torch.nn as nn

class ModelWrapper(ABC, nn.Module):
    
    @abstractmethod
    def forward(t, x, **kwargs):
        pass