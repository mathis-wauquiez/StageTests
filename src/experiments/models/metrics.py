import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from typing import Optional, Union, List

class PerClassAccuracy(Metric):
    """
    A metric that calculates accuracy for a specific class in a multiclass setting.
    
    Args:
        num_classes: Number of classes in the dataset
        class_id: The class ID for which to calculate accuracy (0-indexed)
        ignore_index: Optional class index to ignore
    """
    
    def __init__(
        self,
        num_classes: int,
        class_id: int,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        
        if class_id >= num_classes:
            raise ValueError(f"class_id ({class_id}) must be less than num_classes ({num_classes})")
        
        self.class_id = class_id
        
        # Use MulticlassAccuracy with average="none" internally
        self.accuracy_metric = MulticlassAccuracy(
            num_classes=num_classes,
            average="none",
            ignore_index=ignore_index
        )
        
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        self.accuracy_metric.update(preds, target)
        
    def compute(self) -> torch.Tensor:
        """Compute the accuracy for the specified class."""
        # Get per-class accuracies and extract the one we want
        all_class_accuracies = self.accuracy_metric.compute()
        return all_class_accuracies[self.class_id]
    
    def reset(self) -> None:
        """Reset metric states."""
        self.accuracy_metric.reset()