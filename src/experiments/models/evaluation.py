import torch
from torch.utils.data import DataLoader
from rich.progress import Progress

def iou(dataloader, model: torch.nn.Module, num_classes: int = 3):
    """
    Calculate the Intersection over Union (IoU) for a given dataset and model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing (image, label) pairs.
        model (torch.nn.Module): The model to evaluate.
        num_classes (int): Number of classes.

    Returns:
        dict: Per-class IoU and mean IoU in the format:
              {"per_class_iou": [...], "mIoU": ...}
    """
    with torch.no_grad():
        model.eval()
        iou_per_class = torch.zeros(num_classes, device=model.device)
        total_per_class = torch.zeros(num_classes, device=model.device)

        with Progress() as progress:
            task = progress.add_task("Computing IoU", total=len(dataloader))
            
            for images, labels in dataloader:  # labels are not one-hot encoded
                images = images.to(model.device)
                labels = labels.to(model.device)
                labels = labels[:, 0]  # Remove channel dimension
                outputs = model(images)  # Assume shape (B, C, H, W)
                predictions = outputs.argmax(dim=1)  # Convert logits to class indices

                for c in range(num_classes):
                    pred_c = (predictions == c)
                    label_c = (labels == c)

                    intersection = (pred_c & label_c).sum().float()
                    union = (pred_c | label_c).sum().float()

                    if union > 0:
                        iou_per_class[c] += intersection / union
                        total_per_class[c] += 1
                
                progress.update(task, advance=1)

        # Compute mean IoU while avoiding division by zero
        valid_classes = total_per_class > 0
        iou_per_class[valid_classes] /= total_per_class[valid_classes]

        miou = iou_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0

    return {"per_class_iou": iou_per_class.cpu().tolist(), "mIoU": miou}
