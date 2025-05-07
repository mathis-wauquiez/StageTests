import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, losses, lambdas):
        """
        CombinedLoss constructor.

        Args:
            losses (list of nn.Module): List of loss functions to combine.
            lambdas (list of float): List of weights for each loss function.
        """
        super(CombinedLoss, self).__init__()
        assert len(losses) == len(lambdas), "Length of losses and lambdas must be the same."
        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas

    def forward(self, logits, labels):
        """
        Forward pass for the combined loss.

        Args:
            logits (torch.Tensor): Model predictions (logits).
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Combined loss value.
        """
        total_loss = 0.0
        for loss_fn, lambda_ in zip(self.losses, self.lambdas):
            loss = loss_fn(logits, labels)
            total_loss += lambda_ * loss
        return total_loss

# Example loss functions (MultiClassDiceLoss and LovaszSoftmax)
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        MultiClassDiceLoss constructor.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        Forward pass for the MultiClassDiceLoss.

        Args:
            logits (torch.Tensor): Model predictions (logits).
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Dice loss value.
        """
        probs = torch.softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).permute(0, 3, 1, 2).float()
        intersection = (probs * labels_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        """
        LovaszSoftmax constructor.

        Args:
            reduction (str): Reduction method ('mean' or 'sum').
        """
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Forward pass for the LovaszSoftmax loss.

        Args:
            logits (torch.Tensor): Model predictions (logits).
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Lovasz-Softmax loss value.
        """
        probs = torch.softmax(logits, dim=1)
        probs_flat = probs.view(-1, probs.size(1))
        labels_flat = labels.view(-1)
        loss = lovasz_softmax_flat(probs_flat, labels_flat, self.reduction)
        return loss

def lovasz_softmax_flat(probs, labels, reduction='mean'):
    """
    Multi-class Lovász-Softmax loss for flattened inputs.
    """
    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    
    if reduction == 'mean':
        return torch.mean(torch.stack(losses))
    elif reduction == 'sum':
        return torch.sum(torch.stack(losses))
    else:
        raise ValueError("Reduction method not supported. Use 'mean' or 'sum'.")

def lovasz_grad(fg):
    """
    Compute the gradient of the Lovász extension.
    """
    gts = fg.sum()
    intersection = gts - fg.cumsum(0)
    union = gts + (1 - fg).cumsum(0)
    jaccard = 1. - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


# Example usage
if __name__ == "__main__":
    # Create dummy predictions and targets
    logits = torch.randn(4, 3, 256, 256)  # Batch of 4, 3 classes, 256x256 images
    targets = torch.randint(0, 3, (4, 256, 256))  # Ground truth masks with class indices

    # Instantiate the CombinedLoss
    loss_fn = CombinedLoss(
        losses=[MultiClassDiceLoss(smooth=1.0), LovaszSoftmax(reduction='mean')],
        lambdas=[1.0, 1.0]
    )

    loss = loss_fn(logits, targets)
    print(loss.item())