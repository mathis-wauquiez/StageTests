from src.flows.flow import Flow
import matplotlib.pyplot as plt
import numpy as np

import lpips
from pathlib import Path
from torch_ema import ExponentialMovingAverage
from pytorch_msssim import ssim as _ssim_fn   # pip install pytorch-msssim

_lpips_loss_fn = lpips.LPIPS(net='vgg')

def _get_psnr(x_pred, x_gt):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the predicted and ground truth images.
    """
    mse = ((x_pred - x_gt) ** 2).mean()
    psnr = 10 * np.log10(1 / mse)
    return psnr

def _get_lpips(x_pred, x_gt):
    """
    Calculate the Learned Perceptual Image Patch Similarity (LPIPS) between the predicted and ground truth images.
    """
    if len(x_pred.shape) == 3:
        x_pred = x_pred.unsqueeze(0)
        x_gt = x_gt.unsqueeze(0)

    lpips_score = _lpips_loss_fn(x_pred, x_gt)
    return lpips_score.item()


def _get_ssim(x_pred, x_gt, data_range=1.0):
    """
    Calculate the Structural Similarity Index (SSIM) between the predicted and
    ground-truth images.

    Args
    ----
    x_pred, x_gt : torch.Tensor
        Image tensors shaped  (B, C, H, W) **or** (C, H, W) in the range [0, data_range].
    data_range   : float
        Dynamic range of the images. 1.0 if tensors are already in [0,1],
        255 if they are uint-8 images converted to floats, etc.
    Returns
    -------
    float
        SSIM value averaged over the whole batch; 1 = perfect match.
    """
    # add batch dimension if the caller passed (C, H, W)
    if x_pred.ndim == 3:
        x_pred = x_pred.unsqueeze(0)
        x_gt   = x_gt.unsqueeze(0)

    # safety: clamp into the valid intensity range
    x_pred = x_pred.clamp(0, data_range)
    x_gt   = x_gt.clamp(0, data_range)

    score = _ssim_fn(
        x_pred,
        x_gt,
        data_range=data_range,
        size_average=True
    )

    return score.item()


class InpaintingFlow(Flow):
    """
    InpaintingFlow is a subclass of Flow that implements specific evaluation metrics for texture inpainting.
    """

    def __init__(self, *args, ema_decay=0.99, to_natural_fn=None, **kwargs):

        super().__init__(*args, **kwargs)

        self.to_natural = to_natural_fn 
        self.ema = ExponentialMovingAverage(self.parameters(), decay=ema_decay)
        self.ema.update(self.parameters())                       # Initialize EMA

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.ema.update(self.parameters())
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, *rest = batch
        y = rest[0] if rest else None

        # run the raw model
        x_pred = self(x0, y=y)
        # run the EMA model
        with self.ema.average_parameters():
            x_pred_ema = self(x0, y=y)

        # optional conversion back to natural image range
        if self.to_natural:
            x_pred    = self.to_natural(x_pred)
            x_pred_ema= self.to_natural(x_pred_ema)
            x1         = self.to_natural(x1)
            x0         = self.to_natural(x0)

        # compute the batch loss one more time for val_loss logging
        t, xt = self.path.sample(x0, x1)
        val_loss = self._get_loss(x0, x1, t, xt, y=y)

        return {
            "val_loss":       val_loss,
            "test_pred":      x_pred,
            "test_gt":        x1,
            "ema_test_pred":  x_pred_ema,
            "ema_test_gt":    x1,
            # we only need one sample for snapshots:
            "x0":             x0
        }

    # move the EMA to the correct device at the beginning of the training
    def on_fit_start(self):
        super().on_fit_start()
        if self.ema:
            self.ema.to(device=self.device)