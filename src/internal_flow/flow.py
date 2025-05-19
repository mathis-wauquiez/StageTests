from src.flows.flow import Flow
import matplotlib.pyplot as plt
import numpy as np

import lpips

from .visualization import visualize

from torch_ema import ExponentialMovingAverage

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


class InpaintingFlow(Flow):
    """
    InpaintingFlow is a subclass of Flow that implements specific evaluation metrics for texture inpainting.
    """

    def __init__(self, *args, **kwargs):

        self.viz = kwargs.pop("viz", None)                     # Wether to visualize the results or not
        self.to_natural_fn = kwargs.pop("to_natural_fn", None) # Function to convert to natural image
        decay = kwargs.pop("ema_decay", 0.99)                     # EMA decay rate

        super().__init__(*args, **kwargs)
        
        self.ema = ExponentialMovingAverage(self.parameters(), decay=decay)
        self.ema.update(self.parameters())                       # Initialize EMA

    def step(self, batch, batch_idx, ema=False):

        template = "test" if not ema else "ema_test"

        if len(batch) == 3:
            x_0, x_1, y = batch
        else:
            x_0, x_1 = batch
            y = None

        t, x_t = self.path.sample(x_0, x_1)
        loss = self._get_loss(x_0, x_1, t, x_t, y=y) if y is not None else self._get_loss(
            x_0, x_1, t, x_t
        )

        self.log_dict({f"{template}_loss": loss}, prog_bar=True, on_epoch=True)

        x_pred = self.sample(x_0, y=y).cpu()
        x_0 = x_0.cpu()
        x_1 = x_1.cpu()

        x_pred = self.to_natural_fn(x_pred) if self.to_natural_fn else x_pred
        x_0 = self.to_natural_fn(x_0) if self.to_natural_fn else x_0
        x_1 = self.to_natural_fn(x_1) if self.to_natural_fn else x_1

        if self.viz:
            visualize(x_0, x_1, x_pred, ema=ema)
        
        # Log the PSNR
        psnr = _get_psnr(x_pred, x_1)
        self.log_dict({f"{template}_psnr": psnr}, prog_bar=True, on_epoch=True)

        # Log the LPIPS
        lpips_loss = _get_lpips(x_pred, x_1)
        self.log_dict({f"{template}_lpips": lpips_loss}, prog_bar=True, on_epoch=True)

        return loss


    def test_step(self, batch, batch_idx):
        """
        Test step for the inpainting flow model.
        """
        self.eval()
        self.step(batch, batch_idx, ema=False)

        with self.ema.average_parameters():
            self.ema.to(device=self.device)
            self.step(batch, batch_idx, ema=True)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx) # bad practice

    # move the EMA to the correct device at the beginning of the training
    def on_fit_start(self):
        if self.ema:
            self.ema.to(device=self.device)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        if self.ema:
            self.ema.update(self.parameters())