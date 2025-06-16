from src.flows.flow import Flow
from src.flows.types import Predicts, Guidance

from torch_ema import ExponentialMovingAverage
import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchmetrics
import torchvision as tv

class InpaintingFlow(Flow):
    """
    InpaintingFlow is a subclass of Flow that implements specific evaluation metrics for texture inpainting.

    In this class, we model the generative task as generating the inside of the inpainting mask, given the outside and the mask itself.
    This differs from the other attempt, located in `src/internal_flow/flow.py`, which generates the entire image, given the outside of the mask and the mask itself.

    Here, we adapt our loss: we compute the loss on the masked region only, as we do no longer predict the entire image, but only the inside of the mask.
    Moreover, we adapt the velocity field such that it is null on the outside of the mask, and we cheat the sampling process to only sample the inside of 
    """

    def __init__(self, *args, val_samples_per_batch=16, ema_decay=0.99, to_natural_fn=None, **kwargs):
        # default init + EMA weights

        super().__init__(*args, **kwargs)
        self.val_samples_per_batch = val_samples_per_batch
        self.to_natural = to_natural_fn 


        model_params = [p for name, p in self.named_parameters() if not name.startswith('val_metrics')]
        self.ema = ExponentialMovingAverage(model_params, decay=ema_decay)
        self.ema.update(model_params)

        self.val_metrics = torchmetrics.MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(data_range=1.0),
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True, reduction="mean"), # normalize=True -> [0, 1] range
            }
        )
        self.val_metrics.to(device=self.device)
        self.val_metrics.eval()
        self.path.eval()


    # == Loss function ================================================================================

    def _get_loss(self, x_0: Tensor, x_1: Tensor, t: Tensor, x_t: Tensor,  **kwargs) -> Tensor:
        """ Compute the loss for the given inputs. """

        if self.cfg.guidance == Guidance.CFG:
            random_mask = torch.rand(x_0.shape[0], device=x_0.device) < self.cfg.guided_prob
            kwargs.update({'cond_mask': random_mask})

        outputs = self.model(t, x_t, **kwargs)
        v_theta = self.path.convert_parameterization(t, x_t, outputs, self.cfg.predicts, Predicts.VELOCITY)
        v_target = self.path.target_velocity(t, x_0, x_1)
        
        M = kwargs.get('y', None)  # Inpainting Mask
        if M is not None:
            # If an inpainting mask is provided, we only compute the loss on the masked region
            loss = ((v_theta - v_target) * M).pow(2).mean()
        else:
            raise ValueError("Inpainting mask 'y' must be provided for inpainting flows.")
        
        return loss
    
    def _log_metrics(self, tag: str, loss, batch, on_step: bool = True, on_epoch: bool = True, prog_bar: bool = True, logger: bool = True, log_images: bool = True, log_metrics: bool = True, include_lpips: bool = False):
        x0, x1, *rest = batch
        y = rest[0] if rest else None
        x1_ = self.sample(x0, y=y)  # Sample a predicted image from the model

        if self.to_natural:
            x0 = self.to_natural(x0)
            x1 = self.to_natural(x1)
            x1_ = self.to_natural(x1_)

        # Similarity metrics
        if log_metrics:
            self.log(f"{tag}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=logger)
            self.log(f"{tag}_psnr", self.val_metrics["psnr"](x1, x1_), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)
            self.log(f"{tag}_ssim", self.val_metrics["ssim"](x1, x1_), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)
            if include_lpips:
                # Log LPIPS only if requested
                self.log(f"{tag}_lpips", self.val_metrics["lpips"](x1, x1_), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)

        if log_images:
            x_shape = x0.shape
            update_fn = lambda x: x[:9].detach().cpu().view(9, *x_shape[1:])
            x0_grid = tv.utils.make_grid(update_fn(x0[:9]), nrow=3, normalize=True, scale_each=True)
            x1_grid = tv.utils.make_grid(update_fn(x1[:9]), nrow=3, normalize=True, scale_each=True)
            x1_pred_grid = tv.utils.make_grid(update_fn(x1_[:9]), nrow=3, normalize=True, scale_each=True)
            
            # Log the images
            for name, grid in zip(
                [f"{tag}_x0", f"{tag}_x1", f"{tag}_x1_pred"],
                [x0_grid, x1_grid, x1_pred_grid]
            ):
                self.logger.log_image(
                    key=name,          # name of the panel in W&B
                    images=[grid],     # list of one image
                    caption=[f"epoch {self.current_epoch}"]
                )


    def _apply_corruption(self, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sigma = 0.0
        # Bruit uniquement dans la zone masquée
        noise = torch.randn_like(clean) * mask
        corrupt = clean * (1.0 - mask) + noise
        if sigma is not None:
            eps = torch.randn_like(clean) * mask
            corrupt = corrupt * (1 - sigma) ** 0.5 + eps * sigma ** 0.5
        return corrupt





    # == Vector field and sampling methods =========================================================

    def estimated_velocity(self, t, x_t, y):
        """
        On the outside of the inpainting mask, we want the velocity to be zero, no matter what the model predicts.
        """
        estimated_velocity = super().estimated_velocity(t, x_t, y=y)
        M = y # Inpainting Mask
        return estimated_velocity * M
    
    # actually, we don't need to override the sample method, as the estimated_velocity method does not longer need x_1


    # == Lightning hooks ============================================================================

    def training_step(self, batch, batch_idx):
        # Update the EMA weights after each training step
        loss = super().training_step(batch, batch_idx)
        # model_params = [p for name, p in self.named_parameters() if not name.startswith('val_metrics')]
        # self.ema.update(model_params)

        # # Log the metrics on the training step
        # self._log_metrics(
        #     tag="train",
        #     loss=loss,
        #     batch=batch,
        #     log_images=False
        # )

        return loss        

    def validation_step(self, batch, batch_idx):
        """
        Evaluate the model with/without EMA weights on a batch of data.
        """
        x0, x1, *rest = batch
        y = rest[0] if rest else None

        # We repeat the tensors to sample multiple times
        n_repeats = self.val_samples_per_batch
        x1 = x1.repeat(n_repeats, 1, 1, 1)
        if y is not None:
            y = y.repeat(n_repeats, 1, 1, 1)

        x0 = self._apply_corruption(x1, y) if y is not None else torch.randn_like(x1)
        

        # run the raw model
        x_pred = self.sample(x0, y=y)
        # run the EMA model
        with self.ema.average_parameters():
            x_pred_ema = self.sample(x0, y=y)

        # compute the batch loss one more time for val_loss logging
        t, xt = self.path.sample(x0, x1)
        val_loss = self._get_loss(x0, x1, t, xt, y=y)

        # log the metrics
        self._log_metrics(
            tag="val",
            loss=val_loss,
            batch=(x0, x1, y) if y is not None else (x0, x1),
            on_step=False,  # No need to log on step for validation
            prog_bar=False,  # Show in progress bar
            log_images=True,  # Log images
            log_metrics=True,  # Log metrics
            include_lpips= True
        )


        # conversion back to natural image range
        if self.to_natural:
            x_pred    = self.to_natural(x_pred)
            x_pred_ema= self.to_natural(x_pred_ema)
            x1         = self.to_natural(x1)
            x0         = self.to_natural(x0)


        return {
            "test_pred":      x_pred,
            "test_gt":        x1,
            "ema_test_pred":  x_pred_ema,
            "x0":             x0
        }

    # move the EMA to the correct device at the beginning of the training
    def on_fit_start(self):
        super().on_fit_start()
        if self.ema:
            self.ema.to(device=self.device)
        self.val_metrics.to(device=self.device)