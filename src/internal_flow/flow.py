from src.flows.flow import Flow

from typing import Optional
from torch_ema import ExponentialMovingAverage
import torch
from torch import nn
from torch import Tensor

from torchdiffeq import odeint

from typing import Any, Dict, Callable


class InpaintingFlow(Flow):
    """
    InpaintingFlow is a subclass of Flow that implements specific evaluation metrics for texture inpainting.
    """

    def __init__(self, *args, ema_decay=0.99, to_natural_fn=None, **kwargs):

        super().__init__(*args, **kwargs)

        self.to_natural = to_natural_fn 
        self.ema = ExponentialMovingAverage(self.parameters(), decay=ema_decay)
        self.ema.update(self.parameters())                       # Initialize EMA

    def tweaked_velocity(self, x_1, t, x_t, y):
        estimated_velocity = super().estimated_velocity(t, x_t, y=y)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x_t.shape[0])
        true_velocity = self.path.convert_parameterization(t, x_t, x_1, source_parameterization="x1", target_parameterization="velocity")
        
        M = y # Inpainting Mask

        estimated_velocity = estimated_velocity * M + (1 - M) * true_velocity
        return estimated_velocity

    def sample_cheat(self, x_0: Tensor, x_1, y: Optional[Tensor] = None, **solver_cfg: Any) -> Tensor:
        
        # Override of the sample method, to use the exact solution for the exterior of the inpainting mask.
        

        t_span = torch.tensor([0., 1.], device=x_0.device)
        solver_cfg = {'method':'dopri5'} | self._merge_config(solver_cfg)

        # Pass x_1
        velocity_field = lambda t, x: self.tweaked_velocity(x_1, t, x, y=y)

        # Solve the ODE
        with torch.no_grad():
            trajectory = odeint(
                velocity_field,
                x_0,
                t_span,
                **solver_cfg
            )
        
        return trajectory[-1, ...] # Trajectory is of shape (2, BS, ...)


    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.ema.update(self.parameters())
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, *rest = batch
        y = rest[0] if rest else None

        # run the raw model
        x_pred = self.sample_cheat(x0, x1, y=y)
        # run the EMA model
        with self.ema.average_parameters():
            x_pred_ema = self.sample_cheat(x0, x1, y=y)

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