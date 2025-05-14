from abc import ABC, abstractmethod

from torch import nn, Tensor
import torch

from scipy.interpolate import CubicSpline
import numpy as np



class Scheduler(ABC, nn.Module):
    """
    Abstract class for a scheduler.
    Defines the properties and methods that all schedulers must implement.
    """

    @property
    @abstractmethod
    def alpha(self, t) -> Tensor:
        """
        Returns the alpha value for the timesteps.
        """
        pass

    @property
    @abstractmethod
    def alpha_dt(self, t) -> Tensor:
        """
        Returns the time derivative of alpha for the timesteps.
        """
        pass

    @property
    @abstractmethod
    def sigma(self, t) -> Tensor:
        """
        Returns the sigma value for the timesteps.
        """
        pass

    @property
    @abstractmethod
    def sigma_dt(self, t) -> Tensor:
        """
        Returns the time derivative of sigma for the timesteps.
        """
        pass


    def snr(self, t) -> Tensor:
        """
        Returns the signal-to-noise ratio for the timesteps.
        """
        sigma = self.sigma(t)

        snr = self.alpha(t) / sigma
        snr[sigma == 0] = float("inf")
        return snr
    
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Sample from the scheduler.
        Args:
            x_0: Tensor (bs, ...), the initial state
            x_1: Tensor (bs, ...), the final state
            t: Tensor (bs,), the time steps
        
        Returns:
            x_t: Tensor (bs, ...), the state at time t, equal to alpha(t) * x_1 + sigma(t) * x_0
        """
        t = t.to(x_0.device)
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)

        # Ensure proper shape for broadcasting
        while alpha_t.ndim < x_1.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
        while sigma_t.ndim < x_0.ndim:
            sigma_t = sigma_t.unsqueeze(-1)

        x_t = alpha_t * x_1 + sigma_t * x_0
        return x_t

class OTScheduler(Scheduler):
    """
    Linear scheduler, X_t = t * X_1 + (1-t) * X_0
    """

    def alpha(self, t) -> Tensor:
        return t
    
    def alpha_dt(self, t) -> Tensor:
        return torch.ones_like(t)
    
    def sigma(self, t) -> Tensor:
        return 1-t
    
    def sigma_dt(self, t) -> Tensor:
        return -torch.ones_like(t)
    

LinearScheduler = OTScheduler

class CosineScheduler(Scheduler):
    """
    Cosine scheduler, X_t = sin(t * pi / 2) * X_1 + cos(t * pi / 2) * X_0
    """

    def alpha(self, t) -> Tensor:
        return torch.sin(t * torch.pi / 2)
    
    def alpha_dt(self, t) -> Tensor:
        return torch.cos(t * torch.pi / 2) * (torch.pi / 2)
    
    def sigma(self, t) -> Tensor:
        return torch.cos(t * torch.pi / 2)
    
    def sigma_dt(self, t) -> Tensor:
        return -torch.sin(t * torch.pi / 2) * (torch.pi / 2)
    
    def snr(self, t) -> Tensor:
        """
        Returns the signal-to-noise ratio for the current timestep.
        """
        return torch.tan(t * torch.pi / 2)
    

class InterpolatedScheduler(Scheduler):
    """
    This scheduler uses a cubic spline to interpolate a DDPM VP scheduler.
    """
    def __init__(self, alphas_cumprod: Tensor):
        """
        alphas_cumprod: 1D Tensor of length T, values in (0,1].
        """
        super().__init__()
        # keep original tensor (for device/dtype) and build numpy spline
        self.alphas_cumprod = alphas_cumprod
        acp_np = alphas_cumprod.cpu().numpy()
        grid = np.linspace(0.0, 1.0, len(acp_np))
        self._cs = CubicSpline(grid, acp_np)
        self._cs_der = self._cs.derivative()

    def alpha(self, t: Tensor) -> Tensor:
        # α(t) = sqrt( ᾱ(1−t) )
        t_np = t.cpu().numpy()
        s = 1.0 - t_np
        abar = self._cs(s)
        out = np.sqrt(abar)
        return torch.tensor(out, device=t.device, dtype=t.dtype)

    def sigma(self, t: Tensor) -> Tensor:
        # σ(t) = sqrt(1 − ᾱ(1−t))
        t_np = t.cpu().numpy()
        s = 1.0 - t_np
        abar = self._cs(s)
        out = np.sqrt(1.0 - abar)
        return torch.tensor(out, device=t.device, dtype=t.dtype)

    def alpha_dt(self, t: Tensor) -> Tensor:
        """
        dα/d(1−t) = d/ds [ sqrt( ᾱ(s) ) ]
                  = (1/(2√ᾱ(s))) * ᾱ′(s)
        """
        t_np = t.cpu().numpy()
        s = 1.0 - t_np
        abar = self._cs(s)
        d_abar = self._cs_der(s)
        out = d_abar / (2.0 * np.sqrt(abar))
        return torch.tensor(out, device=t.device, dtype=t.dtype)

    def sigma_dt(self, t: Tensor) -> Tensor:
        """
        dσ/d(1−t) = d/ds [ sqrt(1−ᾱ(s)) ]
                  = −ᾱ′(s) / (2√(1−ᾱ(s)))
        """
        t_np = t.cpu().numpy()
        s = 1.0 - t_np
        abar = self._cs(s)
        d_abar = self._cs_der(s)
        out = -d_abar / (2.0 * np.sqrt(1.0 - abar))
        return torch.tensor(out, device=t.device, dtype=t.dtype)