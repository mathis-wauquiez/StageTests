from abc import ABC, abstractmethod

from torch import nn, Tensor
import torch

from .models import ModelWrapper


class Scheduler(ABC, nn.Module):
    """
    Abstract class for a scheduler.
    Defines the properties and methods that all schedulers must implement.
    """

    @property
    @abstractmethod
    def alpha(self, t) -> Tensor:
        """
        Returns the alpha value for the current timestep.
        """
        pass

    @property
    @abstractmethod
    def alpha_dt(self, t) -> Tensor:
        """
        Returns the time derivative of alpha for the current timestep.
        """
        pass

    @property
    @abstractmethod
    def sigma(self, t) -> Tensor:
        """
        Returns the sigma value for the current timestep.
        """
        pass

    @property
    @abstractmethod
    def sigma_dt(self, t) -> Tensor:
        """
        Returns the time derivative of sigma for the current timestep.
        """
        pass


    def snr(self, t) -> Tensor:
        """
        Returns the signal-to-noise ratio for the current timestep.
        """
        return self.alpha(t) / self.sigma(t)
    
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """
        Sample from the scheduler.
        Args:
            x_0: Tensor (bs, ...), the initial state
            x_1: Tensor (bs, ...), the final state
            t: Scalar Tensor or (bs,) Tensor, the time step(s) to sample at.
        
        Returns:
            x_t: Tensor (bs, ...), the state at time t, equal to alpha(t) * x_1 + sigma(t) * x_0
        """
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
    