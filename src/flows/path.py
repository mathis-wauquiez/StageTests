from abc import ABC, abstractmethod
from typing import Literal, Optional

from torch import nn, Tensor
import torch

from .schedulers import Scheduler
from .types import Predicts

from .models import ModelWrapper
from torch.distributions import Distribution
from torch.distributions.uniform import Uniform

import numpy as np


# -----------------------------------------------------------------------------
# Affine Paths Class
# -----------------------------------------------------------------------------


class AffinePath(nn.Module):
    """
    Represents an affine interpolation path between two endpoints in a flow matching framework.

    This module allows sampling intermediate states between two fixed endpoints `x_0` and `x_1`
    using a scheduler, and supports converting model predictions between different parameterizations
    (e.g., `x_0`, `x_1`, `velocity`, `score`), as defined in the Meta PFGM++ framework.

    Parameters
    ----------
    scheduler : Scheduler
        A scheduler object that defines the interpolation rules (e.g., alpha, sigma).
    t_law : Optional[Distribution]
        A PyTorch distribution to sample random time values from. Defaults to Uniform(0, 1).
        
    Methods
    -------
    target_velocity(t, x_0, x_1):
        Computes the target velocity at time t using the scheduler's alpha and sigma functions.
    
    sample(x_0, x_1, t=None):
        Samples an intermediate point x_t between x_0 and x_1 at time t.
        
    convert_parameterization(x_t, t, f_A, source_parameterization, target_parameterization):
        Converts a model prediction from one parameterization to another.
        The parameterizations can be `x_0`, `x_1`, `velocity`, or `score`.

    _get_parameterization_conversion_coefficients(origin, target, t):
        Computes the coefficients for converting between different parameterizations.
        This is based on the equations provided in the Meta paper, Table 1.

    Example usage:
        >>> scheduler = OTScheduler()
        >>> path = AffinePath(scheduler)
        >>> t, x_t = path.sample(x_0, x_1) # samples t~p_t(t) and x_t~p_t(x_t|x_0,x_1)
        >>> v_t = path.target_velocity(t, x_0, x_1)
        >>> f_A = model.predict(t, x_t) # model prediction at time t. can be x_0, x_1, score or velocity
        >>> x_0_estimated = path.convert_parameterization(t, x_t, f_A, source_parameterisation, "x_1")
        
    The convention is to always use the order (t, x_t) when calling methods.
    """
            
    def __init__(
            self,
            scheduler: Scheduler,
            t_law: Optional[Distribution] = Uniform(low=0.0, high=1.0),
            tol: float = 1e-3,
            e_x: Optional[Tensor] = None # E[X_1]
    ):
        super().__init__()
        if not isinstance(scheduler, Scheduler):
            raise ValueError("Scheduler must be an instance of the Scheduler class.")
        if not hasattr(t_law, "sample"):
            raise ValueError("t_law must be a torch Distribution with a sample(batch_size) method.")

        self.scheduler = scheduler
        self.t_law = t_law
        self.tol = tol
        self.e_x = e_x

    def sample(self, x_0: Tensor, x_1: Tensor, t: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        
        if t is None:
            t = self.t_law.sample((x_0.shape[0],)).to(x_0.device)

        if t.dim() != 1:
            raise ValueError("t must be a 1D tensor.")

        x_t = self.scheduler.sample(x_0, x_1, t)
        return t, x_t
    

    def target_velocity(self, t: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
        if t.dim() != 1:
            raise ValueError("t must be 1-D (batch dimension only).")
        if x_0.shape != x_1.shape:
            raise ValueError("x_0 and x_1 must be the same shape.")

        alpha_dt = self._broadcast(self.scheduler.alpha_dt(t), x_1.ndim)
        sigma_dt = self._broadcast(self.scheduler.sigma_dt(t), x_0.ndim)
        return alpha_dt * x_1 + sigma_dt * x_0

    # ------------------------------------------------------------------
    #  Convert between parameterizations
    # ------------------------------------------------------------------


    def convert_parameterization(self,
                                 t: Tensor,
                                 x_t: Tensor,
                                 f_A: Tensor,
                                 source_parameterization: Predicts,
                                 target_parameterization: Predicts) -> Tensor:
        """
        Convert one type of parameterization to another.
        Can be used to convert between x_0, x_1, score and velocity, according to the table 1 in the Meta paper.

        Example usage:
            >>> x_0 = torch.randn(10, 3)
            >>> x_1 = torch.randn(10, 3)
            >>> t, x_t = path.sample(x_0, x_1)
            >>> f_A = model.predict(t, x_t)
            >>> converted_value = path.convert_parameterization(t, x_t, f_A, "x_0", "score")
        """

        # ------------------------------------------------------ sanity checks
        assert x_t.shape == f_A.shape, f"x_t and f_A must be the same shape, got {x_t.shape} and {f_A.shape}"
        assert t.dim() == 1, f"t must be a 1-D tensor, got {t.dim()} dimensions"
        
        source = Predicts.from_any(source_parameterization)
        target = Predicts.from_any(target_parameterization)

        
        self._check_valid_conversion(t, source_parameterization, target_parameterization)
                
        t_0_mask = torch.isclose(t, torch.zeros_like(t), atol=self.tol)
        t_1_mask = torch.isclose(t, torch.ones_like(t), atol=self.tol)

        output = torch.zeros_like(x_t)

        # ------------------------------------------------------ t_0 and t_1 cases
        if t_0_mask.any():
            output[t_0_mask] = self._convert_parameterization_t0(t[t_0_mask], x_t[t_0_mask], f_A[t_0_mask], source, target)
        if t_1_mask.any():
            output[t_1_mask] = self._convert_parameterization_t1(t[t_1_mask], x_t[t_1_mask], f_A[t_1_mask], source, target)

        # ------------------------------------------------------ General case
        mask_t = torch.logical_not(t_0_mask) & torch.logical_not(t_1_mask)
        if mask_t.any():
            # Source to velocity
            a, b = self._to_velocity_coeffs(t[mask_t], source)
            a = self._broadcast(a, x_t.ndim)
            b = self._broadcast(b, x_t.ndim)
            velocity = a * x_t[mask_t] + b * f_A[mask_t]

            # Velocity to target
            a, b = self._from_velocity_coeffs(t[mask_t], target)
            a = self._broadcast(a, x_t.ndim)
            b = self._broadcast(b, x_t.ndim)
            output[mask_t] = a * x_t[mask_t] + b * velocity
        
        return output



    def _check_valid_conversion(self, t: Tensor, source: Predicts, target: Predicts) -> bool:
        t_0_mask = torch.isclose(t, torch.zeros_like(t), atol=self.tol)
        t_1_mask = torch.isclose(t, torch.ones_like(t), atol=self.tol)

        # ------------------------------------------------------ t_1 check
        if t_1_mask.any():
            if target == Predicts.SCORE and source != Predicts.SCORE:
                raise ValueError(f"Cannot convert from {source} to {target} when t=1")

        # when t != 0 and t != 1, we can convert from any parameterization to any other


    def _convert_parameterization_t0(self,
                                 t: Tensor,
                                 x_t: Tensor,
                                 f_A: Tensor,
                                 source: Predicts,
                                 target: Predicts) -> Tensor:
        """
        Convert from one parameterization to another when t=0.
        """
        assert torch.isclose(t, torch.zeros_like(t), atol=self.tol).all(), f"t must be 0, got {t}"

        # Get the alpha and sigma values at time t
        alpha_dt = self.scheduler.alpha_dt(t)
        sigma = self.scheduler.sigma(t)
        sigma_dt = self.scheduler.sigma_dt(t)

        # Ensure proper shape for broadcasting
        alpha_dt = self._broadcast(alpha_dt, x_t.ndim)
        sigma = self._broadcast(sigma, x_t.ndim)
        sigma_dt = self._broadcast(sigma_dt, x_t.ndim)

        if source == target:
            return f_A
        
        if target == Predicts.X1:
            return self._get_e_x(x_t)
        
        if target == Predicts.VELOCITY:
            return sigma_dt * x_t + alpha_dt * self._get_e_x(x_t)
        
        if target == Predicts.X0:
            return x_t

        if target == Predicts.SCORE:
            return - sigma**2 * x_t

    def _convert_parameterization_t1(self,
                                 t: Tensor,
                                 x_t: Tensor,
                                 f_A: Tensor,
                                 source: Predicts,
                                 target: Predicts) -> Tensor:
        """
        Convert from one parameterization to another when t=1.
        """
        assert torch.isclose(t, torch.ones_like(t), atol=self.tol).all(), f"t must be 1, got {t}"

        alpha_dt = self.scheduler.alpha_dt(t)
        alpha_dt = self._broadcast(alpha_dt, x_t.ndim)

        if source == target:
            return x_t
        
        if target == Predicts.X0:
            return torch.zeros_like(x_t)
        if target == Predicts.X1:
            return x_t
        if target == Predicts.VELOCITY:
            return alpha_dt * x_t

    def _to_velocity_coeffs(self,
                    t: Tensor,
                    source: Predicts) -> Tensor:
        """
        Convert from one parameterization to velocity when t!=0 and t!=1.
        """
        assert t.dim() == 1, f"t must be a 1-D tensor, got {t.dim()} dimensions"
        t_0_mask = torch.isclose(t, torch.zeros_like(t), atol=self.tol)
        t_1_mask = torch.isclose(t, torch.ones_like(t), atol=self.tol)
        assert not (t_0_mask | t_1_mask).any(), "this function does not support t=0 or t=1. please use convert_parameterization() directly"

        sigma = self.scheduler.sigma(t)
        sigma_dt = self.scheduler.sigma_dt(t)
        alpha_dt = self.scheduler.alpha_dt(t)
        alpha = self.scheduler.alpha(t)

        if source == Predicts.VELOCITY:
            return (torch.zeros_like(t), torch.ones_like(t))
        if source == Predicts.X1:
            return sigma_dt/sigma, \
                    (alpha_dt*sigma-sigma_dt*alpha)/sigma
        if source == Predicts.X0:
            return alpha_dt/alpha,\
                    (sigma_dt*alpha-alpha_dt*sigma)/alpha
        if source == Predicts.SCORE:
            return alpha_dt/alpha,\
                    -sigma * (sigma_dt*alpha-alpha_dt*sigma)/alpha

        raise ValueError(f"Cannot convert from {source} to velocity when t!=0 and t!=1")
    

    def _from_velocity_coeffs(self,
                    t: Tensor,
                    target: Predicts) -> tuple[Tensor, Tensor]:
        """
        Convert from velocity to one parameterization when t!=0 and t!=1.
        """
        if target == Predicts.VELOCITY:
            return torch.zeros_like(t), torch.ones_like(t)
        
        a, b = self._to_velocity_coeffs(t, target)

        # v = a * x_t + b * f_A
        # => f_A = (v - a * x_t) / b
        # <=> f_A = -a/b * x_t + 1/b * v

        return -a/b, 1/b

            
    @staticmethod
    def _broadcast(coeff: Tensor, target_rank: int) -> Tensor:
        """Right‑broadcast *coeff* to match *target_rank*."""
        return coeff.view(-1, *[1] * (target_rank - coeff.ndim))


    def _get_e_x(self, x_t: Tensor) -> Tensor:
        if self.e_x is None:
            return torch.zeros_like(x_t)
        
        x_shape = x_t.shape[1:]
        e_x = self.e_x.view(1, *x_shape)
        e_x = e_x.expand(x_t.shape[0], *x_shape)
        return e_x