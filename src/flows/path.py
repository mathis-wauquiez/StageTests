from abc import ABC, abstractmethod
from typing import Literal, Optional

from torch import nn, Tensor
import torch

from .schedulers import Scheduler
from .models import ModelWrapper
from torch.distributions import Distribution
from torch.distributions.uniform import Uniform

import numpy as np



class AffinePath(nn.Module):
    """
    Represents an affine interpolation path between two endpoints in a flow matching framework.

    This module allows sampling intermediate states between two fixed endpoints `x_0` and `x_1`
    using a scheduler, and supports converting model predictions between different parameterizations
    (e.g., `x_0`, `x_1`, `velocity`, `score`), as defined in the Meta PFGM++ framework.

    Args:
        scheduler (Scheduler): A scheduler object that defines the interpolation rules (e.g., alpha, sigma).
        model (ModelWrapper): A wrapper around the model used to predict intermediate quantities (e.g., score, x0).
        predicts (Literal["x_0", "x_1", "score"]): Type of prediction made by the model.
        guidance (Optional[Literal["CFG", "classifier"]], optional): Type of conditional guidance to apply.
        guidance_scale (float, optional): Scale for the guidance. Defaults to 1.0.
        t_law (Optional[Distribution], optional): A PyTorch distribution to sample random time values from. Defaults to Uniform(0, 1).
        classifier (Optional[nn.Module], optional): A classifier network used if `guidance="classifier"` is specified.

    Raises:
        ValueError: If classifier guidance is selected without providing a classifier.

    Arguments in the methods:
        x_0: Tensor (bs, ...), the initial state
        x_1: Tensor (bs, ...), the final state
        t: Tensor (bs, ...), the time step.
        f_A: The value to convert (score, x_0, x_1, or velocity)
        source_parameterization: The parameterization of the value to convert. Either x_0, x_1, score or velocity
        target_parameterization: The parameterization to convert to. Either x_0, x_1, score or velocity

    Methods:
        sample(x_0, x_1, t=None):
            Samples an intermediate point x_t between x_0 and x_1 at time t.
        
        estimated_velocity(x, t, **kwargs):
            Estimates the velocity at time t using the model prediction.

        convert_parameterization(x_t, t, f_A, source_parameterization, target_parameterization):
            Converts a model prediction from one parameterization to another.

    Example usage:
        >>> model = ...  # Your model here. Should take (t, x) as input and output a tensor representing either x_0, x_1, or score.
        >>> scheduler = OTScheduler()
        >>> path = AffinePath(scheduler, model, predicts="score")
        >>> t, x_t = path.sample(x_0, x_1) # samples t~p_t(t) and x_t~p_t(x_t|x_0,x_1)
        >>> v_t = path.estimated_velocity(x_t, t=t) # u^\theta(t, x_t)
        >>> x_0_estimated = path.convert_parameterization(t, x_t, v_t, "v", "x_0")
        
    The convention is to always use the order (t, x_t) when calling methods.
    """
            
    def __init__(
            self,
            scheduler: Scheduler,
            t_law: Optional[Distribution] = Uniform(low=0.0, high=1.0),
    ):
        super().__init__()

        self.scheduler = scheduler
        self.t_law = t_law

        # Ensure the t_law is a valid distribution
        if not hasattr(t_law, "sample"):
            raise ValueError("t_law must be a valid distribution with a sample method")
        
    def sample(self, x_0, x_1, t=None):
        """
        Sample from the conditional path.

        Args:
            x_0: Tensor (bs, ...), the initial state
            x_1: Tensor (bs, ...), the final state
            t: Optional[Tensor], the time steps to sample at. If None, sample from the t_law.
        """
        if t is None:
            t = self.t_law.sample((x_0.shape[0],)).to(x_0.device)


        if t.dim() != 1:
            raise ValueError("t must be a 1D tensor")
        
        if x_0.shape != x_1.shape:
            raise ValueError("x_0 and x_1 must be the same shape")
    
        x_t = self.scheduler.sample(x_0, x_1, t)

        return t, x_t
    
    def target_velocity(self, t: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
        """Closed‑form ground truth for ``v(t, x_t)`` (eq. 4.38)."""
        if x_0.shape != x_1.shape:
            raise ValueError("x_0 and x_1 must have identical shape.")
        if t.dim() != 1:
            raise ValueError("t must be 1‑D (batch dimension only).")

        alpha_dt = self._broadcast_coeff(self.scheduler.alpha_dt(t), x_1.ndim)
        sigma_dt = self._broadcast_coeff(self.scheduler.sigma_dt(t), x_0.ndim)
        return alpha_dt * x_1 + sigma_dt * x_0


    def convert_parameterization(self, t, x_t, f_A, source_parameterization, target_parameterization):
        """
        Convert one type of parameterization to another.
        Can be used to convert between x_0, x_1, score and velocity, according to the table 1 in the Meta paper.

        Args:
            t: The time step.
            x_t: The value of the path at time t
            f_A: The value to convert
            source_parameterization: The parameterization of the value to convert. Either x_0, x_1, score or velocity
            target_parameterization: The parameterization to convert to. Either x_0, x_1, score or velocity
        """

        assert source_parameterization in ["v", "x_0", "x_1", "score", "velocity"], f"{source_parameterization} is not a valid parameterization"
        assert target_parameterization in ["v", "x_0", "x_1", "score", "velocity"], f"{target_parameterization} is not a valid parameterization"
        assert x_t.shape == f_A.shape, f"x_t and f_A must be the same shape, got {x_t.shape} and {f_A.shape}"

        # Coefficients are of shape (bs, )
        a, b = self._get_parameterization_conversion_coefficients(source_parameterization, target_parameterization, t)

        # Ensure proper shape for broadcasting
        while a.ndim < x_t.ndim:
            a = a.unsqueeze(-1)
        while b.ndim < x_t.ndim:
            b = b.unsqueeze(-1)

        return a * x_t + b * f_A
    
        

    def _get_parameterization_conversion_coefficients(self, origin, target, t):
        """
        Get the coefficients to convert from one parameterization to another. See table 1 p. 33 in the Meta paper.
        
        Args:
            origin: The parameterization to convert from. Either x_0, x_1, score or velocity
            target: The parameterization to convert to. Either x_0, x_1, score or velocity
            t: The time step to sample at.
        """

        if origin == "velocity":
            origin = "v"
        if target == "velocity":
            target = "v"

        assert origin in ["v", "x_0", "x_1", "score"]
        assert target in ["v", "x_0", "x_1", "score"]

        a = torch.zeros_like(t)
        b = torch.zeros_like(t)

        # Get the alpha and sigma values at time t
        alpha = self.scheduler.alpha(t)
        alpha_dt = self.scheduler.alpha_dt(t)

        sigma = self.scheduler.sigma(t)
        sigma_dt = self.scheduler.sigma_dt(t)

        # We manage the singularities when alpha = 0 or sigma = 0 (i.e. t = 0 or t = 1)
        t_0_mask = torch.isclose(t, torch.zeros_like(t), atol=1e-3)
        t_1_mask = torch.isclose(t, torch.ones_like(t), atol=1e-3)

        # When t=0 or t=1, we can only convert to the velocity parameterization
        # In this case, b=0 and a=sigma_dt or a=alpha_dt
        if t_0_mask.any():
            assert target == "v" or target==origin, f"Cannot convert to {target} when t=0"
            a[t_0_mask] = sigma_dt[t_0_mask]
            b[t_0_mask] = 0.0

        if t_1_mask.any():
            assert target == "v" or target==origin, f"Cannot convert to {target} when t=1"
            a[t_1_mask] = alpha_dt[t_1_mask]
            b[t_1_mask] = 0.0

        # Coefficients for the conversion
        target_origin_coefficients = {
            "v": {                  # target
                "v": (0.0, 1.0),    # origin
                "x_1": (sigma_dt/sigma, (alpha_dt*sigma-sigma_dt*alpha)/sigma),
                "x_0": (alpha_dt/alpha, (sigma_dt*alpha-alpha_dt*sigma)/alpha),
                "score": (alpha_dt/alpha, -sigma * (sigma_dt*alpha-alpha_dt*sigma)/alpha),
            },
            "x_1": {
                "x_1": (0.0, 1.0),
                "x_0": (1/alpha, -sigma/alpha),
                "score": (1/alpha, -sigma**2/alpha),
            },
            "x_0": {
                "x_0": (0.0, 1.0),
                "score": (0, -sigma),
            },
            "score": {
                "score": (0.0, 1.0)
            }
        }

        mask = torch.logical_not(t_0_mask) & torch.logical_not(t_1_mask)

        if origin in target_origin_coefficients[target]:
            a_, b_ = target_origin_coefficients[target][origin]
        else:
            a_, b_ = target_origin_coefficients[origin][target]
            a_, b_ = -a_/b_, 1/b_

        a[mask], b[mask] = a_[mask], b_[mask]
    
        return a, b
    
    @staticmethod
    def _broadcast_coeff(coeff: Tensor, target_rank: int) -> Tensor:
        """Right‑broadcast *coeff* to match *target_rank*."""
        return coeff.view(-1, *[1] * (target_rank - coeff.ndim))


