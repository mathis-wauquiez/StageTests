from abc import ABC, abstractmethod
from typing import Literal, Optional

from torch import nn, Tensor


from .schedulers import Scheduler
from .models import ModelWrapper
from torch.distributions import Distribution
from torch.distributions.uniform import Uniform



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
        t_law (Optional[Distribution], optional): A PyTorch distribution to sample random time values from. Defaults to Uniform(0, 1).
        classifier (Optional[nn.Module], optional): A classifier network used if `guidance="classifier"` is specified.

    Raises:
        ValueError: If classifier guidance is selected without providing a classifier.

    Methods:
        sample(x_0, x_1, t=None):
            Samples an intermediate point x_t between x_0 and x_1 at time t.
        
        estimated_velocity(x, t, **kwargs):
            Estimates the velocity at time t using the model prediction.

        convert_parameterization(x_t, t, f_A, source_parameterization, target_parameterization):
            Converts a model prediction from one parameterization to another.

    Example:
        >>> path = AffinePath(scheduler, model, predicts="score")
        >>> x_t, t = path.sample(x_0, x_1)
        >>> v_t = path.estimated_velocity(x_t, t=t)
        >>> x_0_estimated = path.convert_parameterization(x_t, t, v_t, "v", "x_0")
        """
            
    def __init__(
            self,
            scheduler: Scheduler,
            model: ModelWrapper,
            predicts: Literal["x_0", "x_1", "score"],
            guidance: Optional[Literal["CFG", "classifier"]] = None,
            t_law: Optional[Distribution] = Uniform(low=0.0, high=1.0),
            classifier: Optional[nn.Module] = None,
    ):
        
        assert predicts in ["x_0", "x_1", "score"]
        assert guidance in [None, "CFG", "classifier"]

        if guidance == "classifier" and classifier is None:
            raise ValueError("If classifier guidance is used, a classifier must be provided")

        self.scheduler = scheduler
        self.model = model
        
        self.predicts = predicts
        self.guidance = guidance
        self.t_law = t_law

    def sample(self, x_0, x_1, t=None):
        """
        Sample from the conditional path.

        Args:
            x_0: Tensor (bs, ...), the initial state
            x_1: Tensor (bs, ...), the final state
            t: Optional[Scalar], the time step to sample at. If None, sample from the t_law.
        """
        if t is None:
            t = self.t_law.sample()

        if t.dim() > 0:
            raise ValueError("t must be a scalar")
        
        if x_0.shape != x_1.shape:
            raise ValueError("x_0 and x_1 must be the same shape")
    
        x_t = self.scheduler.sample(x_0, x_1, t)

        return x_t, t
    

    def estimated_velocity(self, x, t, **kwargs):
        """
        Estimate the velocity at time t.

        Args:
            x: Tensor (bs, ...), the state at time t
            t: Scalar, the time step to estimate at.
            **kwargs: Additional arguments to pass to the model
        """
        
        outputs = self.model(x, t, **kwargs)
        velocity = self.convert_parameterization(x, t, outputs, self.predicts, "v")
        return velocity

    def convert_parameterization(self, x_t, t, f_A, source_parameterization, target_parameterization):
        """
        Convert one type of parameterization to another.
        Can be used to convert between x_0, x_1, score and velocity, according to the table 1 in the Meta paper.

        Args:
            x_t: The value of the path at time t
            f_A: The value to convert
            source_parameterization: The parameterization of the value to convert. Either x_0, x_1, score or velocity
            target_parameterization: The parameterization to convert to. Either x_0, x_1, score or velocity
        """

        assert source_parameterization in ["v", "x_0", "x_1", "score", "velocity"], f"{source_parameterization} is not a valid parameterization"
        assert target_parameterization in ["v", "x_0", "x_1", "score", "velocity"], f"{target_parameterization} is not a valid parameterization"
        assert x_t.shape == f_A.shape, f"x_t and f_A must be the same shape, got {x_t.shape} and {f_A.shape}"

        a, b = self._get_parameterization_conversion_coefficients(source_parameterization, target_parameterization, t)

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

        alpha = self.scheduler.alpha(t)
        alpha_dt = self.scheduler.alpha_dt(t)

        sigma = self.scheduler.sigma(t)
        sigma_dt = self.scheduler.sigma_dt(t)

        coefficients = {
            "v": {                  # target
                "v": (0.0, 1.0),    # origin
                "x_0": (sigma_dt/sigma, (alpha_dt*sigma-sigma_dt*alpha)/sigma),
                "x_1": (alpha_dt/alpha, (sigma_dt*alpha-alpha_dt*sigma)/alpha),
                "score": (alpha_dt/alpha, -(sigma_dt*sigma*alpha-alpha_dt*sigma**2)/alpha),
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

        if origin not in coefficients[target]:
            a, b = coefficients[origin][target]
            a, b = -a/b, 1/b
        
        else:
            a, b = coefficients[origin][target]

        return a, b

