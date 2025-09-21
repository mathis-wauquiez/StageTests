from torch.distributions.uniform import Uniform
import torch
from torch import Tensor
from typing import Union, Tuple



# See https://arxiv.org/abs/2403.03206, section 3.1
# Best results are usually obtained with LogitNormal(loc=0.0, scale=1.00)


default = Uniform(low=0.0, high=1.0)

class LogitNormal:
    """
    Logit–Normal distribution π_ln(t ; m, s)

    Parameters
    ----------
    loc  : float | Tensor
        Location parameter (μ in the underlying Normal)
    scale: float | Tensor
        Scale     parameter (σ in the underlying Normal); must be positive
    device / dtype follow normal PyTorch broadcasting semantics.

    Example
    -------
    >>> dist = LogitNormal(loc=0.0, scale=1.5, device="cuda")
    >>> samples = dist.sample((128, 64))      # (128, 64) batch of t \in (0,1)
    """
    def __init__(
        self,
        loc:   Union[float, Tensor],
        scale: Union[float, Tensor],
        *,
        device: Union[str, torch.device] | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.loc   = torch.as_tensor(loc,   device=device, dtype=dtype)
        self.scale = torch.as_tensor(scale, device=device, dtype=dtype)

        if torch.any(self.scale <= 0):
            raise ValueError("scale (s) must be strictly positive")

    def sample(self, shape: Union[Tuple[int, ...], torch.Size] = torch.Size()) -> Tensor:
        """
        Draw samples with the requested leading shape (e.g. (batch, dim …)).

        Parameters
        ----------
        shape : tuple | torch.Size
            Desired sample shape.  The final broadcasted shape is
            `shape + torch.broadcast_shapes(self.loc.shape, self.scale.shape)`.
        """
        eps = torch.randn(*shape, *self.loc.shape, device=self.loc.device, dtype=self.loc.dtype)
        u   = self.loc + self.scale * eps          # reparameterised Normal(m,s)
        return torch.sigmoid(u)                    # logit^{-1}(u) → (0,1)


    def log_prob(self, t: Tensor) -> Tensor:
        """
        Log-probability at points `t` in (0,1).  Useful if you want to compute
        losses; not required for simple sampling.
        """
        if torch.any((t <= 0) | (t >= 1)):
            raise ValueError("All t must satisfy 0 < t < 1")

        # logit transform
        u = torch.log(t) - torch.log1p(-t)
        # Normal log-pdf
        norm_logpdf = -0.5 * ((u - self.loc) / self.scale)**2 \
                      - torch.log(self.scale) \
                      - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=t.device))
        # Jacobian adjustment |d logit^{-1}(u)/du| = t(1-t)
        return norm_logpdf - torch.log(t) - torch.log1p(-t)