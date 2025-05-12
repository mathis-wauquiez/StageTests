

from functools import partial
from enum import Enum
from typing import Optional, Dict, Any, Generator, Literal, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor

from torchdiffeq import odeint
from pytorch_lightning import LightningModule

from hydra.utils import instantiate
from omegaconf import OmegaConf

from .path import AffinePath
from .schedulers import Scheduler
from .models import ModelWrapper
from .utils import _to_tensor_scalar

# -----------------------------------------------------------------------------
# Enumerations
# -----------------------------------------------------------------------------

Predicts = Literal["x_0", "x_1", "score"]

class Predicts(str, Enum):
    X0 = "x_0"
    X1 = "x_1"
    SCORE = "score"

    _ALIASES: Dict[str, Predicts] = {"x0": X0, "x_0": X0, "x1": X1, "x_1": X1, "score": SCORE}

    @classmethod
    def from_any(cls, value: Predicts | str) -> Predicts:
        if isinstance(value, cls):
            return value
        return cls._ALIASES[str(value).lower()]


class Guidance(str, Enum):
    NONE = "none"
    CFG = "CFG"
    CLASSIFIER = "classifier"


# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------

@dataclass
class FlowConfig:
    predicts: Predicts = Predicts.X1
    guidance: Guidance = Guidance.NONE
    guidance_scale: float = 1.0
    compile: bool = True  # ← new optional flag

class Flow(LightningModule):

    def __init__(
        self,
        *,
        path: AffinePath,
        loss_fn: nn.Module,
        model: ModelWrapper,
        cfg: FlowConfig,
        classifier: Optional[nn.Module] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
        solver_cfg: Optional[dict] = None,
    ) -> None:
        """
        Time‑continuous flow model with optional guidance.

        Parameters
        ----------
        path : AffinePath
            Encapsulates the scheduler and the interpolation X_t | X_0,.
        loss_fn : nn.Module
            Supervised loss applied to the chosen prediction target.
        model : ModelWrapper
            Neural network that predicts ``x_0``, ``x_1`` or the score.
        predicts : Predicts, default=Predicts.SCORE
            What *kind* of object ``model`` outputs.
        guidance : Guidance, default=Guidance.NONE
            Guidance strategy to use **during sampling only**.
        guidance_scale : float, default=1.0
            Scale for guidance (see DDPM / CFG literature).
        classifier : nn.Module | None, default=None
            Classifier used for classifier guidance.
        optimizer_cfg, scheduler_cfg, solver_cfg : dict | OmegaConf | None
            Hydra‑style instantiation configs.
        """
        super().__init__()

        # --------------------------- canonicalise enums ---------------------------
        predicts = Predicts.from_any(predicts)
        if guidance is None:
            guidance = Guidance.NONE
        else:
            guidance = Guidance(guidance) if not isinstance(guidance, Guidance) else guidance

        # --------------------------- sanity checks ---------------------------
        if guidance == Guidance.CLASSIFIER and classifier is None:
            raise ValueError("Classifier guidance requested but no classifier provided.")
        if guidance == Guidance.CFG and classifier is not None:
            raise ValueError("Cannot mix Classifier‑Free Guidance with an explicit classifier.")
        if guidance == Guidance.CLASSIFIER and predicts not in (Predicts.X0, Predicts.SCORE):
            raise ValueError("Classifier guidance needs x_0 or score prediction.")

        # --------------------------- members ---------------------------
        self.path = path
        self.model = torch.compile(model) if self.cfg.compile else model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.classifier = classifier.to(self.device).eval() if classifier else None
        if classifier:
            for p in classifier.parameters():
                p.requires_grad = False


        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.solver_cfg: Dict[str, Any] = solver_cfg or {}

        # --------------------------- hparams snapshot ---------------------------
        self.save_hyperparameters(ignore=["model", "classifier"])


    # ------------------------------------------------------------------
    # Velocity field
    # ------------------------------------------------------------------

    def estimated_velocity(self, t, x, **kwargs):
        """Network‑based estimate of the true velocity ``v_θ(t,x_t)``."""

        # Torchdiffeq passes scalars → make them batch tensors
        t = _to_tensor_scalar(t)    
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        predicts = self.cfg.predicts

        # ---------------------------------------------------------------- cfg
        if self.cfg.guidance == Guidance.CFG:

            if "y" not in kwargs:
                raise ValueError("Classifier-free guidance requires labels 'y'.")

            # Run conditional & unconditional in *one* pass
            t_cat = torch.cat([t, t], dim=0)
            x_cat = torch.cat([x, x], dim=0)
            cond_mask = torch.tensor([1, 0], device=x.device, dtype=torch.bool).repeat_interleave(
                x.shape[0]
            )
            out = self.model(t_cat, x_cat, cond_mask=cond_mask, **kwargs)

            # Convert to score
            out = self.path.convert_parameterization(
                t_cat, x_cat, out, predicts, "score"
            )

            # Apply guidance
            cond, uncond = out.chunk(2)
            outputs = uncond + self.cfg.guidance_scale * (cond - uncond)
            predicts = "score"

        
        # ------------------------------------------------------------ classifier
        elif self.cfg.guidance == Guidance.CLASSIFIER:
            if "y" not in kwargs:
                raise ValueError("Classifier guidance requires class labels 'y'.")
            y = kwargs.pop("y")
            
            # Calculate the unconditional score
            net_out = self.model(t, x, **kwargs)
            score = self.path.convert_parameterization(t, x, net_out, self.cfg.predicts.value, "score")

            # Calculate ∇ₓ log p(y|x)
            x_req = x.detach().requires_grad_()
            logits = self.classifier(x_req)  # shape: (bs, num_classes)
            log_p = torch.log_softmax(logits, dim=-1)[torch.arange(len(y), device=x.device), y]
            grad_x = torch.autograd.grad(log_p.sum(), x_req, create_graph=False)[0].detach()

            _, b = self.path._get_parameterization_conversion_coefficients(self.cfg.predicts.value, "score", t)
            outputs = score + b * self.cfg.guidance_scale * grad_x # eq 4.87 p.33
            predicts = Predicts.SCORE

        # ---------------------------------------------------------------- none
        else:
            outputs = self.model(t, x, **kwargs)
            predicts = self.cfg.predicts

        # Final conversion → velocity
        source_parameterization = predicts.value if isinstance(predicts, Enum) else predicts
        return self.path.convert_parameterization(t, x, outputs, source_parameterization, "v")


    # ------------------------------------------------------------------
    #  Prediction helpers
    # ------------------------------------------------------------------

    def predict_x0(self, t: Tensor, x: Tensor, **kw) -> Tensor:
        net_out = self.model(t, x, **kw)
        return self.path.convert_parameterization(t, x, net_out, self.predict, Predicts.X0)

    def predict_x1(self, t: Tensor, x: Tensor, **kw) -> Tensor:
        net_out = self.model(t, x, **kw)
        return self.path.convert_parameterization(t, x, net_out, self.predict, Predicts.X1)

    def predict_score(self, t: Tensor, x: Tensor, **kw) -> Tensor:
        net_out = self.model(t, x, **kw)
        return self.path.convert_parameterization(t, x, net_out, self.predict, Predicts.SCORE)

    def predict_v(self, t: Tensor, x: Tensor, **kw) -> Tensor:
        """ Predicts the conditional velocity field."""
        return self.estimated_velocity(t, x, **kw)

    def predict(self, t: Tensor, x: Tensor, kind: Optional[Predicts | str] = None, **kw) -> Tensor:
        """Unified entry point—*kind* may be Enum, canonical name, or alias."""
        kind_resolved = Predicts.from_any(kind) if kind is not None else self.cfg.predicts
        method = f"predict_{kind_resolved.value.replace('_', '').lower()}"
        return getattr(self, method)(t, x, **kw)

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------

    def sample_trajectory(
        self,
        x_0: Tensor,
        *,
        n_steps: int = 50,
        y: Optional[Tensor] = None,
        **solver_cfg: Any,
    ) -> tuple[Tensor, Tensor]:
        """Sample a trajectory by solving the ODE from t=0 to t=1 using torchdiffeq's odeint."""
        
        # Merge solver configurations
        solver_cfg = {"method": "midpoint", **self.solver_cfg, **solver_cfg}

        t_span = torch.linspace(0, 1, n_steps, device=x_0.device)

        if "method" not in solver_cfg:
            solver_cfg["method"] = "midpoint"

        # Define the velocity field depending on guidance
        if self.cfg.guidance in (Guidance.CFG, Guidance.CLASSIFIER):
            if y is None:
                raise ValueError("Guidance requires labels 'y'.")
            velocity_field = lambda t, x: self.estimated_velocity(t, x, y=y)
        else:
            velocity_field = self.estimated_velocity


        with torch.no_grad():
            trajectory = odeint(
                velocity_field,
                x_0,
                t_span,
                **solver_cfg
            )
        
        return trajectory, t_span
    
    def sample(self, x_0: Tensor, *, n_steps: int = 50, y: Optional[Tensor] = None, **solver_cfg: Any) -> Tensor:
        """Sample a single final state by solving the ODE from t=0 to t=1."""
        trajectory, _ = self.sample_trajectory(x_0, n_steps=n_steps, y=y, **solver_cfg)
        return trajectory[-1]
    
    
    # ------------------------------------------------------------------
    #  Training / validation
    # ------------------------------------------------------------------

    def _get_loss(self, x_0: Tensor, x_1: Tensor, t: Tensor, x_t: Tensor, **kwargs) -> Tensor:
        if self.cfg.predicts == Predicts.X0:
            return self.loss_fn(self.model(t, x_t, **kwargs), x_0)
        elif self.cfg.predicts == Predicts.X1:
            return self.loss_fn(self.model(t, x_t, **kwargs), x_1)
        elif self.cfg.predicts == Predicts.SCORE:
            pred_score = self.model(t, x_t, **kwargs)
            v_theta = self.path.convert_parameterization(t, x_t, pred_score, "score", "v")
            v_target = self.target_velocity(t, x_0, x_1)
            return self.loss_fn(v_theta, v_target)
        else:
            raise RuntimeError("Unknown prediction type.")


    # Basic Lightning hooks ----------------------------------------------------

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x_0, x_1, y = batch
        else:
            x_0, x_1 = batch
            y = None

        t, x_t = self.path.sample(x_0, x_1)
        loss = self._get_loss(x_0, x_1, t, x_t, y=y) if y is not None else self._get_loss(
            x_0, x_1, t, x_t
        )
        self.log_dict({"train_loss": loss}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x_0, x_1, y = batch
        else:
            x_0, x_1 = batch
            y = None

        t, x_t = self.path.sample(x_0, x_1)
        loss = self._get_loss(x_0, x_1, t, x_t, y=y) if y is not None else self._get_loss(
            x_0, x_1, t, x_t
        )
        self.log_dict({"val_loss": loss}, prog_bar=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    #  Optimiser & scheduler
    # ------------------------------------------------------------------


    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        if self.scheduler_cfg:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        return optimizer
