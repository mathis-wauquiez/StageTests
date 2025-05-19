

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
from omegaconf import DictConfig

from .path import AffinePath
from .schedulers import Scheduler
from .models import ModelWrapper
from .utils import _to_tensor_scalar
from .types import Predicts, Guidance, FlowConfig


# -----------------------------------------------------------------------------
# Flow Class
# -----------------------------------------------------------------------------

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
        cfg: FlowConfig
            Configuration for the flow model, including prediction type and guidance.
        classifier : nn.Module | None, default=None
            Classifier used for classifier guidance.
        optimizer_cfg, scheduler_cfg, solver_cfg : dict | OmegaConf | None
            Hydra‑style instantiation configs.
            If None, defaults to the model's optimizer and scheduler.
        """
        super().__init__()

        # --------------------------- sanity checks ---------------------------
        
        guidance = cfg.guidance
        
        if guidance == Guidance.CLASSIFIER and classifier is None:
            raise ValueError("Classifier guidance requested but no classifier provided.")
        if guidance == Guidance.CFG and classifier is not None:
            raise ValueError("Cannot mix Classifier‑Free Guidance with an explicit classifier.")

        # --------------------------- members ---------------------------
        self.path = path
        self.model = torch.compile(model) if cfg.compile else model
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

    def estimated_velocity(self, t, x, y=None, **kwargs):
        """Network‑based estimate of the true velocity ``v_θ(t,x_t)``."""

        # Torchdiffeq passes scalars → make them batch tensors
        t = _to_tensor_scalar(t)    
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        source_parameterization = self.cfg.predicts


        # ---------------------------------------------------------------- cfg
        if self.cfg.guidance == Guidance.CFG and y is not None:
            # Run conditional & unconditional in *one* pass
            t_cat = torch.cat([t, t], dim=0)
            x_cat = torch.cat([x, x], dim=0)
            y_cat = torch.cat([y, y], dim=0)
            cond_mask = torch.tensor([1, 0], device=x.device, dtype=torch.bool).repeat_interleave(
                x.shape[0]
            )
            out = self.model(t_cat, x_cat, cond_mask=cond_mask, y=y_cat, **kwargs)

            # Convert to score
            out = self.path.convert_parameterization(
                t_cat, x_cat, out, source_parameterization, Predicts.SCORE
            )

            # Apply guidance
            cond, uncond = out.chunk(2)
            outputs = uncond + self.cfg.guidance_scale * (cond - uncond)
            source_parameterization = Predicts.SCORE

        
        # ------------------------------------------------------------ classifier
        elif self.cfg.guidance == Guidance.CLASSIFIER and y is not None:
            # --- unconditional score
            net_out = self.model(t, x, **kwargs)
            score  = self.path.convert_parameterization(
                t, x, net_out, self.cfg.predicts.value, Predicts.SCORE
            )

            # --- classifier gradient (re-enable grad)
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                logits = self.classifier(x_req)
                log_p  = torch.log_softmax(logits, dim=-1)[
                            torch.arange(len(y), device=x.device), y
                        ]
                grad_x = torch.autograd.grad(log_p.sum(), x_req)[0]

            grad_x = grad_x.detach()


            # eq 4.90 p.34

            outputs = self.path.convert_parameterization(
                t,
                x,
                f_A=score + self.cfg.guidance_scale * grad_x,
                source_parameterization=Predicts.SCORE,
                target_parameterization=Predicts.VELOCITY,
            )

            source_parameterization = Predicts.VELOCITY
        # ---------------------------------------------------------------- none
        else:
            if self.cfg.guidance == Guidance.CFG: # y is None and CFG is requested
                y = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
                cond_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
                kwargs.update({"y": y, "cond_mask": cond_mask})

            # No guidance
            outputs = self.model(t, x, **kwargs)
            source_parameterization = self.cfg.predicts

        # Final conversion → velocity
        return self.path.convert_parameterization(t, x, outputs, source_parameterization, Predicts.VELOCITY)


    # ------------------------------------------------------------------
    #  Prediction helpers
    # ------------------------------------------------------------------

    def predict_x0(self, t: Tensor, x: Tensor, **kw) -> Tensor:
        net_out = self.model(t, x, **kw)
        return self.path.convert_parameterization(t, x, net_out, self.cfg.predicts, Predicts.X0)

    def predict_x1(self, t: Tensor, x: Tensor, **kw) -> Tensor:
        net_out = self.model(t, x, **kw)
        return self.path.convert_parameterization(t, x, net_out, self.cfg.predicts, Predicts.X1)

    def predict_score(self, t: Tensor, x: Tensor, **kw) -> Tensor:
        net_out = self.model(t, x, **kw)
        return self.path.convert_parameterization(t, x, net_out, self.cfg.predicts, Predicts.SCORE)

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

        if n_steps == 50 and "n_steps" in solver_cfg:
            n_steps = solver_cfg.pop("n_steps")

        t_span = torch.linspace(0, 1, n_steps, device=x_0.device)

        if "method" not in solver_cfg:
            solver_cfg["method"] = "midpoint"

        velocity_field = lambda t, x: self.estimated_velocity(t, x, y=y)


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
    
    def forward(self, *args, **kwargs) -> Tensor:
        return self.sample(*args, **kwargs)

    # ------------------------------------------------------------------
    #  Training / validation
    # ------------------------------------------------------------------

    def _get_loss(self, x_0: Tensor, x_1: Tensor, t: Tensor, x_t: Tensor, **kwargs) -> Tensor:
        
        if self.cfg.guidance == Guidance.CFG:
            random_mask = torch.rand(x_0.shape[0], device=x_0.device) < self.cfg.guided_prob
            kwargs.update({'cond_mask': random_mask})

        if self.cfg.predicts == Predicts.X0:
            return self.loss_fn(self.model(t, x_t, **kwargs), x_0)
        elif self.cfg.predicts == Predicts.X1:
            return self.loss_fn(self.model(t, x_t, **kwargs), x_1)
        elif self.cfg.predicts == Predicts.SCORE:
            pred_score = self.model(t, x_t, **kwargs)
            v_theta = self.path.convert_parameterization(t, x_t, pred_score, "score", "v")
            v_target = self.target_velocity(t, x_0, x_1)
            return self.loss_fn(v_theta, v_target)
        elif self.cfg.predicts == Predicts.VELOCITY:
            v_theta = self.model(t, x_t, **kwargs)
            v_target = self.path.target_velocity(t, x_0, x_1)
            return self.loss_fn(v_theta, v_target)
        else:
            raise RuntimeError("Unknown prediction type.")


    # ------------------------------------------------------------------
    #  Basic Lightning Hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x_0, x_1, y = batch
        else:
            x_0, x_1 = batch
            y = None

        pass_y = y if self.cfg.guidance in (Guidance.CFG, Guidance.CLASSIFIER) else None

        t, x_t = self.path.sample(x_0, x_1)
        loss = self._get_loss(x_0, x_1, t, x_t, y=y) if pass_y is not None else self._get_loss(
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
    
    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x_0, x_1, y = batch
        else:
            x_0, x_1 = batch
            y = None

        t, x_t = self.path.sample(x_0, x_1)
        loss = self._get_loss(x_0, x_1, t, x_t, y=y) if y is not None else self._get_loss(
            x_0, x_1, t, x_t
        )
        self.log_dict({"test_loss": loss}, prog_bar=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    #  Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        
        # 1) optimizer
        opt_cfg = self.optimizer_cfg
        if isinstance(opt_cfg, (DictConfig, dict)):
            optimizer = instantiate(opt_cfg, params=self.parameters())
        else:                                 # already a partial or callable
            optimizer = opt_cfg(params=self.parameters())

        # 2) scheduler
        # if self.scheduler_cfg is not None:
        #     scheduler = _make_scheduler(self.scheduler_cfg, optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "interval": "step",
        #             "frequency": 1,
        #         },
        #     }
        return optimizer

        return optimizer
    
from functools import partial
from typing import Any, Dict
import torch.optim.lr_scheduler as lrs
import hydra

CONTAINER_TYPES = (
    lrs.SequentialLR,          # warm-up → cosine, chained, etc.
    lrs.ChainedScheduler,
)


def _make_scheduler(cfg, optimizer):
    """
    Instantiate *leaf* and *container* LR-schedulers, whether the config comes
    in as a DictConfig/dict or a functools.partial.

    • Leaf scheduler  – receives `optimizer` and is returned.
    • Container       – children are built first, then passed in.
    """
    if isinstance(cfg, partial):
        cls = cfg.func        # the actual scheduler class

        if cls in CONTAINER_TYPES:
            child_cfgs = cfg.keywords.pop("schedulers")
            children = [_make_scheduler(c, optimizer) for c in child_cfgs]

            # instantiate the container with the finished children
            return cls(optimizer=optimizer, schedulers=children, **cfg.keywords)

        return cfg(optimizer=optimizer)

    if isinstance(cfg, (DictConfig, dict)):
        target = cfg.get("_target_", "")
        # map string class names to real classes for the container test
        cls = hydra.utils.get_class(target)
        if cls in CONTAINER_TYPES:
            child_cfgs = cfg.pop("schedulers")
            children = [_make_scheduler(c, optimizer) for c in child_cfgs]
            return instantiate(
                cfg, optimizer=optimizer, schedulers=children, _recursive_=False
            )
        return instantiate(cfg, optimizer=optimizer)

    raise TypeError(f"Unsupported scheduler_cfg type: {type(cfg)}")