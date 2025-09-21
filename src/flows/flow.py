

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
        path: AffinePath,
        loss_fn: nn.Module,
        model: ModelWrapper,
        cfg: FlowConfig,
        classifier: Optional[nn.Module] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
        solver_cfg: Optional[dict] = None,
        pass_y: Optional[bool] = False
    ) -> None:
        """
        Time‑continuous flow model with optional guidance, trained using Flow Matching.

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

        if cfg.guidance == Guidance.CLASSIFIER and classifier is None:
            raise ValueError("Classifier guidance requires a classifier.")
        if cfg.guidance == Guidance.CFG and classifier is not None:
            raise ValueError("Cannot combine CFG with explicit classifier.")

        # --------------------------- members ---------------------------
        
        self.path = path
        self.cfg = cfg
        self.device_auto = None
        self._compile = cfg.compile

        self.model = model
        self.loss_fn = loss_fn
        self.classifier = classifier
        self.pass_y = pass_y

        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.solver_cfg: Dict[str, Any] = solver_cfg or {}

        # --------------------------- hparams snapshot ---------------------------
        # self.save_hyperparameters(ignore=["model", "classifier"])


    # ------------------------------------------------------------------
    # Velocity field
    # ------------------------------------------------------------------

    def estimated_velocity(self, t, x, y=None, **kwargs):
        """Network‑based estimate of the true velocity ``v_θ(t,x_t)``."""

        # High-level overview:
        # 1. If we are using guidance, we need to
        #    - get the outputs from the model with and without conditioning
        #    - convert the outputs to the score function
        #    - apply the guidance formula to get the final score estimation
        #    - convert the score to the velocity field
        # 2. If we are using classifier guidance, we need to
        #    - get the outputs from the model and convert them to the score function
        #    - compute the classifier log_p gradient
        #    - apply the guidance formula to get the final velocity estimation
        # 3. If no guidance is used or that y is None, we simply
        #    - get the outputs from the model and convert them to the velocity field.


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
            net_out = self.model(t_cat, x_cat, cond_mask=cond_mask, y=y_cat, **kwargs)

            # Convert to score
            scores = self.path.convert_parameterization(
                t_cat, x_cat, net_out, source_parameterization, Predicts.SCORE
            )

            # Apply guidance
            cond, uncond = scores.chunk(2)
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
            outputs = score + self.cfg.guidance_scale * grad_x
            source_parameterization = Predicts.SCORE
        # ---------------------------------------------------------------- none
        else:
            if self.cfg.guidance == Guidance.CFG: # y is None and CFG is used, we fall back to unconditional
                y = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
                cond_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
                kwargs.update({"y": y, "cond_mask": cond_mask})

            if self.pass_y:
                # If `pass_y` is True, we pass `y` to the model anyway
                kwargs.update({"y": y})

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
        t_span: Optional[Tensor] = None,
        n_steps: Optional[int] = None,
        y: Optional[Tensor] = None,
        **solver_cfg: Any,
    ) -> tuple[Tensor, Tensor]:
        """Sample a trajectory by solving the ODE from t=0 to t=1 using torchdiffeq's odeint."""

        assert t_span or n_steps, "Either `t_span` or `n_steps` must be provided."

        if y is None and self.pass_y:
            raise ValueError("`y` must be provided if `pass_y` is True.")

        # If t_span is not provided, create it
        if n_steps is not None and t_span is None:
            t_span = torch.linspace(0, 1, n_steps, device=x_0.device)

        # Merge solver configurations - if no method is provided, default to 'dopri5'
        solver_cfg = {'method':'dopri5'} | self._merge_config(solver_cfg)

        velocity_field = lambda t, x: self.estimated_velocity(t, x, y=y)

        with torch.no_grad():
            trajectory = odeint(
                velocity_field,
                x_0,
                t_span,
                **solver_cfg
            )
        
        return trajectory
    
    def sample(self, x_0: Tensor, *, y: Optional[Tensor] = None, **solver_cfg: Any) -> Tensor:
        """Sample a single final state by solving the ODE from t=0 to t=1."""

        if y is None and self.pass_y:
            raise ValueError("`y` must be provided if `pass_y` is True.")

        t_span = torch.tensor([0., 1.], device=x_0.device)
        solver_cfg = {'method':'dopri5'} | self._merge_config(solver_cfg)

        # Condition the velocity field on `y` if provided
        velocity_field = lambda t, x: self.estimated_velocity(t, x, y=y)

        # Solve the ODE
        with torch.no_grad():
            trajectory = odeint(
                velocity_field,
                x_0,
                t_span,
                **solver_cfg
            )
        
        return trajectory[-1, ...] # Trajectory is of shape (2, BS, ...)
    

    def forward(self, *args, **kwargs) -> Tensor:
        return self.sample(*args, **kwargs)

    # ------------------------------------------------------------------
    #  Training / validation
    # ------------------------------------------------------------------

    def _get_loss(self, x_0: Tensor, x_1: Tensor, t: Tensor, x_t: Tensor, **kwargs) -> Tensor:
        """ Compute the loss for the given inputs. """

        # There are two cases me might want to consider:
        # 1. We want to define the loss wrt to the model prediction of x_0, x_1, v or score.
        # 2. We want to define the loss wrt to the prediction of the velocity field.
        # We might also want to add a time coefficient to the loss, which has been shown to improve the results in some cases.

        
        if self.cfg.guidance == Guidance.CFG:
            random_mask = torch.rand(x_0.shape[0], device=x_0.device) < self.cfg.guided_prob
            kwargs.update({'cond_mask': random_mask})

        outputs = self.model(t, x_t, **kwargs)
        v_theta = self.path.convert_parameterization(t, x_t, outputs, self.cfg.predicts, Predicts.VELOCITY)
        v_target = self.path.target_velocity(t, x_0, x_1)
        return self.loss_fn(v_theta, v_target)

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

    def _step(self, batch, batch_idx):
        """
        Perform a single training/validation/test step to get the loss.
        This is really the default function, and of course validation_step and test_step can be overridden
        """

        if len(batch) == 3:
            x_0, x_1, y = batch
        else:
            x_0, x_1 = batch
            y = None

        t, x_t = self.path.sample(x_0, x_1) # Sample a random point and time

        if self.cfg.guidance in (Guidance.CFG, Guidance.CLASSIFIER) or self.pass_y:
            if y is None and self.cfg.guidance != Guidance.CLASSIFIER:
                raise ValueError("`y` must be provided if `pass_y` is True or classifier-free guidance is used.")
            return self._get_loss(x_0, x_1, t, x_t, y=y)

        return self._get_loss(x_0, x_1, t, x_t)

        
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log_dict({"val_loss": loss}, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log_dict({"test_loss": loss}, prog_bar=True, on_epoch=True)
        return loss


    def on_fit_start(self) -> None:
        # Move and compile after device is known
        self.model = self.model.to(self.device)
        if self._compile:
            self.model = torch.compile(self.model)
        if self.classifier:
            self.classifier = self.classifier.to(self.device).eval()
            for p in self.classifier.parameters():
                p.requires_grad = False

    def on_train_epoch_start(self) -> None:
        self.path._tol = self.path.tol
        self.path.tol = 0

    def on_train_epoch_end(self) -> None:
        self.path.tol = self.path._tol


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
        if self.scheduler_cfg:
            scheduler = _make_scheduler(self.scheduler_cfg, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return optimizer

    # ------------------------------------------------------------------
    #  Class helpers
    # ------------------------------------------------------------------

    def _merge_config(self, cfg: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Merge the provided config with the flow's solver config."""
        solver_cfg = self.solver_cfg if isinstance(self.solver_cfg, dict) else OmegaConf.to_container(self.solver_cfg)
        merged_cfg = solver_cfg | cfg | kwargs
        return merged_cfg




# -----------------------------------------------------------------------------
#  Helper functions for LR-schedulers
# -----------------------------------------------------------------------------

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