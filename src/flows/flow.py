import torch

from .path import AffinePath
from .schedulers import Scheduler
from .models import ModelWrapper
from typing import Optional, Literal
from torch import nn

from torchdiffeq import odeint
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pytorch_lightning import LightningModule
from torch import Tensor


from .utils import _to_tensor_scalar

class Flow(LightningModule):

    def __init__(
            self,
            path: AffinePath,
            loss_fn: nn.Module,
            model: ModelWrapper,
            predicts: Literal["x_0", "x_1", "score"],
            guidance: Optional[Literal["CFG", "classifier"]] = None,
            guidance_scale: float = 1.0,
            classifier: Optional[nn.Module] = None,
            optimizer_cfg: Optional[dict] = None,
            scheduler_cfg: Optional[dict] = None,
            solver_cfg: Optional[dict] = None
    ):
        """
        Flow class for training and sampling from a flow model.

        Args:
        - path: AffinePath, the path object that defines the flow, the model and the scheduler
        - loss_fn: nn.Module, the loss function to use for training
        - device: str, the device to use for training (default: "cuda")
        """
        super().__init__()

        assert predicts in ["x_0", "x_1", "score"]
        assert guidance in [None, "CFG", "classifier"]

        if guidance == "classifier" and classifier is None:
            raise ValueError("If classifier guidance is used, a classifier must be provided")

        if guidance == "classifier" and predicts not in ["x_0", "score"]:
            raise ValueError("Classifier guidance can only be used with x_0 or score predictions")

        if guidance == "CFG" and classifier is not None:
            raise ValueError("Cannot use both CFG and classifier guidance at the same time")

        if classifier is not None:
            classifier = classifier.eval().to(self.device)
        
        self.model = model
        
        self.predicts = predicts
        self.guidance = guidance
        self.guidance_scale = guidance_scale
        self.classifier = classifier


        self.save_hyperparameters("predicts", "guidance", "guidance_scale")
        self.path = path
        self.loss_fn = loss_fn
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.solver_cfg = solver_cfg

    def estimated_velocity(self, t, x, **kwargs):
        """
        Estimate the velocity at time t, :math:`u^\theta_t(x_t)`.

        Args:
            x: Tensor (bs, ...), the state at time t
            t: Tensor (bs, ...) or Scalar, the time step
            **kwargs: Additional arguments to pass to the model
        """
        t = _to_tensor_scalar(t)
    
        if t.dim() == 0: # Useful for simulating the ODE with torchdiffeq
            t = t.unsqueeze(0).expand(x.shape[0])

        predicts = self.predicts

        if self.guidance == "CFG": # Classifier-Free Guidance
            outputs = self.model(t, x, **kwargs)
            outputs_uncond = self.model(t, x, uncond=True, **kwargs)
            outputs = outputs + self.guidance_scale * (outputs - outputs_uncond)
        
        elif self.guidance == "classifier":  # Classifier Guidance
            
            assert "y" in kwargs, "Classifier guidance requires 'y' in kwargs (class labels)."
            
            x = x.detach().requires_grad_()  # Make x differentiable
            outputs = self.model(t, x, **kwargs)
            score = self.path.convert_parameterization(t, x, outputs, self.predicts, "score")

            logits = self.classifier(x)  # shape: (bs, num_classes)
            y = kwargs["y"]              # shape: (bs,) – class indices

            # Compute log p(y|x)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(len(y)), y]  # shape: (bs,)

            # Backprop to get ∇ₓ log p(y|x)
            grads = torch.autograd.grad(
                selected_log_probs.sum(), x, create_graph=False
            )[0]

            grads = grads.detach()

            _, b = self.path._get_parameterization_conversion_coefficients(self.predicts, "score", t)
            # Add classifier gradient to model output
            # eq 4.87 p.33
            outputs = score + b * self.guidance_scale * grads
            predicts = "score"
        else:
            outputs = self.model(t, x, **kwargs)

        # Convert the output to the desired parameterization
        velocity = self.path.convert_parameterization(t, x, outputs, predicts, "v")
        return velocity
    

    def get_x0(self, t, x, **kwargs):
        v = self.estimated_velocity(t, x, **kwargs)
        return self.path.convert_parameterization(t, x, v, "v", "x_0")
    
    def get_x1(self, t, x, **kwargs):
        v = self.estimated_velocity(t, x, **kwargs)
        return self.path.convert_parameterization(t, x, v, "v", "x_1")
    
    def get_score(self, t, x, **kwargs):
        v = self.estimated_velocity(t, x, **kwargs)
        return self.path.convert_parameterization(t, x, v, "v", "score")
    
    def get_v(self, t, x, **kwargs):
        v = self.estimated_velocity(t, x, **kwargs)
        return v

    def target_velocity(self, t, x_0, x_1):
        """
        Get the target velocity at time t.

        Args:
            x_0: Tensor (bs, ...), the initial state
            x_1: Tensor (bs, ...), the final state
            t: Tensor (bs, ...), the time step
        """
        if x_0.shape != x_1.shape:
            raise ValueError("x_0 and x_1 must be the same shape")

        if t.dim() != 1:
            raise ValueError("t must be a 1D tensor")
        
        alpha_dt = self.path.scheduler.alpha_dt(t)
        sigma_dt = self.path.scheduler.sigma_dt(t)

        # Ensure proper shape for broadcasting
        while alpha_dt.ndim < x_1.ndim:
            alpha_dt = alpha_dt.unsqueeze(-1)
        while sigma_dt.ndim < x_0.ndim:
            sigma_dt = sigma_dt.unsqueeze(-1)

        # Calculate the target velocity
        velocity = alpha_dt * x_1 + sigma_dt * x_0

        return velocity



    def sample_trajectory(self, x_0: torch.Tensor, n_steps: Optional[int]=50, **solver_cfg):

        t = torch.linspace(0, 1, n_steps, device=self.device)

        if self.solver_cfg is not None:
            solver_cfg = OmegaConf.to_container(self.solver_cfg).update(solver_cfg)

        if "method" not in solver_cfg:
            solver_cfg["method"] = "midpoint"

        if self.guidance == "classifier":
            if not "args" in solver_cfg or not "y" in solver_cfg["args"]:
                raise ValueError("Classifier guidance requires 'y' in solver_cfg['args'] (odeint(func,y0,t,args=(123, 456)))")

        with torch.no_grad():
            trajectory = odeint(
                self.estimated_velocity,
                x_0,
                t,
                **solver_cfg
            )
        
        return trajectory, t
    
    def sample(self, x_0: torch.Tensor, n_steps: Optional[int]=50, **solver_cfg):
        """
        Sample from the flow model.
        
        Args:
            x_0: Tensor (bs, ...), the initial state
            n_steps: int, the number of steps to sample
            solver_cfg: dict, additional arguments for the solver
        """
        return self.sample_trajectory(x_0, n_steps, **solver_cfg)[0][-1]
    
    def _get_loss(self, x_0, x_1, t, x_t, **kwargs):
        """
        Compute the loss for the flow model.

        Args:
            x_0: Tensor (bs, ...), the initial state
            x_1: Tensor (bs, ...), the final state
            t: Tensor (bs, ...), the time step
            x_t: Tensor (bs, ...), the state at time t
            
        Returns:
            loss: Tensor (1,), the computed loss
        """

        if self.predicts == "x_0":
            outputs = self.model(t, x_t, **kwargs)
            return self.loss_fn(outputs, x_0)
        elif self.predicts == "x_1":
            outputs = self.model(t, x_t, **kwargs)
            return self.loss_fn(outputs, x_1)
        elif self.predicts == "score":
            outputs = self.model(t, x_t, **kwargs)
            v = self.path.convert_parameterization(t, x_t, outputs, "score", "v")
            v_theta = self.target_velocity(t, x_0, x_1)
            return self.loss_fn(v, v_theta)
        else:
            raise ValueError("Invalid prediction type. Must be one of ['x_0', 'x_1', 'score']")

    # Basic training and validation steps. Can be overridden in subclasses to define more complex training and validation loops, including FID logging for example.

    def training_step(self, batch, batch_idx):
        use_y = False
        if len(batch) == 3:
            x_0, x_1, y = batch
            use_y = True if self.guidance == "classifier" else False
        else:
            x_0, x_1 = batch

        # Sample a random time step
        t, x_t = self.path.sample(x_0, x_1)

        loss = self._get_loss(x_0, x_1, t, x_t, y=y) if use_y else self._get_loss(x_0, x_1, t, x_t)

        # Log the loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        use_y = False
        if len(batch) == 3:
            x_0, x_1, y = batch
            use_y = True
        else:
            x_0, x_1 = batch

        # Sample a random time step
        t, x_t = self.path.sample(x_0, x_1)

        loss = self._get_loss(x_0, x_1, t, x_t, y=y) if use_y else self._get_loss(x_0, x_1, t, x_t)

        # Log the loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss    

    def configure_optimizers(self):
        return self.optimizer_cfg(params=self.parameters())
    
    def configure_schedulers(self, optimizer):
        if self.scheduler_cfg is not None:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            return [scheduler], []
        else:
            return None