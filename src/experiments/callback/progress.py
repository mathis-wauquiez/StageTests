import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


# callbacks.py
import sys, logging
from tqdm.auto import tqdm
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.callbacks.progress.tqdm_progress import _update_n  # optional helper

logger = logging.getLogger(__name__)

class ProgressLogger(TQDMProgressBar):
    """One continuous tqdm bar driven by global_step + per-epoch metric logs."""

    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    def __init__(self, precision: int = 2, refresh_rate: int = 10, leave: bool = True):
        super().__init__(refresh_rate=refresh_rate, leave=leave)
        self.precision = precision

    def init_train_tqdm(self):
        """Create one bar: global 0 / max_steps."""
        total = (
            self.trainer.max_steps
            if self.trainer.max_steps != -1
            else self.trainer.estimated_stepping_batches
        )
        return tqdm(
            desc="training",
            total=total,
            dynamic_ncols=True,
            leave=self._leave,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    def on_train_epoch_start(self, *args, **kwargs):
        """Override the parent method so the bar is *not* reset each epoch."""
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Advance bar by one every batch (or jump to global_step for safety)."""
        # Option 1: simple +1
        # self.train_progress_bar.update(1)

        # Option 2: set absolute position = global_step
        _update_n(self.train_progress_bar, trainer.global_step)

        # still let the parent add postfixes such as lr, loss, etc.
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, **kw):
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule, **kw):
        logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, **kw):
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, **kw):
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        metrics_str = []

        for k, v in trainer.callback_metrics.items():
            parts = k.split("_")
            if len(parts) != 2:
                continue
            split, name = parts
            metric = metric_format.format(v.item())
            mname = name if split == "train" else f"v_{name}"
            metrics_str.append(f"{mname} {metric}")

        if metrics_str:
            logger.info(line + ": " + "  ".join(metrics_str))

