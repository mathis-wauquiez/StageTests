from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.parsing import AttributeDict


def primitiveize(obj):
    if isinstance(obj, dict):
        return {k: primitiveize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [primitiveize(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # 0-D torch / numpy
        try:
            return obj.item()
        except Exception:
            pass
    return obj  # already a primitive

class SaveConfigAndMetrics(Callback):
    """Write config + final metrics to <output_dir>/<filename>."""
    def __init__(self, filename="run.yaml"):
        self.filename = filename

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        cfg  = trainer.lightning_module.hparams
        mets = trainer.callback_metrics

        clean_mets = primitiveize(mets)          # now just a plain dict
        out        = OmegaConf.create({"config": cfg, "metrics": clean_mets})
        run_dir = Path(trainer.log_dir).resolve()          # = outputs/v_n
        OmegaConf.save(out, run_dir / self.filename)
