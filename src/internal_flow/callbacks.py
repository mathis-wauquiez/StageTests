import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import lpips
import imageio
import matplotlib.pyplot as plt
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
import torchvision as tv

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def primitiveize(obj):
    if isinstance(obj, dict):
        return {k: primitiveize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [primitiveize(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj

# -----------------------------------------------------------------------------
# Callback 1 – save final config + metrics
# -----------------------------------------------------------------------------

class SaveConfigAndMetrics(Callback):
    """Write Hydra config & final Lightning metrics to <run_dir>/<filename>."""

    def __init__(self, filename: str = "run.yaml"):
        super().__init__()
        self.filename = filename

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        cfg  = primitiveize(trainer.lightning_module.hparams)
        mets = primitiveize(trainer.callback_metrics)
        out  = OmegaConf.create({"config": cfg, "metrics": mets})
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        OmegaConf.save(out, run_dir / self.filename)

# -----------------------------------------------------------------------------
# Callback 2 – full training report
# -----------------------------------------------------------------------------

class FullReportCallback(Callback):
    """
    * Save train/val losses (.npy accumulated).
    * Save per‑epoch validation sample images on disk only (store paths in memory).
    * Build a PDF report + GIF at the end of training.
    * Show Hydra overrides in the report.
    """


    def _make_subdir(self, path: Path):
        """Create a subdirectory at the given path, if it does not exist."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __init__(self, output_dir: str | None = None, gif_name: str = "evolution.gif"):
        super().__init__()
        run_dir = output_dir or HydraConfig.get().runtime.output_dir
        self.out = Path(run_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self._make_subdir(self.out / "metrics")
        self.losses_dir = self._make_subdir(self.out / "losses")
        self.val_samples_dir = self._make_subdir(self.out / "val_samples")
        self.last_val_samples_dir = self._make_subdir(self.out / "samples")

        self.gif_name = gif_name

        # Hydra overrides (key=value strings)
        self.overrides: list[str] = sorted(HydraConfig.get().overrides.task)


        # history (store only paths for images)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'images': []   # list of {epoch, tag, path}
        }
        self._train_batch_losses: list[float] = []
        self._val_outputs: list[dict] = []

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer, pl_module):
        device = pl_module.device

        hydra_cfg = HydraConfig.get()        
        # save Hydra config file
        config_path = self.out / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(self.cfg, f)

        print(OmegaConf.to_yaml(self.cfg))

        hydra_config_path = self.out / "hydra_config.yaml"
        with open(hydra_config_path, "w") as f:
            OmegaConf.save(hydra_cfg, f)


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict):
            buf = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in outputs.items()}
            self._val_outputs.append(buf)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        outs = self._val_outputs

        # --- metrics + sample images per tag
        device = pl_module.device
        gts   = torch.cat([o["test_gt"]   for o in outs], dim=0)

        for tag in ('test', 'ema_test'):
            if not all(f"{tag}_pred" in o for o in outs):
                continue
            preds = torch.cat([o[f"{tag}_pred"] for o in outs], dim=0)            
            make_grid = lambda x: tv.utils.make_grid(
                x[:9].cpu(),
                nrow=3, normalize=False, padding=10, value_range=(0, 1)
            )

            # ---- save combined sample image (GT | Pred) to disk only
            comb = torch.cat([gts[0:1], preds], dim=0)
            grid = make_grid(comb)
            tag_dir = self.val_samples_dir / tag
            tag_dir.mkdir(exist_ok=True)
            img_path = tag_dir / f'epoch{epoch}.png'
            
            tv.utils.save_image(grid, img_path)
            # overwrite latest shortcut
            latest_path = self.out / f'val_sample_{tag}_last.png'
            tv.utils.save_image(grid, latest_path)

            # keep only path in memory
            self.history['images'].append({'epoch': epoch, 'tag': tag, 'path': str(img_path)})

        # free buffer
        self._val_outputs.clear()

    def on_train_end(self, trainer, pl_module):
        pdf_path = self.out / 'training_report.pdf'
        run_name = self.out.name
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        last_epoch = max((e for e, _ in self.history['train_loss']), default=None)
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_path) as pdf:
            self._add_title(pdf, run_name, ts)
            self._plot_overrides(pdf)
            if last_epoch is not None:
                self._plot_last_images(pdf, last_epoch)
        self._make_gif()
        print(f"Report saved to {pdf_path}\nGIF saved to {self.out/self.gif_name}")

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _add_title(self, pdf, run, ts):
        fig = plt.figure(figsize=(8, 6))
        fig.text(0.5, 0.6, f"Run: {run}", ha='center', va='center', fontsize=24)
        fig.text(0.5, 0.4, f"Date: {ts}", ha='center', va='center', fontsize=12)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    def _plot_overrides(self, pdf):
        fig, ax = plt.subplots(figsize=(8, 0.6 + 0.35 * len(self.overrides)))
        ax.axis("off")
        rows = [kv.split("=", 1) if "=" in kv else [kv, ""] for kv in self.overrides]
        tbl = ax.table(cellText=rows, colLabels=["Override", "Value"], loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    def _plot_last_images(self, pdf, last_epoch):
        tags = sorted({r["tag"] for r in self.history["images"] if r["epoch"] == last_epoch})
        fig, axes = plt.subplots(len(tags), 1, figsize=(10, 4 * len(tags)))
        if len(tags) == 1:
            axes = [axes]
        for ax, tag in zip(axes, tags):
            rec = next(r for r in self.history["images"] if r["epoch"] == last_epoch and r["tag"] == tag)
            img = imageio.imread(rec["path"])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{tag} – Epoch {last_epoch}")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    def _make_gif(self):
        frames = []
        for rec in sorted(self.history["images"], key=lambda r: (r["tag"], r["epoch"])):
            img = imageio.imread(rec["path"])
            frames.append(img)
        if frames:
            out_path = self.out / self.gif_name
            imageio.mimsave(out_path, frames, duration=0.6)
