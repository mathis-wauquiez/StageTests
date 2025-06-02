import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import imageio
import matplotlib.pyplot as plt
from datetime import datetime
from hydra.core.hydra_config import HydraConfig

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

    # shared LPIPS model (re‑used across callback instances)
    _shared_lpips = None

    def __init__(self, output_dir: str | None = None, gif_name: str = "evolution.gif"):
        super().__init__()
        run_dir = output_dir or HydraConfig.get().runtime.output_dir
        self.out = Path(run_dir) / "reports"
        self.out.mkdir(parents=True, exist_ok=True)
        self.val_samples_dir = self.out / "val_samples"
        self.val_samples_dir.mkdir(parents=True, exist_ok=True)
        self.gif_name = gif_name

        # Hydra overrides (key=value strings)
        self.overrides: list[str] = sorted(HydraConfig.get().overrides.task)

        # metric modules (LPIPS reused)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        if FullReportCallback._shared_lpips is None:
            FullReportCallback._shared_lpips = lpips.LPIPS(net="vgg").eval()
        self.lpips = FullReportCallback._shared_lpips

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
        self.psnr.to(device)
        self.ssim.to(device)
        self.lpips.to(device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if isinstance(loss, torch.Tensor):
            self._train_batch_losses.append(loss.item())
        elif isinstance(loss, (int, float)):
            self._train_batch_losses.append(float(loss))

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if self._train_batch_losses:
            avg = float(np.mean(self._train_batch_losses))
            self.history['train_loss'].append((epoch, avg))
            np.save(self.out / 'loss_train.npy', np.array([v for _, v in self.history['train_loss']], dtype=float))
            self._train_batch_losses.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict):
            buf = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in outputs.items()}
            self._val_outputs.append(buf)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        outs = self._val_outputs

        # --- aggregate val loss
        val_losses = [o['val_loss'].item() for o in outs if 'val_loss' in o]
        if val_losses:
            avg_val = float(np.mean(val_losses))
            self.history['val_loss'].append((epoch, avg_val))
            np.save(self.out / 'loss_val.npy', np.array([v for _, v in self.history['val_loss']], dtype=float))

        # --- metrics + sample images per tag
        device = next(self.lpips.parameters()).device
        for tag in ('test', 'ema_test'):
            if not all(f"{tag}_pred" in o for o in outs):
                continue
            preds = torch.cat([o[f"{tag}_pred"] for o in outs], dim=0).to(device)
            gts   = torch.cat([o[f"{tag}_gt"]   for o in outs], dim=0).to(device)
            with torch.no_grad():
                ps = float(self.psnr(preds, gts).item())
                ss = float(self.ssim(preds, gts).item())
                lp = float(self.lpips(preds, gts).mean().item())
            self.history['val_metrics'].append({'epoch': epoch, 'tag': tag, 'psnr': ps, 'ssim': ss, 'lpips': lp})
            np.save(self.out / f"metrics_{tag}_epoch{epoch}.npy", np.array([ps, ss, lp], dtype=float))

            # ---- save combined sample image (GT | Pred) to disk only
            gt_img   = gts[0].cpu().permute(1, 2, 0).numpy()
            pred_img = preds[0].cpu().permute(1, 2, 0).numpy()
            comb = np.concatenate([gt_img, pred_img], axis=1)
            tag_dir = self.val_samples_dir / tag
            tag_dir.mkdir(exist_ok=True)
            img_path = tag_dir / f'epoch{epoch}.png'
            plt.imsave(img_path, comb, dpi=200)
            # overwrite latest shortcut
            latest_path = self.out / f'val_sample_{tag}_last.png'
            plt.imsave(latest_path, comb, dpi=200)

            # keep only path in memory
            self.history['images'].append({'epoch': epoch, 'tag': tag, 'path': str(img_path)})

        # free buffer
        self._val_outputs.clear()

    def on_train_end(self, trainer, pl_module):
        pdf_path = self.out / 'training_report.pdf'
        run_name = Path(HydraConfig.get().runtime.output_dir).name
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        last_epoch = max((e for e, _ in self.history['train_loss']), default=None)
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_path) as pdf:
            self._add_title(pdf, run_name, ts)
            self._plot_overrides(pdf)
            self._plot_loss(pdf)
            self._plot_metrics(pdf)
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

    def _plot_loss(self, pdf):
        fig, ax = plt.subplots()
        for label, series in (("train", self.history["train_loss"]), ("val", self.history["val_loss"])):
            if series:
                epochs, vals = zip(*series)
                ax.plot(epochs, vals, marker="o", label=f"{label} loss")
        ax.set(title="Loss by Epoch", xlabel="Epoch", ylabel="Loss")
        ax.legend()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    def _plot_metrics(self, pdf):
        metric_names = ["psnr", "ssim", "lpips"]
        tags = sorted({m["tag"] for m in self.history["val_metrics"]})
        for m in metric_names:
            fig, ax = plt.subplots()
            for tag in tags:
                recs = [r for r in self.history["val_metrics"] if r["tag"] == tag]
                if recs:
                    epochs = [r["epoch"] for r in recs]
                    vals = [r[m] for r in recs]
                    ax.plot(epochs, vals, marker="o", label=f"{tag} {m.upper()}")
                    # annotate best/worst
                    idx = int(np.argmax(vals) if m != "lpips" else np.argmin(vals))
                    ax.annotate(f"{vals[idx]:.3f}", (epochs[idx], vals[idx]), textcoords="offset points", xytext=(0, 5))
            ax.set(title=f"{m.upper()} by Epoch", xlabel="Epoch", ylabel=m.upper())
            ax.legend()
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
