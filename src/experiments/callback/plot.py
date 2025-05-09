import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import os

class GenerationPlotCallback(pl.Callback):
    def __init__(self, save_dir: str = None, num_samples: int = 16):
        """
        Args:
            save_dir (str or None): If provided, saves the plot to this directory.
            num_samples (int): Number of samples to generate and plot.
        """
        super().__init__()
        self.save_dir = save_dir
        self.num_samples = num_samples
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            x0 = torch.randn(self.num_samples, 28*28).to(pl_module.device)
            x0 = x0.view(self.num_samples, 1, 28, 28)
            samples = pl_module.sample(x0)  # Shape: (N, C, H, W)
            samples = (samples * 0.3081) + 0.1307
            samples = torch.clamp(samples, 0, 1)
            samples = samples.cpu()

        n = int(self.num_samples ** 0.5)
        fig, axes = plt.subplots(n, n, figsize=(n, n))

        for i, ax in enumerate(axes.flat):
            img = samples[i]
            img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
            ax.axis("off")

        plt.tight_layout()

        if self.save_dir:
            path = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch:03d}.png")
            plt.savefig(path)
            plt.close(fig)
        else:
            plt.show()
