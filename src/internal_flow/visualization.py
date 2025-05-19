import matplotlib.pyplot as plt
import hydra

from .utils import current_version_dir
import itertools

from pathlib import Path

def visualize(x_0, x_1, x_pred, ema=False):
    """
    Visualize the original and inpainted images.
    """
    x_0 = x_0[0]; x_1 = x_1[0]; x_pred = x_pred[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(x_1.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Ground Truth Image")
    axes[0].axis("off")

    axes[1].imshow(x_pred.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Inpainted Image")
    axes[1].axis("off")

    axes[2].imshow(x_0.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title("What the model sees")
    axes[2].axis("off")
    plt.tight_layout()
    
    # Save the figure
    dir = Path(current_version_dir())

    filename_ext = "_ema" if ema else ""

    for i in itertools.count():
        if not (dir/ f"comparison_{i}{filename_ext}.png").exists():
            break
    dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{dir}/comparison_{i}{filename_ext}.png")
    plt.close(fig)
