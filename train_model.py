
import torch
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import sys
import os
from pathlib import Path

from PIL import Image
import numpy as np

from src.internal_flow.flow import InpaintingFlow
from src.internal_flow.utils import next_version_dir, current_version_dir
# register as ${next_version:outputs}
OmegaConf.register_new_resolver("next_version", next_version_dir)


@hydra.main(config_path="confs", config_name="config", version_base=None)
def train_model(cfg: DictConfig):
    """
    Train the model using the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing training parameters.
    """

    train_loader = instantiate(cfg.data.train_loader)
    test_loader = instantiate(cfg.data.test_loader)

    effective_batch_size = cfg.data.train_loader.batch_size * cfg.trainer.accumulate_grad_batches
    lr = 1e-4 * effective_batch_size / 256
    cfg.flow_model.optimizer_cfg.lr = lr

    flow_model = instantiate(cfg.flow_model, to_natural_fn=train_loader.dataset.to_natural)
    trainer = instantiate(cfg.trainer)

    

    trainer.fit(flow_model, train_loader, test_loader)
    trainer.test(flow_model, test_loader)

    output_dir = current_version_dir()
    output_dir = Path(output_dir)
    torch.save(flow_model.state_dict(), output_dir / "model.pth")
    print(f"Model saved to {output_dir / 'model.pth'}")

    # Save the EMA model if it exists
    if hasattr(flow_model, "ema"):
        ema_model = flow_model.ema
        ema_model_path = output_dir / "ema_model.pth"
        torch.save(ema_model.state_dict(), ema_model_path)
        print(f"EMA model saved to {output_dir / 'ema_model.pth'}")
    
    

if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_model()