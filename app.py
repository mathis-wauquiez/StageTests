"""
Skeleton Gradio app for visualising an in-painting flow model.

Usage (after installing the project requirements):
    python inpainting_gradio_app.py --run_dir /path/to/run [--ema] [--share]

Required files in the *run_dir* folder:
    • reports/config.yaml   – Hydra config saved during training
    • checkpoints/last.ckpt – last Lightning checkpoint (used when --ema **not** given)
    • ema_model.pth         – EMA weights (used when --ema *is* given)

UI features:
    – Displays masked input **x₀** and ground-truth **x₁**
    – Slider to tweak Classifier-Free-Guidance (CFG) scale
    – "Generate" button calls `flow_model.sample()` and shows the result

This version keeps **all** your prior tweaks:
    • `print('Running the script')` at startup
    • `tensor_to_image` assumes inputs are already in `[0,1]` (no [-1,1] remap)
    • `dataset.to_natural(...)` conversions for every displayed / generated image
"""
from __future__ import annotations
print('Running the script')

import argparse
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

import gradio as gr

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_cfg(run_dir: Path) -> DictConfig:
    """Load the Hydra config saved in *run_dir*/reports/config.yaml."""
    cfg_path = run_dir / "reports" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Cannot find config at {cfg_path}")
    return OmegaConf.create(yaml.safe_load(cfg_path.read_text()))


def tensor_to_image(x: torch.Tensor) -> "np.ndarray":
    """Convert CHW tensor in **[0,1]** space directly to an H×W×C uint8 image."""
    import numpy as np

    x = x.detach().float().cpu() * 255.0
    x = x.byte()
    return x.permute(1, 2, 0).numpy()


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

def build_model(cfg: DictConfig, device: torch.device):
    """Instantiate the flow model (no trainer)."""
    print("Instantiating flow model …")
    flow_model = instantiate(cfg.flow_model, _convert_="object")
    flow_model.to(device).eval()
    return flow_model


def load_weights(model, run_dir: Path, use_ema: bool, map_location="cpu") -> None:
    """Load EMA weights if *use_ema*, else the last Lightning checkpoint."""
    if use_ema:
        weights_path = run_dir / "ema_model.pth"
    else:
        weights_path = run_dir / "model.pth"

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    print(f"Loading weights from {weights_path.relative_to(run_dir)} …")
    state = torch.load(weights_path, map_location=map_location)

    # Lightning checkpoints wrap parameters under "state_dict"
    if weights_path.suffix == ".ckpt" and "state_dict" in state:
        state = state["state_dict"]
        # Strip optional "flow_model." prefix
        state = {k.replace("flow_model.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Weights loaded ✔   (missing={len(missing)}, unexpected={len(unexpected)})")


# -----------------------------------------------------------------------------
# Data helpers – adapt to your own DataLoader output if necessary
# -----------------------------------------------------------------------------

def get_single_test_sample(cfg: DictConfig, device: torch.device):
    """Return (x₀, x₁, M, dataset) from the first test-loader batch."""
    print("Instantiating test loader … (only the first item will be used)")
    test_loader = instantiate(cfg.data.test_loader)
    x0, x1, mask = next(iter(test_loader))  # adjust tuple order if different
    print("Test sample fetched ✔")
    return x0.to(device), x1.to(device), mask.to(device), test_loader.dataset


# -----------------------------------------------------------------------------
# Gradio callbacks
# -----------------------------------------------------------------------------

def make_gradio_demo(model, x0, x1, mask, dataset):
    """Build the interactive Gradio Blocks demo."""

    def _sample(cfg_scale: float):
        with torch.no_grad():
            if hasattr(model, "cfg") and hasattr(model.cfg, "guidance_scale"):
                model.cfg.guidance_scale = cfg_scale
            sample = model.sample(x0, y=mask)  # tweak kwargs if needed
            sample = dataset.to_natural(sample)  # convert to natural image space
        return tensor_to_image(sample.squeeze(0))

    demo = gr.Blocks()
    with demo:
        gr.Markdown("## In-painting Flow Demo\nAdjust CFG and generate a new reconstruction.")

        with gr.Row():
            gr.Image(
                value=tensor_to_image(dataset.to_natural(x0).squeeze(0)),
                label="x₀ (masked)",
                interactive=False,
            )
            gr.Image(
                value=tensor_to_image(dataset.to_natural(x1).squeeze(0)),
                label="x₁ (ground truth)",
                interactive=False,
            )

        cfg_slider = gr.Slider(0.0, 10.0, value=4.0, step=0.1, label="CFG scale")
        generate_btn = gr.Button("Generate")
        output_img = gr.Image(label="Generated")

        generate_btn.click(_sample, inputs=cfg_slider, outputs=output_img)

    return demo


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gradio demo for an in-painting flow run.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to a training run directory")
    parser.add_argument("--ema", action="store_true", help="Load ema_model.pth instead of checkpoints/last.ckpt")
    parser.add_argument("--share", action="store_true", help="Pass this flag to Gradio to create a public share URL")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_cfg(run_dir)
    model = build_model(cfg, device)
    load_weights(model, run_dir, use_ema=args.ema, map_location=device)

    # Fetch a single sample for visualisation
    x0, x1, mask, dataset = get_single_test_sample(cfg, device)

    print("Building Gradio interface …")
    demo = make_gradio_demo(model, x0, x1, mask, dataset)
    print("Launching demo ✔")
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
