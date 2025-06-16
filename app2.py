"""
Gradio demo for in-painting flow models.
95% AI-generated app
--------------------------------------

* Auto-discovers runs under:

    outputs/<sanitized-image>/<category>/YYYY-MM-DD/HH-MM-SS/

* Nested pickers:  Image → Category → Date → Time
  – each picker now defaults to the **latest** (lexicographically last) entry.
* Editable `solver_cfg` YAML → forwarded as **kwargs to `model.sample`.
* Reload-sample & Generate buttons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

import gradio as gr

app_text = open("app.md").read()  # load app description from file

# ──────────────────────────────────────────────────────────────────────────
# Auto-discovery utilities
# ──────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path("outputs")          # fixed base directory


def is_valid_run(path: Path) -> bool:
    return (path / "reports" / "config.yaml").is_file() and (
        (path / "model.pth").is_file() or (path / "ema_model.pth").is_file()
    )


def discover_runs(base: Path = OUTPUTS_DIR) -> Dict[str, Dict[str, Dict[str, Dict[str, Path]]]]:
    """
    Build 4-level dict:  image → category → date → time → Path
    Leaves that fail `is_valid_run` are skipped.
    """
    tree: Dict[str, Dict[str, Dict[str, Dict[str, Path]]]] = {}
    for leaf in base.glob("*/*/*/*"):                 # depth-4
        if not leaf.is_dir() or not is_valid_run(leaf):
            continue
        img, cat, date, time = leaf.parts[-4:]
        tree.setdefault(img, {}).setdefault(cat, {}).setdefault(date, {})[time] = leaf.resolve()
    return tree


# ──────────────────────────────────────────────────────────────────────────
#  Model / data helpers
# ──────────────────────────────────────────────────────────────────────────
def load_cfg(run_dir: Path) -> DictConfig:
    return OmegaConf.create(
        yaml.safe_load((run_dir / "reports" / "config.yaml").read_text())
    )


def tensor_to_image(t: torch.Tensor) -> "np.ndarray":
    return (t.detach().float().cpu() * 255).byte().permute(1, 2, 0).numpy()


def build_model(cfg: DictConfig, device: torch.device):
    m = instantiate(cfg.flow_model, _convert_="object")
    m.to(device).eval()
    return m


def load_weights(model, run_dir: Path, use_ema: bool, map_location="cpu"):
    path = run_dir / ("ema_model.pth" if use_ema else "model.pth")
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:        # lightning ckpt
        state = {k.replace("flow_model.", "", 1): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=False)


def sample_test_batch(cfg: DictConfig, device: torch.device):
    loader = instantiate(cfg.data.test_loader)
    x0, x1, mask = next(iter(loader))
    return x0.to(device), x1.to(device), mask.to(device), loader.dataset


def solver_cfg_to_yaml(cfg: DictConfig) -> str:
    if "solver_cfg" in cfg.flow_model:
        return yaml.safe_dump(OmegaConf.to_container(cfg.flow_model.solver_cfg, resolve=True), sort_keys=False)
    return "# type your solver_cfg here"


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def newest(seq):           # helper – lexicographic max == latest
    return sorted(seq)[-1]


def main():
    parser = argparse.ArgumentParser("In-painting Flow Gradio demo")
    parser.add_argument("run_dirs", nargs="*", type=Path,
                        help="Explicit run folders (skip auto-discovery)")
    parser.add_argument("--ema",   action="store_true", help="Load ema_model.pth")
    parser.add_argument("--share", action="store_true", help="Create public share URL")
    args = parser.parse_args()

    explicit = [p.expanduser().resolve() for p in args.run_dirs]
    use_nested = not explicit

    if use_nested:
        tree = discover_runs(OUTPUTS_DIR)
        if not tree:
            raise RuntimeError(f"No valid runs under {OUTPUTS_DIR.resolve()}")
        # pick LATEST everywhere
        first_img  = newest(tree)
        first_cat  = newest(tree[first_img])
        first_date = newest(tree[first_img][first_cat])
        first_time = newest(tree[first_img][first_cat][first_date])
        first_path = tree[first_img][first_cat][first_date][first_time]
    else:
        for d in explicit:
            if not is_valid_run(d):
                raise FileNotFoundError(d)
        first_path = explicit[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache: Dict[str, Dict[str, Any]] = {}

    def load_run(path: Path):
        key = str(path)
        if key not in cache:
            cfg = load_cfg(path)
            model = build_model(cfg, device)
            load_weights(model, path, use_ema=args.ema, map_location=device)
            x0, x1, mask, ds = sample_test_batch(cfg, device)
            cache[key] = dict(cfg=cfg, model=model, x0=x0, x1=x1, mask=mask, ds=ds)
        return cache[key]

    first = load_run(first_path)

    # ────────────────────────────  Gradio UI  ─────────────────────────────
    with gr.Blocks() as demo:
        gr.Markdown(app_text)

        with gr.Row():
            # pickers
            if use_nested:
                img_dd  = gr.Dropdown(sorted(tree),               value=first_img,  label="Image")
                cat_dd  = gr.Dropdown(sorted(tree[first_img]),    value=first_cat,  label="Category")
                date_dd = gr.Dropdown(sorted(tree[first_img][first_cat]),
                                      value=first_date, label="Date")
                time_dd = gr.Dropdown(sorted(tree[first_img][first_cat][first_date]),
                                      value=first_time, label="Time")
            else:
                run_dd = gr.Dropdown([str(p) for p in explicit], value=str(first_path), label="Run folder")

            cfg_slider = gr.Slider(0, 10, value=4, step=0.1, label="CFG scale")
            reload_btn = gr.Button("Reload sample", variant="secondary")

        with gr.Row():
            x0_img = gr.Image(tensor_to_image(first["ds"].to_natural(first["x0"]).squeeze(0)), label="x₀ (masked)")
            x1_img = gr.Image(tensor_to_image(first["ds"].to_natural(first["x1"]).squeeze(0)), label="x₁ (ground-truth)")

        solver_box = gr.Code(solver_cfg_to_yaml(first["cfg"]), language="yaml", label="solver_cfg (YAML)")
        gen_btn    = gr.Button("Generate")
        out_img    = gr.Image(label="Generated")

        # ─────────── helpers ───────────
        def path_nested(img, cat, date, time):
            return tree[img][cat][date][time]

        def safe_path(img, cat, date, time):
            return tree.get(img, {}).get(cat, {}).get(date, {}).get(time)

        # ─────────── cascade updates (always pick newest) ───────────
        if use_nested:
            def upd_cat(img):
                cats = sorted(tree[img])
                cat  = newest(cats)
                dates = sorted(tree[img][cat])
                date  = newest(dates)
                times = sorted(tree[img][cat][date])
                return (
                    gr.update(choices=cats,  value=cat),
                    gr.update(choices=dates, value=date),
                    gr.update(choices=times, value=newest(times)),
                )

            def upd_date(img, cat):
                dates = sorted(tree[img][cat])
                date  = newest(dates)
                times = sorted(tree[img][cat][date])
                return (
                    gr.update(choices=dates, value=date),
                    gr.update(choices=times, value=newest(times)),
                )

            def upd_time(img, cat, date):
                times = sorted(tree[img][cat][date])
                return gr.update(choices=times, value=newest(times))

            img_dd.change(upd_cat,  img_dd,               [cat_dd, date_dd, time_dd])
            cat_dd.change(upd_date, [img_dd, cat_dd],     [date_dd, time_dd])
            date_dd.change(upd_time,[img_dd, cat_dd, date_dd], time_dd)

            def switch(img, cat, date, time):
                data = load_run(path_nested(img, cat, date, time))
                return (
                    tensor_to_image(data["ds"].to_natural(data["x0"]).squeeze(0)),
                    tensor_to_image(data["ds"].to_natural(data["x1"]).squeeze(0)),
                    solver_cfg_to_yaml(data["cfg"]),
                    None,
                )

            time_dd.change(switch, [img_dd, cat_dd, date_dd, time_dd],
                           [x0_img, x1_img, solver_box, out_img])
        else:
            def switch_flat(run_lbl):
                data = load_run(Path(run_lbl))
                return (
                    tensor_to_image(data["ds"].to_natural(data["x0"]).squeeze(0)),
                    tensor_to_image(data["ds"].to_natural(data["x1"]).squeeze(0)),
                    solver_cfg_to_yaml(data["cfg"]),
                    None,
                )

            run_dd.change(switch_flat, run_dd, [x0_img, x1_img, solver_box, out_img])

        # reload
        def current_path(*sel):
            return path_nested(*sel[:4]) if use_nested else Path(sel[0])

        reload_in = [img_dd, cat_dd, date_dd, time_dd] if use_nested else [run_dd]

        def reload_sample(*sel):
            data = load_run(current_path(*sel))
            x0, x1, mask, _ = sample_test_batch(data["cfg"], device)
            data.update(x0=x0, x1=x1, mask=mask)
            return (
                tensor_to_image(data["ds"].to_natural(x0).squeeze(0)),
                tensor_to_image(data["ds"].to_natural(x1).squeeze(0)),
                None,
            )

        reload_btn.click(reload_sample, reload_in, [x0_img, x1_img, out_img])

        # generate
        gen_inputs = ([img_dd, cat_dd, date_dd, time_dd] if use_nested else [run_dd]) + [cfg_slider, solver_box]

        def generate(*vals):
            if use_nested:
                img, cat, date, time, scale, yaml_txt = vals
                run_path = safe_path(img, cat, date, time)
                if run_path is None:
                    raise gr.Error("Combination doesn’t exist. Pick another time.")
            else:
                run_lbl, scale, yaml_txt = vals
                run_path = Path(run_lbl)

            data = load_run(run_path)
            model, x0, mask, ds = data["model"], data["x0"], data["mask"], data["ds"]

            if hasattr(model, "cfg") and hasattr(model.cfg, "guidance_scale"):
                model.cfg.guidance_scale = scale

            try:
                solver_kwargs = yaml.safe_load(yaml_txt) or {}
                if not isinstance(solver_kwargs, dict):
                    raise ValueError
            except Exception as e:
                raise gr.Error(f"YAML error: {e}")

            with torch.no_grad():
                sample = model.sample(x0, y=mask, **solver_kwargs)
                sample = ds.to_natural(sample)

            return tensor_to_image(sample.squeeze(0))

        gen_btn.click(generate, gen_inputs, out_img)

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
