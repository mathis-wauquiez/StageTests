import os
from pathlib import Path
from threading import Event

import gradio as gr
from PIL import Image

import itertools
import pathlib

def get_mask(image_path):

    if isinstance(image_path, (Path, os.PathLike)):
        image_path = str(image_path)

    done, result = Event(), {}

    def _on_done(editor_value):
        """Callback fired when the **Done** button is pressed."""
        layers = editor_value.get("layers", [])
        if not layers:
            raise ValueError("No mask layer was drawn — please paint a mask before clicking Done.")
        mask_ref = layers[0]  # filepath because we used type="filepath"
        result["mask"] = Image.open(mask_ref)
        done.set()
        demo.close()

    with gr.Blocks() as demo:
        editor = gr.ImageMask(value=image_path, type="filepath")
        gr.Button("Done").click(_on_done, inputs=editor)

    demo.launch(inbrowser=True)
    done.wait()
    return result["mask"]

def next_version_dir(base="outputs"):
    base = pathlib.Path(base)
    for i in itertools.count():
        if not (base / f"v_{i}").exists():
            return f"v_{i}"

def current_version_dir(base="outputs"):
    base = pathlib.Path(base)
    for i in itertools.count():
        if not (base / f"v_{i+1}").exists():
            return base/f"v_{i}"