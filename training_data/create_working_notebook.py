#!/usr/bin/env python3
"""Create working.ipynb for SAM 3.1 cluster fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "working.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").splitlines()],
    }


cells = [
    md(
        """
# CGH Pathology SAM 3.1 Fine-Tuning

Run this notebook from inside the `training_data` folder after uploading it to
your GPU cluster.

The package is built from the QuPath/Yolo training export:

- 20 pathology tiles
- cell boundary instance masks
- nucleus instance masks
- connective tissue candidate masks
- compact cell and clear cell labels preserved from `cell_instances.csv`

SAM 3.1 is a gated Hugging Face checkpoint. The Hugging Face model card says
the repo hosts checkpoints only and points users to `facebookresearch/sam3` for
code/training. This notebook therefore uses the official SAM3 repo, not
`transformers.AutoModel`.
"""
    ),
    code(
        """
from pathlib import Path
import json
import os
import subprocess
import sys

PACKAGE_ROOT = Path.cwd().resolve()
assert (PACKAGE_ROOT / "dataset").exists(), f"Start Jupyter from training_data, got {PACKAGE_ROOT}"

SAM3_REPO = Path(os.environ.get("SAM3_REPO", PACKAGE_ROOT.parent / "sam3")).resolve()
HF_MODEL_ID = "facebook/sam3"
DATASET_ROOT = PACKAGE_ROOT / "dataset"
COCO_ROOT = DATASET_ROOT / "coco_sam3" / "cgh_pathology_sam31"
OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "sam31_runs"

print("PACKAGE_ROOT:", PACKAGE_ROOT)
print("SAM3_REPO:", SAM3_REPO)
print("COCO_ROOT:", COCO_ROOT)
"""
    ),
    md(
        """
## 1. Install Environment

Run this in a terminal on the cluster before starting the notebook, or set
`INSTALL_SAM3 = True` in the next cell if your Jupyter kernel can install
packages.

```bash
conda create -n sam3 python=3.11
conda activate sam3

# Blackwell / sm_120 GPUs:
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# V100 / older CUDA 11.8 nodes:
# pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements_sam31.txt

git clone https://github.com/facebookresearch/sam3.git ../sam3
cd ../sam3
pip install --user -e ".[train]"

hf auth login
wandb login
```
"""
    ),
    code(
        """
INSTALL_SAM3 = False

if INSTALL_SAM3:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(PACKAGE_ROOT / "requirements_sam31.txt")], check=True)
    if not SAM3_REPO.exists():
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/sam3.git", str(SAM3_REPO)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-e", ".[train]"], cwd=SAM3_REPO, check=True)
else:
    print("Skipping install. Set INSTALL_SAM3=True only if package installation is allowed in this notebook kernel.")
"""
    ),
    md(
        """
## 2. Verify Hugging Face Access

You must request/accept access to `facebook/sam3` and authenticate with
`hf auth login` or set `HF_TOKEN` before training. If you use a fine-grained
token, enable access to public gated repositories.
"""
    ),
    code(
        """
try:
    from huggingface_hub import hf_hub_download, model_info
    info = model_info(HF_MODEL_ID)
    hf_hub_download(repo_id=HF_MODEL_ID, filename="config.json")
    print("HF model accessible:", info.modelId)
except Exception as exc:
    print("Could not verify HF access.")
    print("Make sure you accepted the facebook/sam3 gated model terms and ran: hf auth login")
    print("For fine-grained tokens, enable access to public gated repositories.")
    print(type(exc).__name__, exc)
"""
    ),
    md(
        """
## 3. Rebuild And Audit Dataset

This regenerates COCO JSON from the copied masks. It is safe to run repeatedly.
By default, uncertain cell boundaries stay in `sam31_manifest.csv` but are not
included in COCO training.
"""
    ),
    code(
        """
subprocess.run([sys.executable, str(PACKAGE_ROOT / "prepare_sam31_dataset.py")], check=True)

summary_path = DATASET_ROOT / "sam31_dataset_summary.json"
summary = json.loads(summary_path.read_text())
summary
"""
    ),
    code(
        """
import pandas as pd

manifest = pd.read_csv(DATASET_ROOT / "sam31_manifest.csv")
display(manifest.groupby(["split", "category", "include_for_training"]).size().rename("n").reset_index())
display(manifest.head())
"""
    ),
    md(
        """
## 4. Visual Check

Overlay one tile with nuclei, clear/compact cell boundaries, and connective
tissue candidate masks. This catches path or mask corruption before training.
"""
    ),
    code(
        """
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

tile_id = "p2_tile_20"
image = np.asarray(Image.open(DATASET_ROOT / "images" / f"{tile_id}.png").convert("RGB"))
cell_mask = np.asarray(Image.open(DATASET_ROOT / "cell_instance_masks" / f"{tile_id}.png"))
nucleus_mask = np.asarray(Image.open(DATASET_ROOT / "auxiliary_masks" / f"{tile_id}_gt_nucleus_instances.png"))
stroma_mask = np.asarray(Image.open(DATASET_ROOT / "auxiliary_masks" / f"{tile_id}_gt_stroma.png")) > 0

overlay = image.astype(float) / 255.0
overlay[nucleus_mask > 0] = overlay[nucleus_mask > 0] * 0.55 + np.array([0.05, 0.25, 1.0]) * 0.45
overlay[cell_mask > 0] = overlay[cell_mask > 0] * 0.55 + np.array([0.0, 0.8, 0.25]) * 0.45
overlay[stroma_mask] = overlay[stroma_mask] * 0.55 + np.array([1.0, 0.1, 0.05]) * 0.45

plt.figure(figsize=(7, 7))
plt.imshow(overlay)
plt.title(tile_id)
plt.axis("off")
plt.show()
"""
    ),
    md(
        """
## 5. Create SAM3 Training Config

The official SAM3 `train.py` uses Hydra config names, so the helper below
copies a patched config into `SAM3_REPO/sam3/train/configs/cgh_pathology/`.
"""
    ),
    code(
        """
assert SAM3_REPO.exists(), f"SAM3 repo not found: {SAM3_REPO}"
config_proc = subprocess.run(
    [sys.executable, str(PACKAGE_ROOT / "write_sam3_config.py"), "--sam3-repo", str(SAM3_REPO)],
    check=True,
    capture_output=True,
    text=True,
)
CONFIG_PATH = Path(config_proc.stdout.strip())
subprocess.run([sys.executable, str(PACKAGE_ROOT / "patch_sam3_cluster.py"), "--sam3-repo", str(SAM3_REPO)], check=True)
CONFIG_NAME = "configs/cgh_pathology/cgh_pathology_sam31_seg.yaml"
print("Wrote:", CONFIG_PATH)
print("Use config name:", CONFIG_NAME)

loss_lines = [line.strip() for line in CONFIG_PATH.read_text().splitlines() if "SemanticSegCriterion" in line]
print("Loss target(s):")
for line in loss_lines:
    print(" ", line)
if any("sam3.losses.loss_fns.SemanticSegCriterion" in line for line in loss_lines):
    raise RuntimeError("Config still points to missing sam3.losses.* target. Rerun write_sam3_config.py after updating this package.")
"""
    ),
    code(
        """
config_text = CONFIG_PATH.read_text()
for token in [
    "roboflow_vl_100_root:",
    "experiment_log_dir:",
    "enable_segmentation:",
    "resolution:",
    "max_ann_per_img:",
    "max_train_queries:",
    "max_val_queries:",
    "max_data_epochs:",
    "skip_saving_ckpts:",
    "amp_dtype:",
]:
    for line in config_text.splitlines():
        if token in line:
            print(line)
            break
"""
    ),
    md(
        """
## 6. Launch Fine-Tuning

Start with one GPU and small batch size. Because this dataset is currently
small, watch validation overlays and avoid over-training. Increase epochs only
after checking qualitative output.
"""
    ),
    code(
        """
RUN_TRAINING = False
USE_CLUSTER = 0
NUM_GPUS = 1
NUM_NODES = 1
PARTITION = None
ACCOUNT = None
QOS = None

cmd = [
    sys.executable,
    str(SAM3_REPO / "sam3" / "train" / "train.py"),
    "-c",
    CONFIG_NAME,
    "--use-cluster",
    str(USE_CLUSTER),
    "--num-gpus",
    str(NUM_GPUS),
    "--num-nodes",
    str(NUM_NODES),
]
if PARTITION:
    cmd += ["--partition", PARTITION]
if ACCOUNT:
    cmd += ["--account", ACCOUNT]
if QOS:
    cmd += ["--qos", QOS]

print(" ".join(cmd))
if RUN_TRAINING:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_PROJECT", "sam31-cgh")
    env.setdefault("WANDB_RUN_NAME", "cgh_pathology_sam31")
    subprocess.run(cmd, cwd=SAM3_REPO, env=env, check=True)
else:
    print("Dry run only. Set RUN_TRAINING=True when ready.")
"""
    ),
    md(
        """
## 7. Monitor Training

SAM3 writes logs/checkpoints under `outputs/sam31_runs`. Use TensorBoard for
loss curves. If you ran `wandb login`, W&B also receives the same scalar logs.
"""
    ),
    code(
        """
print("TensorBoard log dir:", OUTPUT_ROOT / "tensorboard")
print("W&B project:", os.environ.get("WANDB_PROJECT", "sam31-cgh"))
print("If running in notebook, execute:")
print("%load_ext tensorboard")
print(f"%tensorboard --logdir {OUTPUT_ROOT / 'tensorboard'}")
"""
    ),
    md(
        """
## 8. Package For Transfer

Run from the parent of this folder if you need a tarball for cluster upload:

```bash
tar -czf training_data_sam31_cgh.tar.gz training_data
```
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=2))
print(OUT)
