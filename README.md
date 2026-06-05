# SAM31 CGH Training Data

This repository contains the CGH pathology training package used for SAM3/SAM31
fine-tuning, plus helper scripts and pinned Python dependencies.

The trained SAM3 checkpoint is not stored in GitHub because it is several GB.
Store and download model weights from Hugging Face instead.

## Contents

- `training_data_sam31_cgh_20260604.tar.gz`: full training package archive.
- `training_data/requirements_sam31.txt`: Python dependencies for dataset prep,
  logging, and SAM3 training support.
- `training_data/working.ipynb`: notebook workflow.
- `training_data/prepare_sam31_dataset.py`: rebuilds/audits COCO data from the
  masks.
- `training_data/write_sam3_config.py`: writes the SAM3 training config.
- `training_data/patch_sam3_cluster.py`: cluster compatibility patches for SAM3.
- `training_data/sam31_dataset_summary.json`: dataset summary from the prepared
  package.

## Clone And Unpack

```bash
git clone https://github.com/nttssv/sam31-cgh-training-data.git
cd sam31-cgh-training-data
tar -xzf training_data_sam31_cgh_20260604.tar.gz
```

## Install Dependencies

Install PyTorch separately for your GPU, then install the helper dependencies.

For Blackwell / CUDA 12.8:

```bash
python -m pip install --user --force-reinstall \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

For V100 / CUDA 11.8:

```bash
python -m pip install --user --force-reinstall \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest:

```bash
python -m pip install --user -r training_data/requirements_sam31.txt
```

## Download SAM3 Checkpoint From Hugging Face

Adjust the repo ID if the checkpoint was uploaded under a different Hugging Face
model name.

```bash
pip install -U huggingface_hub
hf auth login

hf download nttssv/sam31-cgh-sam3 \
  --local-dir hf_sam31_model
```

## Compare With The YOLO Model

The archive includes the YOLO reference model at:

```text
training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt
```

Open `training_data/working.ipynb` after unpacking the archive. The notebook can
load the SAM3 training log/checkpoint and run YOLO validation on the same
validation split.

## Notes

- GitHub stores the code and compact training package archive.
- Hugging Face stores the large SAM3 checkpoint.
- W&B offline logs can be synced separately if online logging is needed.
