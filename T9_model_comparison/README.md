# T9 Model Comparison: SAM3 vs YOLO

Mini project for comparing the newly fine-tuned SAM3/SAM31 model with the
existing YOLO segmentation model on the CGH pathology validation split.

The SAM3 checkpoint is stored on Hugging Face:

```text
https://huggingface.co/nttssv/sam31-cgh-sam3
```

Because that repo is private, run `hf auth login` first with a token that can
read the repo.

## What This Compares

- SAM3: validation AP/AP50/AP75 and training curves parsed from the SAM3
  `training_log.txt` uploaded to Hugging Face.
- YOLO: fresh Ultralytics validation on
  `training_data/dataset/yolo_seg_dataset/data.yaml` using
  `training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt`.

This gives a clean metric table and plots. It does not yet run full SAM3
inference overlays, because that requires the SAM3 source checkout, model config,
and checkpoint loading path.

## Setup

From the repository root:

```bash
python -m pip install -r T9_model_comparison/requirements.txt
hf auth login
```

If the dataset archive has not been unpacked yet:

```bash
tar -xzf training_data_sam31_cgh_20260604.tar.gz
```

## Run

```bash
python T9_model_comparison/compare_models.py \
  --project-root . \
  --hf-repo nttssv/sam31-cgh-sam3 \
  --run-yolo
```

Outputs are written to:

```text
T9_model_comparison/outputs/
```

Open the report:

```bash
open T9_model_comparison/outputs/comparison_report.html
```

On Linux:

```bash
xdg-open T9_model_comparison/outputs/comparison_report.html
```

## Faster SAM3 Log-Only Mode

By default the script downloads only small files from Hugging Face:

- `*.txt`
- `*.yaml`
- `*.yml`
- `*.json`

It does not download the multi-GB checkpoint unless requested.

To download checkpoint artifacts too:

```bash
python T9_model_comparison/compare_models.py \
  --project-root . \
  --hf-repo nttssv/sam31-cgh-sam3 \
  --download-checkpoint \
  --run-yolo
```

## Outputs

- `comparison_summary.csv`: side-by-side metrics.
- `sam3_training_metrics.csv`: parsed SAM3 train metrics over epochs.
- `sam3_validation_metrics.csv`: parsed SAM3 validation metrics over epochs.
- `comparison_report.html`: browser-friendly summary.
- `plots/*.png`: loss, mIoU, AP, and comparison plots.

