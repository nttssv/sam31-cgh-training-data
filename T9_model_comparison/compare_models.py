#!/usr/bin/env python3
"""Compare the fine-tuned SAM3 run with the reference YOLO model.

The SAM3 side is parsed from the training log produced by SAM3 training.
The YOLO side can be evaluated live with Ultralytics on the same validation
split. This keeps the comparison reproducible without requiring full SAM3
inference setup on every machine.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAM3_REPO_DEFAULT = "nttssv/sam31-cgh-sam3"
YOLO_MODEL_DEFAULT = "training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt"
YOLO_DATA_DEFAULT = "training_data/dataset/yolo_seg_dataset/data.yaml"
OUT_DIR_DEFAULT = "T9_model_comparison/outputs"
HF_DIR_DEFAULT = "T9_model_comparison/artifacts/sam3"


@dataclass
class RunArtifacts:
    output_dir: Path
    plot_dir: Path
    hf_dir: Path


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def ensure_dirs(output_dir: Path, hf_dir: Path) -> RunArtifacts:
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(output_dir=output_dir, plot_dir=plot_dir, hf_dir=hf_dir)


def download_hf_artifacts(repo_id: str, hf_dir: Path, download_checkpoint: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        warn(f"huggingface_hub is not installed, skipping HF download: {exc}")
        return

    patterns = ["*.txt", "*.yaml", "*.yml", "*.json"]
    if download_checkpoint:
        patterns.extend(["*.pt", "*.pth", "*.ckpt", "*.safetensors"])

    print(f"Downloading HF artifacts from {repo_id} into {hf_dir}")
    print("Allowed patterns:", ", ".join(patterns))
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(hf_dir),
            allow_patterns=patterns,
        )
    except Exception as exc:
        warn(
            "Could not download from Hugging Face. If the repo is private, run "
            f"`hf auth login` first. Details: {type(exc).__name__}: {exc}"
        )


def find_sam3_log(hf_dir: Path, explicit_log: Path | None) -> Path | None:
    if explicit_log is not None:
        return explicit_log if explicit_log.exists() else None

    preferred = [
        hf_dir / "training_log.txt",
        hf_dir / "log.txt",
        hf_dir / "logs" / "cgh_pathology_sam31" / "log.txt",
    ]
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(hf_dir.rglob("*.txt"))
    for path in candidates:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if "coco_eval_bbox_AP" in text or "Losses and meters:" in text:
            return path
    return None


def parse_metric_payload(line: str) -> dict[str, Any] | None:
    if "Losses and meters:" in line:
        payload = line.split("Losses and meters:", 1)[1].strip()
    elif "Meters:" in line:
        payload = line.split("Meters:", 1)[1].strip()
    else:
        return None
    try:
        value = ast.literal_eval(payload)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def parse_sam3_log(log_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for line in log_path.read_text(errors="ignore").splitlines():
        payload = parse_metric_payload(line)
        if not payload:
            continue
        epoch = payload.get("Trainer/epoch")
        row = {"epoch": epoch}
        row.update(payload)
        if any(key.startswith("Meters_train/val_") for key in payload):
            val_rows.append(row)
        if any(key.startswith("Losses/train_") for key in payload):
            train_rows.append(row)

    return train_rows, val_rows


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def plot_sam3_curves(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    plot_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    import pandas as pd

    paths: dict[str, Path] = {}

    train_df = pd.DataFrame(train_rows)
    if not train_df.empty:
        train_specs = [
            ("Losses/train_all_loss", "SAM3 Train Loss", "sam3_train_loss.png"),
            (
                "Losses/train_all_miou_semantic_seg",
                "SAM3 Train Semantic mIoU",
                "sam3_train_miou.png",
            ),
            (
                "Losses/train_all_loss_semantic_dice",
                "SAM3 Train Semantic Dice Loss",
                "sam3_train_dice_loss.png",
            ),
        ]
        for column, title, filename in train_specs:
            if column not in train_df.columns:
                continue
            tmp = train_df[["epoch", column]].dropna()
            if tmp.empty:
                continue
            path = plot_dir / filename
            plt.figure(figsize=(7.5, 4.2))
            plt.plot(tmp["epoch"], tmp[column], marker="o", linewidth=1.8)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(column.split("/")[-1])
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            paths[filename] = path

    val_df = pd.DataFrame(val_rows)
    if not val_df.empty:
        val_specs = [
            (
                "Meters_train/val_roboflow100/detection/coco_eval_bbox_AP",
                "bbox AP",
            ),
            (
                "Meters_train/val_roboflow100/detection/coco_eval_bbox_AP_50",
                "bbox AP50",
            ),
            (
                "Meters_train/val_roboflow100/detection/coco_eval_bbox_AP_75",
                "bbox AP75",
            ),
        ]
        usable = [(col, label) for col, label in val_specs if col in val_df.columns]
        if usable:
            path = plot_dir / "sam3_val_ap.png"
            plt.figure(figsize=(7.5, 4.2))
            for column, label in usable:
                tmp = val_df[["epoch", column]].dropna()
                if not tmp.empty:
                    plt.plot(tmp["epoch"], tmp[column], marker="o", linewidth=1.8, label=label)
            plt.title("SAM3 Validation COCO BBox AP")
            plt.xlabel("Epoch")
            plt.ylabel("AP")
            plt.ylim(bottom=0)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            paths["sam3_val_ap.png"] = path

    return paths


def metric_from_obj(obj: Any, dotted: str) -> float | None:
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return to_float(cur)


def run_yolo_validation(
    model_path: Path,
    data_yaml: Path,
    output_dir: Path,
    imgsz: int,
    device: str | None,
) -> dict[str, Any] | None:
    if not model_path.exists():
        warn(f"YOLO model not found: {model_path}")
        return None
    if not data_yaml.exists():
        warn(f"YOLO data YAML not found: {data_yaml}")
        return None

    try:
        from ultralytics import YOLO
    except Exception as exc:
        warn(f"ultralytics is not installed, skipping YOLO validation: {exc}")
        return None

    print("Running YOLO validation")
    print("YOLO model:", model_path)
    print("YOLO data:", data_yaml)
    model = YOLO(str(model_path))

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": "val",
        "imgsz": imgsz,
        "plots": True,
        "save_json": True,
        "project": str(output_dir / "yolo_runs"),
        "name": "val",
        "exist_ok": True,
    }
    if device:
        kwargs["device"] = device

    metrics = model.val(**kwargs)

    row = {
        "model": "YOLO",
        "box_mAP50-95": metric_from_obj(metrics, "box.map"),
        "box_mAP50": metric_from_obj(metrics, "box.map50"),
        "box_mAP75": metric_from_obj(metrics, "box.map75"),
        "mask_mAP50-95": metric_from_obj(metrics, "seg.map"),
        "mask_mAP50": metric_from_obj(metrics, "seg.map50"),
        "mask_mAP75": metric_from_obj(metrics, "seg.map75"),
    }

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        with (output_dir / "yolo_results_dict.json").open("w") as fh:
            json.dump(results_dict, fh, indent=2, sort_keys=True)
        for key, value in results_dict.items():
            row[f"ultralytics/{key}"] = to_float(value)

    return row


def summarize_sam3(val_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not val_rows:
        return None, None

    ap_key = "Meters_train/val_roboflow100/detection/coco_eval_bbox_AP"

    final = val_rows[-1]
    best = max(
        val_rows,
        key=lambda row: to_float(row.get(ap_key)) if to_float(row.get(ap_key)) is not None else -1.0,
    )
    return final, best


def sam3_summary_row(row: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "model": label,
        "epoch": row.get("Trainer/epoch") or row.get("epoch"),
        "box_mAP50-95": to_float(
            row.get("Meters_train/val_roboflow100/detection/coco_eval_bbox_AP")
        ),
        "box_mAP50": to_float(
            row.get("Meters_train/val_roboflow100/detection/coco_eval_bbox_AP_50")
        ),
        "box_mAP75": to_float(
            row.get("Meters_train/val_roboflow100/detection/coco_eval_bbox_AP_75")
        ),
        "box_AR100": to_float(
            row.get("Meters_train/val_roboflow100/detection/coco_eval_bbox_AR_maxDets@100")
        ),
        "mask_mAP50-95": None,
        "mask_mAP50": None,
        "mask_mAP75": None,
    }


def plot_comparison(summary_rows: list[dict[str, Any]], plot_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    metrics = ["box_mAP50-95", "box_mAP50", "box_mAP75", "mask_mAP50-95", "mask_mAP50"]
    metrics = [metric for metric in metrics if metric in df.columns and df[metric].notna().any()]
    if df.empty or not metrics:
        return None

    path = plot_dir / "model_metric_comparison.png"
    plot_df = df.set_index("model")[metrics]
    ax = plot_df.plot(kind="bar", figsize=(9, 4.8), rot=20)
    ax.set_title("Model Metric Comparison")
    ax.set_ylabel("Metric value")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def relative_to_report(path: Path, report_path: Path) -> str:
    return os.path.relpath(path, report_path.parent)


def format_value(value: Any) -> str:
    number = to_float(value)
    if number is not None:
        return f"{number:.4f}"
    if value is None:
        return "-"
    return html.escape(str(value))


def build_html_report(
    report_path: Path,
    summary_rows: list[dict[str, Any]],
    plot_paths: dict[str, Path],
    sam3_log: Path | None,
    hf_repo: str,
    yolo_model: Path,
    yolo_data: Path,
) -> None:
    columns = [
        "model",
        "epoch",
        "box_mAP50-95",
        "box_mAP50",
        "box_mAP75",
        "box_AR100",
        "mask_mAP50-95",
        "mask_mAP50",
        "mask_mAP75",
    ]

    rows_html = []
    for row in summary_rows:
        cells = "".join(f"<td>{format_value(row.get(column))}</td>" for column in columns)
        rows_html.append(f"<tr>{cells}</tr>")

    images_html = []
    for name, path in plot_paths.items():
        if path and path.exists():
            rel = html.escape(relative_to_report(path, report_path))
            images_html.append(
                f"<section><h2>{html.escape(name)}</h2><img src=\"{rel}\" alt=\"{html.escape(name)}\"></section>"
            )

    log_text = str(sam3_log) if sam3_log else "not found"

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAM3 vs YOLO Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #52616b; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; min-width: 900px; margin: 16px 0 32px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; }}
    img {{ max-width: 900px; width: 100%; border: 1px solid #d9e2ec; border-radius: 6px; }}
    section {{ margin: 28px 0; }}
    code {{ background: #f0f4f8; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>SAM3 vs YOLO Comparison</h1>
  <div class="meta">
    SAM3 HF repo: <code>{html.escape(hf_repo)}</code><br>
    SAM3 log: <code>{html.escape(log_text)}</code><br>
    YOLO model: <code>{html.escape(str(yolo_model))}</code><br>
    YOLO data: <code>{html.escape(str(yolo_data))}</code>
  </div>

  <h2>Summary</h2>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(column)}</th>' for column in columns)}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>

  <p>
    SAM3 mask mAP is blank because the current SAM3 training log reports COCO
    bbox AP, while YOLO validation reports both box and mask metrics.
  </p>

  {''.join(images_html)}
</body>
</html>
"""
    report_path.write_text(html_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--hf-repo", default=SAM3_REPO_DEFAULT)
    parser.add_argument("--hf-dir", type=Path, default=Path(HF_DIR_DEFAULT))
    parser.add_argument("--sam3-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(OUT_DIR_DEFAULT))
    parser.add_argument("--yolo-model", type=Path, default=Path(YOLO_MODEL_DEFAULT))
    parser.add_argument("--yolo-data", type=Path, default=Path(YOLO_DATA_DEFAULT))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="YOLO device, for example 0, cpu, or mps.")
    parser.add_argument("--skip-hf-download", action="store_true")
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--run-yolo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if not project_root.exists():
        fail(f"Project root does not exist: {project_root}")

    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    hf_dir = (project_root / args.hf_dir).resolve() if not args.hf_dir.is_absolute() else args.hf_dir
    yolo_model = (project_root / args.yolo_model).resolve() if not args.yolo_model.is_absolute() else args.yolo_model
    yolo_data = (project_root / args.yolo_data).resolve() if not args.yolo_data.is_absolute() else args.yolo_data
    sam3_log_arg = None
    if args.sam3_log:
        sam3_log_arg = (project_root / args.sam3_log).resolve() if not args.sam3_log.is_absolute() else args.sam3_log

    artifacts = ensure_dirs(output_dir, hf_dir)

    if not args.skip_hf_download:
        download_hf_artifacts(args.hf_repo, artifacts.hf_dir, args.download_checkpoint)

    sam3_log = find_sam3_log(artifacts.hf_dir, sam3_log_arg)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    if sam3_log:
        print("Parsing SAM3 log:", sam3_log)
        train_rows, val_rows = parse_sam3_log(sam3_log)
        write_csv(artifacts.output_dir / "sam3_training_metrics.csv", train_rows)
        write_csv(artifacts.output_dir / "sam3_validation_metrics.csv", val_rows)
    else:
        warn("SAM3 training log not found. The report will include YOLO only if --run-yolo is set.")

    plot_paths = plot_sam3_curves(train_rows, val_rows, artifacts.plot_dir)

    summary_rows: list[dict[str, Any]] = []
    sam3_final, sam3_best = summarize_sam3(val_rows)
    if sam3_final:
        summary_rows.append(sam3_summary_row(sam3_final, "SAM3 final"))
    if sam3_best and sam3_best is not sam3_final:
        summary_rows.append(sam3_summary_row(sam3_best, "SAM3 best-val"))

    if args.run_yolo:
        yolo_row = run_yolo_validation(yolo_model, yolo_data, artifacts.output_dir, args.imgsz, args.device)
        if yolo_row:
            summary_rows.append(yolo_row)
    else:
        print("Skipping YOLO validation. Add --run-yolo to evaluate YOLO.")

    if summary_rows:
        write_csv(artifacts.output_dir / "comparison_summary.csv", summary_rows)
        comparison_plot = plot_comparison(summary_rows, artifacts.plot_dir)
        if comparison_plot:
            plot_paths[comparison_plot.name] = comparison_plot
    else:
        warn("No summary rows produced.")

    report_path = artifacts.output_dir / "comparison_report.html"
    build_html_report(
        report_path=report_path,
        summary_rows=summary_rows,
        plot_paths=plot_paths,
        sam3_log=sam3_log,
        hf_repo=args.hf_repo,
        yolo_model=yolo_model,
        yolo_data=yolo_data,
    )

    print("\nDone.")
    print("Report:", report_path)
    summary_csv = artifacts.output_dir / "comparison_summary.csv"
    if summary_csv.exists():
        print("Summary CSV:", summary_csv)


if __name__ == "__main__":
    main()
