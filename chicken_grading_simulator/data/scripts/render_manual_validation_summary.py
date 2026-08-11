#!/usr/bin/env python3
"""Rebuild manual-validation summary assets from latest masks and annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import pycocotools.mask as mask_utils
from matplotlib.colors import LinearSegmentedColormap

from paths import validation_manual_dir

CLASS_NAMES = ("Comb", "Shank")
OVERLAY_COLORS_BGR = {
    "Comb": (0, 0, 255),
    "Shank": (255, 0, 0),
}
ERROR_COLORS_BGR = {
    "tp": (64, 200, 64),
    "fp": (48, 48, 255),
    "fn": (0, 215, 255),
}
CONFUSION_MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "peach_confusion",
    ["#fff8f2", "#f9dfc7", "#f5c08f", "#ee9b5c", "#d97a34"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-dir",
        default=str(validation_manual_dir()),
        help="Manual validation package directory.",
    )
    parser.add_argument(
        "--pred-masks-dir",
        default=str(validation_manual_dir() / "predicted_masks_latest"),
        help="Directory containing predicted binary masks from evaluate_manual_validation_masks.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="clean_validation_summary",
        help="Output folder name under validation-dir.",
    )
    parser.add_argument(
        "--backup-prefix",
        default="clean_validation_summary_backup",
        help="Backup folder prefix for an existing output-dir.",
    )
    parser.add_argument(
        "--qual-grid-size",
        type=int,
        default=16,
        help="How many worst samples to include in the qualitative error grid.",
    )
    return parser.parse_args()


def normalize_class_name(name: str) -> str:
    text = (name or "").strip().lower()
    if text == "comb":
        return "Comb"
    if text == "shank":
        return "Shank"
    raise ValueError(f"Unsupported class name: {name!r}")


def sample_id_from_image(image: Dict[str, object]) -> str:
    extra = image.get("extra")
    if isinstance(extra, dict):
        name = extra.get("name")
        if isinstance(name, str) and name:
            return Path(name).stem

    file_name = str(image.get("file_name", ""))
    if "_png.rf." in file_name:
        return file_name.split("_png.rf.", 1)[0]
    return Path(file_name).stem


def decode_rle_to_mask(segmentation: Dict[str, object]) -> np.ndarray:
    rle = {
        "size": segmentation["size"],
        "counts": segmentation["counts"].encode("utf-8")
        if isinstance(segmentation["counts"], str)
        else segmentation["counts"],
    }
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    group: str,
    label: str,
    confidence: float | None = None,
) -> np.ndarray:
    out = image_bgr.copy()
    overlay_color = OVERLAY_COLORS_BGR[group]
    if mask.any():
        color_layer = np.zeros_like(out, dtype=np.uint8)
        color_layer[mask] = overlay_color
        out = cv2.addWeighted(out, 1.0, color_layer, 0.35, 0.0)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, overlay_color, 2)

    text = label if confidence is None else f"{label} conf={confidence:.2f}"
    cv2.putText(
        out,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        overlay_color,
        2,
        cv2.LINE_AA,
    )
    return out


def make_pred_vs_gt(pred_bgr: np.ndarray, gt_bgr: np.ndarray, sample_id: str) -> np.ndarray:
    h, w = pred_bgr.shape[:2]
    banner_h = 58
    canvas = np.full((h + banner_h, w * 2 + 8, 3), 245, dtype=np.uint8)
    canvas[banner_h:, :w] = pred_bgr
    canvas[banner_h:, w + 8:] = gt_bgr
    canvas[:banner_h, :] = 240

    cv2.putText(canvas, "Pred", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (32, 32, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (w + 24, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (32, 32, 32), 2, cv2.LINE_AA)
    text_size, _ = cv2.getTextSize(sample_id, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cx = max(8, (canvas.shape[1] - text_size[0]) // 2)
    cv2.putText(canvas, sample_id, (cx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(canvas, (w + 4, banner_h), (w + 4, canvas.shape[0]), (255, 255, 255), 8)
    return canvas


def render_table_png(rows: List[Dict[str, object]], output_path: Path) -> None:
    columns = list(rows[0].keys())
    cell_text = [[str(row[col]) for col in columns] for row in rows]

    fig_h = 1.4 + 0.55 * len(rows)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_metrics_heatmap(rows: List[Dict[str, object]], output_path: Path) -> None:
    groups = [row["group"] for row in rows]
    metric_cols = ["accuracy", "precision", "recall", "f1", "iou"]
    data = np.array([[float(row[col]) for col in metric_cols] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_title("Mask Metrics Heatmap")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    output_path: Path,
    title: str,
    normalize: bool,
) -> None:
    display = matrix.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums != 0)

    raw_scale = 1.0
    raw_top = 1.0
    if not normalize:
        raw_scale = 1e7 if float(display.max()) >= 1e7 else 1.0
        raw_top = max(raw_scale, np.ceil(float(display.max()) / raw_scale) * raw_scale)

    fig = plt.figure(figsize=(4.8, 4.2))
    ax = fig.add_axes([0.24, 0.17, 0.60, 0.70])
    if normalize:
        im = ax.imshow(display, cmap=CONFUSION_MATRIX_CMAP, vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(display, cmap=CONFUSION_MATRIX_CMAP, vmin=0.0, vmax=raw_top)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    threshold = float(display.max()) * 0.55 if display.size else 0.0
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            text = f"{display[i, j]:.3f}" if normalize else f"{int(display[i, j])}"
            text_color = "white" if display[i, j] >= threshold and display[i, j] > 0 else "#4a2a17"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=11)

    cax = fig.add_axes([0.87, 0.17, 0.03, 0.70])
    cbar = fig.colorbar(im, cax=cax)
    if normalize:
        cbar.set_ticks(np.linspace(0.0, 1.0, 6))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}".rstrip("0").rstrip(".")))
    else:
        scale = raw_scale
        if scale > 1.0:
            top = int(np.ceil(raw_top / scale))
            cbar.set_ticks(np.arange(0, top + 1, dtype=float) * scale)
            cbar.ax.set_title("1e7", fontsize=14, pad=6)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos, s=scale: f"{int(round(x / s))}"))
    cbar.ax.tick_params(labelsize=12)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_group_confusion_matrices(
    per_class_counts: Dict[str, Dict[str, int]],
    metrics_dir: Path,
) -> None:
    for cls in CLASS_NAMES:
        tp = per_class_counts[cls]["tp"]
        fp = per_class_counts[cls]["fp"]
        fn = per_class_counts[cls]["fn"]
        tn = per_class_counts[cls]["tn"]
        matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
        stem = cls.lower()

        render_confusion_matrix(
            matrix,
            labels=["Background", cls],
            output_path=metrics_dir / f"{stem}_confusion_matrix.png",
            title=f"{cls} Pixel Confusion Matrix",
            normalize=False,
        )
        render_confusion_matrix(
            matrix,
            labels=["Background", cls],
            output_path=metrics_dir / f"{stem}_confusion_matrix_normalized.png",
            title=f"{cls} Pixel Confusion Matrix (Normalized)",
            normalize=True,
        )


def render_qualitative_grid(items: List[Dict[str, object]], output_path: Path) -> None:
    if not items:
        return

    cols = 4
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.6))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, item in zip(axes.flat, items):
        image_rgb = cv2.cvtColor(item["image_bgr"], cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(
            f"{item['sample_id']}\n{item['group']} IoU={item['iou']:.3f} F1={item['f1']:.3f}",
            fontsize=8,
        )

    fig.suptitle("Qualitative Error Map Grid", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mask_report(report_path: Path) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    with report_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["sample_id"]] = {
                **row,
                "gt_pixels": int(row["gt_pixels"]),
                "pred_pixels": int(row["pred_pixels"]),
                "tp_pixels": int(row["tp_pixels"]),
                "fp_pixels": int(row["fp_pixels"]),
                "fn_pixels": int(row["fn_pixels"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "iou": float(row["iou"]),
            }
    return rows


def load_prediction_confidence(prediction_summary_path: Path) -> Dict[str, float]:
    best_conf: Dict[str, float] = defaultdict(float)
    if not prediction_summary_path.exists():
        return {}

    with prediction_summary_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample_id = row["sample_id"].strip()
            target_group = normalize_class_name(row["group"])
            class_name = normalize_class_name(row["class_name"])
            if target_group != class_name:
                continue
            confidence = float(row["confidence"] or 0.0)
            if confidence > best_conf[sample_id]:
                best_conf[sample_id] = confidence
    return dict(best_conf)


def load_ground_truth_masks(validation_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    gt: Dict[str, Dict[str, np.ndarray]] = {cls: {} for cls in CLASS_NAMES}

    for cls in CLASS_NAMES:
        json_path = validation_dir / "manual_labels" / cls.lower() / "_annotations.coco.json"
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_map = {
            int(image["id"]): sample_id_from_image(image)
            for image in data.get("images", [])
        }
        size_map = {
            int(image["id"]): (int(image["height"]), int(image["width"]))
            for image in data.get("images", [])
        }

        per_sample_masks: Dict[str, List[np.ndarray]] = defaultdict(list)
        for ann in data.get("annotations", []):
            sample_id = image_map[int(ann["image_id"])]
            per_sample_masks[sample_id].append(decode_rle_to_mask(ann["segmentation"]))

        for image_id, sample_id in image_map.items():
            height, width = size_map[image_id]
            union_mask = np.zeros((height, width), dtype=bool)
            for mask in per_sample_masks.get(sample_id, []):
                union_mask |= mask
            gt[cls][sample_id] = union_mask

    return gt


def load_pred_mask(pred_mask_path: Path, fallback_shape: Tuple[int, int]) -> np.ndarray:
    if not pred_mask_path.exists():
        return np.zeros(fallback_shape, dtype=bool)
    pred = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE)
    if pred is None:
        return np.zeros(fallback_shape, dtype=bool)
    return pred > 127


def main() -> None:
    args = parse_args()
    validation_dir = Path(args.validation_dir).resolve()
    pred_masks_dir = Path(args.pred_masks_dir).resolve()
    output_dir = validation_dir / args.output_dir

    if output_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = validation_dir / f"{args.backup_prefix}_{stamp}"
        shutil.move(str(output_dir), str(backup_dir))

    gt_overlay_dir = output_dir / "gt_overlay"
    sam_overlay_dir = output_dir / "sam2_overlay"
    pred_vs_gt_dir = output_dir / "pred_vs_gt"
    metrics_dir = output_dir / "metrics"
    for path in (output_dir, gt_overlay_dir, sam_overlay_dir, pred_vs_gt_dir, metrics_dir):
        ensure_clean_dir(path)

    manifest_rows = load_manifest(validation_dir / "manifest.csv")
    report_rows = load_mask_report(validation_dir / "mask_validation_report.csv")
    gt_masks = load_ground_truth_masks(validation_dir)
    confidences = load_prediction_confidence(validation_dir / "prediction_summary.csv")

    per_class_counts: Dict[str, Dict[str, int]] = {
        cls: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cls in CLASS_NAMES
    }
    qual_rows: List[Dict[str, object]] = []

    for row in manifest_rows:
        sample_id = row["sample_id"].strip()
        group = normalize_class_name(row["group"])
        raw_path = validation_dir / row["raw_image"]

        image_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {raw_path}")

        gt_mask = gt_masks[group][sample_id]
        pred_mask_path = pred_masks_dir / f"{sample_id}_{group.lower()}_pred.png"
        pred_mask = load_pred_mask(pred_mask_path, gt_mask.shape)
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        stats = report_rows[sample_id]
        tp = int(stats["tp_pixels"])
        fp = int(stats["fp_pixels"])
        fn = int(stats["fn_pixels"])
        tn = int(gt_mask.size) - tp - fp - fn
        per_class_counts[group]["tp"] += tp
        per_class_counts[group]["fp"] += fp
        per_class_counts[group]["fn"] += fn
        per_class_counts[group]["tn"] += tn

        conf = confidences.get(sample_id, 0.0)
        gt_overlay = overlay_mask(image_bgr, gt_mask, group, f"GT {group}")
        pred_overlay = overlay_mask(image_bgr, pred_mask, group, f"Model {group}", confidence=conf)
        side_by_side = make_pred_vs_gt(pred_overlay, gt_overlay, sample_id)

        cv2.imwrite(str(gt_overlay_dir / f"{sample_id}_gt.png"), gt_overlay)
        cv2.imwrite(str(sam_overlay_dir / f"{sample_id}_model.png"), pred_overlay)
        cv2.imwrite(str(pred_vs_gt_dir / f"{sample_id}_pred_vs_gt.png"), side_by_side)

        error_map = image_bgr.copy()
        tp_mask = np.logical_and(pred_mask, gt_mask)
        fp_mask = np.logical_and(pred_mask, np.logical_not(gt_mask))
        fn_mask = np.logical_and(np.logical_not(pred_mask), gt_mask)
        for key, mask in (("tp", tp_mask), ("fp", fp_mask), ("fn", fn_mask)):
            if not mask.any():
                continue
            layer = np.zeros_like(error_map, dtype=np.uint8)
            layer[mask] = ERROR_COLORS_BGR[key]
            error_map = cv2.addWeighted(error_map, 1.0, layer, 0.42, 0.0)

        qual_rows.append(
            {
                "sample_id": sample_id,
                "group": group,
                "iou": float(stats["iou"]),
                "f1": float(stats["f1"]),
                "image_bgr": error_map,
            }
        )

    metrics_rows: List[Dict[str, object]] = []
    pixel_confusion_rows: List[Dict[str, object]] = []

    overall_tp = overall_fp = overall_fn = overall_tn = 0
    for cls in CLASS_NAMES:
        tp = per_class_counts[cls]["tp"]
        fp = per_class_counts[cls]["fp"]
        fn = per_class_counts[cls]["fn"]
        tn = per_class_counts[cls]["tn"]
        accuracy = safe_div(tp + tn, tp + fp + fn + tn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        iou = safe_div(tp, tp + fp + fn)

        metrics_rows.append(
            {
                "group": cls,
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "tn_pixels": tn,
                "accuracy": f"{accuracy:.12f}",
                "precision": f"{precision:.12f}",
                "recall": f"{recall:.12f}",
                "f1": f"{f1:.12f}",
                "iou": f"{iou:.12f}",
            }
        )
        pixel_confusion_rows.append(
            {
                "group": cls,
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "tn_pixels": tn,
            }
        )
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        overall_tn += tn

    overall_accuracy = safe_div(overall_tp + overall_tn, overall_tp + overall_fp + overall_fn + overall_tn)
    overall_precision = safe_div(overall_tp, overall_tp + overall_fp)
    overall_recall = safe_div(overall_tp, overall_tp + overall_fn)
    overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)
    overall_iou = safe_div(overall_tp, overall_tp + overall_fp + overall_fn)

    metrics_rows.append(
        {
            "group": "Overall",
            "tp_pixels": overall_tp,
            "fp_pixels": overall_fp,
            "fn_pixels": overall_fn,
            "tn_pixels": overall_tn,
            "accuracy": f"{overall_accuracy:.12f}",
            "precision": f"{overall_precision:.12f}",
            "recall": f"{overall_recall:.12f}",
            "f1": f"{overall_f1:.12f}",
            "iou": f"{overall_iou:.12f}",
        }
    )
    pixel_confusion_rows.append(
        {
            "group": "Overall",
            "tp_pixels": overall_tp,
            "fp_pixels": overall_fp,
            "fn_pixels": overall_fn,
            "tn_pixels": overall_tn,
        }
    )

    with (metrics_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    with (metrics_dir / "pixel_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pixel_confusion_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pixel_confusion_rows)

    render_table_png(metrics_rows, metrics_dir / "metrics_summary.png")
    render_metrics_heatmap(metrics_rows, metrics_dir / "metrics_heatmap.png")

    overall_matrix = np.array(
        [[overall_tn, overall_fp], [overall_fn, overall_tp]],
        dtype=float,
    )
    render_confusion_matrix(
        overall_matrix,
        labels=["Background", "Target"],
        output_path=metrics_dir / "overall_confusion_matrix.png",
        title="Overall Pixel Confusion Matrix",
        normalize=False,
    )
    render_confusion_matrix(
        overall_matrix,
        labels=["Background", "Target"],
        output_path=metrics_dir / "overall_confusion_matrix_normalized.png",
        title="Overall Pixel Confusion Matrix (Normalized)",
        normalize=True,
    )
    write_group_confusion_matrices(per_class_counts, metrics_dir)

    qual_rows.sort(key=lambda item: (item["iou"], item["f1"]))
    render_qualitative_grid(qual_rows[: args.qual_grid_size], metrics_dir / "qualitative_error_map_grid.png")

    readme = """This folder keeps only the deliverables for manual validation:
- gt_overlay/: ground-truth masks drawn on images
- sam2_overlay/: SAM2 predicted masks drawn on images
- pred_vs_gt/: side-by-side comparison images
- metrics/: accuracy, precision, recall, F1, IoU, and confusion matrix outputs
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
