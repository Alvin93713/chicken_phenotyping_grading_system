#!/usr/bin/env python3
"""Run YOLO + SAM on the 40 manual-validation images and score mask metrics.

This evaluator is group-aware:
- `Comb` samples are validated only against `Comb` masks
- `Shank` samples are validated only against `Shank` masks

The manual-validation package is not exhaustively labeled for both classes in
every image, so non-target-class predictions are ignored during scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
from tqdm import tqdm

from paths import validation_manual_dir
from sam2_pipeline import (
    TARGET_CLASSES,
    default_sam_weights,
    default_yolo_weights,
    filter_detections,
    load_models,
    resolve_devices,
    run_sam_masks,
    setup_logging,
)


CLASS_NAMES = ("Comb", "Shank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate manual-validation masks by rerunning YOLO + SAM on images."
    )
    parser.add_argument(
        "--validation-dir",
        default=str(validation_manual_dir()),
        help="Manual validation package directory.",
    )
    parser.add_argument(
        "--yolo-weights",
        default=default_yolo_weights(),
        help="YOLO weights path.",
    )
    parser.add_argument(
        "--sam-weights",
        default=default_sam_weights(),
        help="SAM weights path.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--sam-device", default=None)
    parser.add_argument(
        "--save-pred-masks-dir",
        default="",
        help="Optional directory to save predicted binary masks.",
    )
    parser.add_argument(
        "--pass-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for pass/fail summary. Default: 0.5",
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


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_ground_truth_masks(
    manual_labels_dir: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    gt: Dict[str, Dict[str, np.ndarray]] = {
        cls: {} for cls in CLASS_NAMES
    }

    for cls in CLASS_NAMES:
        json_path = manual_labels_dir / cls.lower() / "_annotations.coco.json"
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


def predict_target_mask(
    image_bgr: np.ndarray,
    target_class_name: str,
    yolo_model,
    sam_model,
    yolo_device: str,
    sam_device: str,
    conf_threshold: float,
) -> np.ndarray:
    with torch.inference_mode():
        yolo_result = yolo_model.predict(
            source=image_bgr,
            conf=conf_threshold,
            verbose=False,
            device=yolo_device,
        )[0]

    filtered = filter_detections(yolo_result, conf_threshold)
    if not filtered:
        return np.zeros(image_bgr.shape[:2], dtype=bool)

    sam_results = run_sam_masks(
        sam_model=sam_model,
        frame_bgr=image_bgr,
        detections=filtered,
        device=sam_device,
    )
    if not sam_results:
        return np.zeros(image_bgr.shape[:2], dtype=bool)

    class_id = next(k for k, v in TARGET_CLASSES.items() if v == target_class_name)
    pred_mask = np.zeros(image_bgr.shape[:2], dtype=bool)
    for det in sam_results:
        if det.class_id != class_id:
            continue
        pred_mask |= det.mask
    return pred_mask


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    setup_logging()

    validation_dir = Path(args.validation_dir).resolve()
    manifest_rows = load_manifest(validation_dir / "manifest.csv")
    gt_masks = load_ground_truth_masks(validation_dir / "manual_labels")

    yolo_device, sam_device = resolve_devices(args.device, args.yolo_device, args.sam_device)
    yolo_model, sam_model = load_models(
        args.yolo_weights,
        args.sam_weights,
        yolo_device,
        sam_device,
    )

    save_dir = Path(args.save_pred_masks_dir).resolve() if args.save_pred_masks_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    per_class_counts: Dict[str, Dict[str, int]] = {
        cls: {"tp": 0, "fp": 0, "fn": 0} for cls in CLASS_NAMES
    }
    per_class_image_ious: Dict[str, List[float]] = {cls: [] for cls in CLASS_NAMES}
    per_class_image_f1s: Dict[str, List[float]] = {cls: [] for cls in CLASS_NAMES}
    per_class_pred_nonempty: Dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    per_class_gt_nonempty: Dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    per_class_pass_count: Dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    per_image_rows: List[Dict[str, object]] = []

    image_iter = tqdm(manifest_rows, desc="Mask validation", unit="image")
    for row in image_iter:
        sample_id = row["sample_id"].strip()
        group = normalize_class_name(row["group"])
        image_path = validation_dir / row["raw_image"]
        gt_mask = gt_masks[group][sample_id]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        pred_mask = predict_target_mask(
            image_bgr=image_bgr,
            target_class_name=group,
            yolo_model=yolo_model,
            sam_model=sam_model,
            yolo_device=yolo_device,
            sam_device=sam_device,
            conf_threshold=args.confidence_threshold,
        )

        if pred_mask.shape != gt_mask.shape:
            raise RuntimeError(
                f"Mask shape mismatch for {sample_id}: pred={pred_mask.shape}, gt={gt_mask.shape}"
            )

        if save_dir is not None:
            out_path = save_dir / f"{sample_id}_{group.lower()}_pred.png"
            cv2.imwrite(str(out_path), (pred_mask.astype(np.uint8) * 255))

        tp = int(np.logical_and(pred_mask, gt_mask).sum())
        fp = int(np.logical_and(pred_mask, np.logical_not(gt_mask)).sum())
        fn = int(np.logical_and(np.logical_not(pred_mask), gt_mask).sum())

        per_class_counts[group]["tp"] += tp
        per_class_counts[group]["fp"] += fp
        per_class_counts[group]["fn"] += fn

        if pred_mask.any():
            per_class_pred_nonempty[group] += 1
        if gt_mask.any():
            per_class_gt_nonempty[group] += 1

        iou = safe_div(tp, tp + fp + fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_class_image_ious[group].append(iou)
        per_class_image_f1s[group].append(f1)
        if iou >= args.pass_iou_threshold:
            per_class_pass_count[group] += 1
        per_image_rows.append(
            {
                "sample_id": sample_id,
                "group": group,
                "gt_pixels": int(gt_mask.sum()),
                "pred_pixels": int(pred_mask.sum()),
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            }
        )

    report_csv = validation_dir / "mask_validation_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(f"Validation dir: {validation_dir}")
    print("Mode: target-class-only segmentation evaluation")
    print(f"YOLO weights: {args.yolo_weights}")
    print(f"SAM weights: {args.sam_weights}")
    print(f"YOLO device: {yolo_device} | SAM device: {sam_device}")
    print(f"Confidence threshold: {args.confidence_threshold:.2f}")
    print(f"Pass IoU threshold: {args.pass_iou_threshold:.2f}")
    print()

    overall_tp = overall_fp = overall_fn = 0
    mean_ious: List[float] = []
    mean_f1s: List[float] = []
    overall_pass_count = 0

    for cls in CLASS_NAMES:
        tp = per_class_counts[cls]["tp"]
        fp = per_class_counts[cls]["fp"]
        fn = per_class_counts[cls]["fn"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        iou = safe_div(tp, tp + fp + fn)
        miou = float(np.mean(per_class_image_ious[cls])) if per_class_image_ious[cls] else 0.0
        mf1 = float(np.mean(per_class_image_f1s[cls])) if per_class_image_f1s[cls] else 0.0

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        mean_ious.extend(per_class_image_ious[cls])
        mean_f1s.extend(per_class_image_f1s[cls])
        overall_pass_count += per_class_pass_count[cls]

        print(f"[{cls}]")
        print(f"  images: {len(per_class_image_ious[cls])}")
        print(f"  nonempty_gt_images: {per_class_gt_nonempty[cls]}")
        print(f"  nonempty_pred_images: {per_class_pred_nonempty[cls]}")
        print(
            f"  pass@iou>={args.pass_iou_threshold:.2f}: "
            f"{per_class_pass_count[cls]}/{len(per_class_image_ious[cls])} "
            f"({per_class_pass_count[cls] / len(per_class_image_ious[cls]):.4f})"
        )
        print(f"  tp_pixels={tp} fp_pixels={fp} fn_pixels={fn}")
        print(f"  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} iou={iou:.4f}")
        print(f"  mean_image_f1={mf1:.4f} mean_image_iou={miou:.4f}")
        print()

    overall_precision = safe_div(overall_tp, overall_tp + overall_fp)
    overall_recall = safe_div(overall_tp, overall_tp + overall_fn)
    overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)
    overall_iou = safe_div(overall_tp, overall_tp + overall_fp + overall_fn)
    overall_mf1 = float(np.mean(mean_f1s)) if mean_f1s else 0.0
    overall_miou = float(np.mean(mean_ious)) if mean_ious else 0.0

    print("[Overall]")
    print(
        f"  pass@iou>={args.pass_iou_threshold:.2f}: "
        f"{overall_pass_count}/{len(per_image_rows)} "
        f"({overall_pass_count / len(per_image_rows):.4f})"
    )
    print(f"  tp_pixels={overall_tp} fp_pixels={overall_fp} fn_pixels={overall_fn}")
    print(
        f"  precision={overall_precision:.4f} "
        f"recall={overall_recall:.4f} "
        f"f1={overall_f1:.4f} "
        f"iou={overall_iou:.4f}"
    )
    print(f"  mean_image_f1={overall_mf1:.4f} mean_image_iou={overall_miou:.4f}")
    print()
    print(f"Per-image report: {report_csv}")
    if save_dir is not None:
        print(f"Saved predicted masks: {save_dir}")


if __name__ == "__main__":
    main()
