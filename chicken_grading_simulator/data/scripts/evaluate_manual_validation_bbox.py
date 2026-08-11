#!/usr/bin/env python3
"""Evaluate SAM manual validation results with bbox metrics.

This validation package is group-aware:
- `Comb` samples are only annotated for `Comb`
- `Shank` samples are only annotated for `Shank`

Predictions for the non-target class in a sample are ignored because the
ground truth is not exhaustively labeled for all classes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from paths import validation_manual_dir

CLASS_NAMES = ("Comb", "Shank")


@dataclass(frozen=True)
class Detection:
    sample_id: str
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM manual validation package with bbox metrics."
    )
    parser.add_argument(
        "--validation-dir",
        default=str(validation_manual_dir()),
        help="Manual validation package directory.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for TP matching. Default: 0.5",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold applied to predictions before fixed-metric scoring.",
    )
    return parser.parse_args()


def normalize_class_name(name: str) -> str:
    text = (name or "").strip().lower()
    if text == "comb":
        return "Comb"
    if text == "shank":
        return "Shank"
    raise ValueError(f"Unsupported class name: {name!r}")


def parse_bbox_xyxy(text: str) -> Tuple[float, float, float, float]:
    values = json.loads(text)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"Invalid bbox_xyxy: {text!r}")
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def bbox_xywh_to_xyxy(values: Iterable[object]) -> Tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in values)
    return (x, y, x + w, y + h)


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


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


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    sample_to_group: Dict[str, str] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["sample_id"].strip()
            sample_to_group[sample_id] = normalize_class_name(row["group"])
    return sample_to_group


def load_ground_truth(manual_labels_dir: Path) -> Dict[str, Dict[str, List[Tuple[float, float, float, float]]]]:
    gt: Dict[str, Dict[str, List[Tuple[float, float, float, float]]]] = {
        cls: defaultdict(list) for cls in CLASS_NAMES
    }

    for cls in CLASS_NAMES:
        json_path = manual_labels_dir / cls.lower() / "_annotations.coco.json"
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_map = {
            int(image["id"]): sample_id_from_image(image)
            for image in data.get("images", [])
        }

        for ann in data.get("annotations", []):
            sample_id = image_map[int(ann["image_id"])]
            gt[cls][sample_id].append(bbox_xywh_to_xyxy(ann["bbox"]))

    return gt


def load_predictions(
    prediction_summary_path: Path,
    sample_to_group: Dict[str, str],
) -> Dict[str, List[Detection]]:
    per_class: Dict[str, List[Detection]] = {cls: [] for cls in CLASS_NAMES}

    with prediction_summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["sample_id"].strip()
            target_group = sample_to_group.get(sample_id)
            if target_group is None:
                continue

            class_name = normalize_class_name(row["class_name"])
            if class_name != target_group:
                continue

            det = Detection(
                sample_id=sample_id,
                class_name=class_name,
                confidence=float(row["confidence"]),
                bbox_xyxy=parse_bbox_xyxy(row["bbox_xyxy"]),
            )
            per_class[class_name].append(det)

    return per_class


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_fixed_threshold(
    predictions: List[Detection],
    gt_by_sample: Dict[str, List[Tuple[float, float, float, float]]],
    iou_threshold: float,
    confidence_threshold: float,
) -> Dict[str, float]:
    filtered = [det for det in predictions if det.confidence >= confidence_threshold]
    filtered.sort(key=lambda det: det.confidence, reverse=True)

    unmatched = {
        sample_id: [False] * len(boxes)
        for sample_id, boxes in gt_by_sample.items()
    }

    tp = 0
    fp = 0
    matched_ious: List[float] = []

    for det in filtered:
        gt_boxes = gt_by_sample.get(det.sample_id, [])
        used = unmatched.get(det.sample_id, [])

        best_idx = -1
        best_iou = 0.0
        for idx, gt_box in enumerate(gt_boxes):
            if used[idx]:
                continue
            overlap = iou_xyxy(det.bbox_xyxy, gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            used[best_idx] = True
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1

    total_gt = sum(len(boxes) for boxes in gt_by_sample.values())
    fn = total_gt - tp
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0,
        "num_predictions": float(len(filtered)),
        "num_ground_truth": float(total_gt),
    }


def evaluate_ap_and_best_f1(
    predictions: List[Detection],
    gt_by_sample: Dict[str, List[Tuple[float, float, float, float]]],
    iou_threshold: float,
) -> Dict[str, float]:
    predictions = sorted(predictions, key=lambda det: det.confidence, reverse=True)
    total_gt = sum(len(boxes) for boxes in gt_by_sample.values())

    unmatched = {
        sample_id: [False] * len(boxes)
        for sample_id, boxes in gt_by_sample.items()
    }

    tps: List[int] = []
    fps: List[int] = []
    recalls: List[float] = []
    precisions: List[float] = []
    best_f1 = 0.0
    best_threshold = 1.0

    cum_tp = 0
    cum_fp = 0

    for det in predictions:
        gt_boxes = gt_by_sample.get(det.sample_id, [])
        used = unmatched.get(det.sample_id, [])

        best_idx = -1
        best_iou = 0.0
        for idx, gt_box in enumerate(gt_boxes):
            if used[idx]:
                continue
            overlap = iou_xyxy(det.bbox_xyxy, gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            used[best_idx] = True
            cum_tp += 1
            tps.append(1)
            fps.append(0)
        else:
            cum_fp += 1
            tps.append(0)
            fps.append(1)

        recall = (cum_tp / total_gt) if total_gt else 0.0
        precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) else 0.0
        recalls.append(recall)
        precisions.append(precision)

        _, _, f1 = precision_recall_f1(cum_tp, cum_fp, total_gt - cum_tp)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = det.confidence

    if not recalls:
        return {
            "ap50": 0.0,
            "best_f1": 0.0,
            "best_f1_threshold": 1.0,
        }

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])

    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]

    return {
        "ap50": ap,
        "best_f1": best_f1,
        "best_f1_threshold": best_threshold,
    }


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    validation_dir = Path(args.validation_dir).resolve()
    manifest_path = validation_dir / "manifest.csv"
    manual_labels_dir = validation_dir / "manual_labels"
    prediction_summary_path = validation_dir / "prediction_summary.csv"

    sample_to_group = load_manifest(manifest_path)
    gt = load_ground_truth(manual_labels_dir)
    predictions = load_predictions(prediction_summary_path, sample_to_group)

    print(f"Validation dir: {validation_dir}")
    print(f"Mode: target-class-only (non-target detections are ignored)")
    print(f"IoU threshold: {args.iou_threshold:.2f}")
    print(f"Confidence threshold: {args.confidence_threshold:.4f}")
    print()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_matched_iou = 0.0
    total_matched_count = 0
    ap_values: List[float] = []

    for cls in CLASS_NAMES:
        gt_by_sample = {sample_id: gt[cls].get(sample_id, []) for sample_id, group in sample_to_group.items() if group == cls}
        class_predictions = predictions[cls]

        fixed = evaluate_fixed_threshold(
            predictions=class_predictions,
            gt_by_sample=gt_by_sample,
            iou_threshold=args.iou_threshold,
            confidence_threshold=args.confidence_threshold,
        )
        ranking = evaluate_ap_and_best_f1(
            predictions=class_predictions,
            gt_by_sample=gt_by_sample,
            iou_threshold=args.iou_threshold,
        )

        tp = int(fixed["tp"])
        fp = int(fixed["fp"])
        fn = int(fixed["fn"])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_matched_iou += fixed["mean_iou"] * tp
        total_matched_count += tp
        ap_values.append(ranking["ap50"])

        print(f"[{cls}]")
        print(f"  samples: {len(gt_by_sample)}")
        print(f"  ground_truth_boxes: {int(fixed['num_ground_truth'])}")
        print(f"  predictions_used: {int(fixed['num_predictions'])}")
        print(f"  TP={tp} FP={fp} FN={fn}")
        print(
            "  precision="
            f"{format_metric(fixed['precision'])} "
            "recall="
            f"{format_metric(fixed['recall'])} "
            "f1="
            f"{format_metric(fixed['f1'])}"
        )
        print(f"  mean_iou_matched: {format_metric(fixed['mean_iou'])}")
        print(f"  AP@{args.iou_threshold:.2f}: {format_metric(ranking['ap50'])}")
        print(
            "  best_f1="
            f"{format_metric(ranking['best_f1'])} "
            "at_confidence>="
            f"{format_metric(ranking['best_f1_threshold'])}"
        )
        print()

    overall_precision, overall_recall, overall_f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    mean_iou = (total_matched_iou / total_matched_count) if total_matched_count else 0.0
    map50 = (sum(ap_values) / len(ap_values)) if ap_values else 0.0

    print("[Overall]")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
    print(
        "  precision="
        f"{format_metric(overall_precision)} "
        "recall="
        f"{format_metric(overall_recall)} "
        "f1="
        f"{format_metric(overall_f1)}"
    )
    print(f"  mean_iou_matched: {format_metric(mean_iou)}")
    print(f"  mAP@{args.iou_threshold:.2f}: {format_metric(map50)}")


if __name__ == "__main__":
    main()
