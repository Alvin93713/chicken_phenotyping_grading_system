#!/usr/bin/env python3
"""
YOLOv11 + SAM (SAM2 weights) mask measurement pipeline.

Commands:
  - run-inference: run YOLO detection + SAM segmentation, write annotated video + metadata.json
  - export-metrics: summarize mask measurements per video into a CSV (pixels)
  - convert-cm: convert CSV pixel measurements to centimetres / cm^2 via a scale factor
  - merge-dlc: merge exported metrics with DeepLabCut features_minimal.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import SAM, YOLO

from paths import (
    data_best_videos_dir,
    local_sam_weight_path,
    local_yolo_weights_path,
    outputs_default_dir,
)

TARGET_CLASSES = {0: "Comb", 1: "Shank"}
CLASS_COLOR = {
    0: (0, 0, 255),  # Comb -> red (BGR)
    1: (215, 47, 31),  # Shank -> royal blue (BGR)
}
SHANK_METRIC_X_SPAN = "x_span"
SHANK_METRIC_PRINCIPAL_AXIS_PERP_MEDIAN = "principal_axis_perpendicular_median_width"
SHANK_METRIC_CONFIG = {
    SHANK_METRIC_X_SPAN: {
        "detection_field": "mask_width_px",
        "video_prefix": "shank_width_px",
        "file_suffix": None,
    },
    SHANK_METRIC_PRINCIPAL_AXIS_PERP_MEDIAN: {
        "detection_field": "principal_axis_perpendicular_median_width_px",
        "video_prefix": "shank_principal_axis_perpendicular_median_width_px",
        "file_suffix": "principal_axis_perpendicular_median_width",
    },
}


@dataclass
class DetectionResult:
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]
    mask: np.ndarray
    mask_width: int
    principal_axis_perpendicular_median_width: Optional[float] = None
    principal_axis_angle_deg: Optional[float] = None


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_devices(
    device_override: Optional[str],
    yolo_device_override: Optional[str],
    sam_device_override: Optional[str],
) -> Tuple[str, str]:
    default_device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device = yolo_device_override or default_device
    sam_device = sam_device_override or default_device

    if not torch.cuda.is_available():
        if yolo_device == "cuda":
            logging.warning("CUDA was requested for YOLO, but this PyTorch build has no CUDA support. Falling back to CPU.")
            yolo_device = "cpu"
        if sam_device == "cuda":
            logging.warning("CUDA was requested for SAM, but this PyTorch build has no CUDA support. Falling back to CPU.")
            sam_device = "cpu"

    return yolo_device, sam_device


def load_models(
    yolo_weights: str,
    sam_weights: str,
    yolo_device: str,
    sam_device: str,
) -> Tuple[YOLO, SAM]:
    logging.info("Loading YOLO weights from %s on %s", yolo_weights, yolo_device)
    yolo_model = YOLO(yolo_weights)
    yolo_model.to(yolo_device)

    logging.info("Loading SAM weights from %s on %s", sam_weights, sam_device)
    sam_model = SAM(sam_weights)
    sam_model.to(sam_device)
    return yolo_model, sam_model


def default_metrics_csv_name(shank_metric_method: str, cm: bool = False) -> str:
    config = SHANK_METRIC_CONFIG[shank_metric_method]
    suffix = config["file_suffix"]
    if suffix is None:
        return "comb_shank_metrics_cm.csv" if cm else "comb_shank_metrics.csv"
    base = f"comb_shank_metrics_{suffix}"
    return f"{base}_cm.csv" if cm else f"{base}.csv"


def shank_metric_detection_field(shank_metric_method: str) -> str:
    return str(SHANK_METRIC_CONFIG[shank_metric_method]["detection_field"])


def shank_metric_video_prefix(shank_metric_method: str) -> str:
    return str(SHANK_METRIC_CONFIG[shank_metric_method]["video_prefix"])


def collect_video_paths(videos_dir: Path) -> List[Path]:
    if not videos_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {videos_dir}")
    video_paths = sorted(
        [p for p in videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    )
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {videos_dir}")
    return video_paths


def filter_detections(result, conf_threshold: float) -> List[Tuple[int, float, np.ndarray]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []

    def iou_xyxy(b1: np.ndarray, b2: np.ndarray) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        a1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
        a2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
        denom = a1 + a2 - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    def nms_per_class(
        dets: List[Tuple[int, float, np.ndarray]],
        iou_thr: float = 0.4,
    ) -> List[Tuple[int, float, np.ndarray]]:
        if not dets:
            return []
        dets_sorted = sorted(dets, key=lambda x: x[1], reverse=True)
        kept: List[Tuple[int, float, np.ndarray]] = []
        for det in dets_sorted:
            _, _, bbox = det
            suppress = False
            for kept_det in kept:
                if iou_xyxy(bbox, kept_det[2]) > iou_thr:
                    suppress = True
                    break
            if not suppress:
                kept.append(det)
        return kept

    class_ids = boxes.cls.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()

    grouped: Dict[int, List[Tuple[int, float, np.ndarray]]] = {k: [] for k in TARGET_CLASSES.keys()}
    for cls_id, conf, bbox in zip(class_ids, confidences, xyxy):
        cls_id = int(cls_id)
        if cls_id not in TARGET_CLASSES:
            continue
        if conf < conf_threshold:
            continue
        grouped.setdefault(cls_id, []).append((cls_id, float(conf), bbox))

    detections: List[Tuple[int, float, np.ndarray]] = []
    for cls_id, dets in grouped.items():
        if not dets:
            continue
        kept = nms_per_class(dets, iou_thr=0.4)
        detections.extend(kept)
    return detections


def compute_principal_axis_perpendicular_median_width(
    mask: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None, None
    if xs.size == 1:
        return 1.0, 0.0

    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    centered = coords - coords.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0):
        return 1.0, 0.0

    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        fallback = float(xs.max() - xs.min() + 1)
        return fallback, 0.0

    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    norm = float(np.linalg.norm(principal))
    if not np.isfinite(norm) or norm == 0.0:
        fallback = float(xs.max() - xs.min() + 1)
        return fallback, 0.0
    principal = principal / norm
    perpendicular = np.array([-principal[1], principal[0]], dtype=np.float32)

    along = centered @ principal
    across = centered @ perpendicular
    slice_ids = np.rint(along).astype(np.int32)

    widths: List[float] = []
    for slice_id in np.unique(slice_ids):
        slice_across = across[slice_ids == slice_id]
        if slice_across.size == 0:
            continue
        widths.append(float(slice_across.max() - slice_across.min() + 1.0))

    if not widths:
        fallback = float(xs.max() - xs.min() + 1)
        return fallback, 0.0

    angle_deg = float((np.degrees(np.arctan2(principal[1], principal[0])) + 180.0) % 180.0)
    return float(np.median(np.asarray(widths, dtype=np.float32))), angle_deg


def run_sam_masks(
    sam_model: SAM,
    frame_bgr: np.ndarray,
    detections: List[Tuple[int, float, np.ndarray]],
    device: str,
) -> List[DetectionResult]:
    if not detections:
        return []

    bboxes = [bbox.tolist() for _, _, bbox in detections]
    sam_result = sam_model.predict(
        source=frame_bgr,
        bboxes=bboxes,
        verbose=False,
        device=device,
    )[0]
    masks_obj = getattr(sam_result, "masks", None)
    if masks_obj is None or masks_obj.data is None:
        return []
    mask_stack = masks_obj.data.detach().cpu().numpy().astype(bool)

    results: List[DetectionResult] = []
    for (cls_id, conf, bbox), mask in zip(detections, mask_stack):
        if not mask.any():
            continue
        xs = np.where(mask)[1]
        mask_width = int(xs.max() - xs.min() + 1) if xs.size else 0
        principal_axis_width = None
        principal_axis_angle = None
        if cls_id == 1:
            principal_axis_width, principal_axis_angle = compute_principal_axis_perpendicular_median_width(mask)
        results.append(
            DetectionResult(
                class_id=cls_id,
                confidence=conf,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                mask=mask,
                mask_width=mask_width,
                principal_axis_perpendicular_median_width=principal_axis_width,
                principal_axis_angle_deg=principal_axis_angle,
            )
        )
    return results


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_progress(
    progress_file: Optional[str],
    video_name: str,
    completed_frames: int,
    total_frames: Optional[int],
    status: str,
) -> None:
    if not progress_file:
        return
    progress_path = Path(progress_file)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": video_name,
        "completed_frames": completed_frames,
        "total_frames": total_frames,
        "status": status,
    }
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def overlay_detections(frame_bgr: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    if not detections:
        return frame_bgr

    overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
    annotated = frame_bgr.copy()

    for det in detections:
        color = CLASS_COLOR.get(det.class_id, (255, 255, 255))
        overlay[det.mask] = color

    annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.5, 0.0)

    for det in detections:
        color = CLASS_COLOR.get(det.class_id, (255, 255, 255))
        x1, y1, x2, y2 = map(int, det.bbox)
        label = f"{TARGET_CLASSES.get(det.class_id, det.class_id)} {det.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def cmd_run_inference(args: argparse.Namespace) -> None:
    yolo_device, sam_device = resolve_devices(args.device, args.yolo_device, args.sam_device)
    logging.info("YOLO device: %s | SAM device: %s", yolo_device, sam_device)

    yolo_model, sam_model = load_models(
        args.yolo_weights,
        args.sam_weights,
        yolo_device,
        sam_device,
    )

    video_paths = collect_video_paths(Path(args.videos_dir))
    if args.max_videos is not None:
        video_paths = video_paths[: args.max_videos]
    logging.info("Found %d video(s) to process.", len(video_paths))

    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    summary: Dict[str, Dict] = {}

    video_iter = tqdm(video_paths, desc="Videos", unit="video")
    for video_path in video_iter:
        video_iter.set_postfix(video=video_path.name)
        video_out_dir = output_root / video_path.stem
        ensure_dir(video_out_dir)

        metadata_path = video_out_dir / "metadata.json"
        if args.skip_existing and metadata_path.exists():
            logging.info("Skipping %s (metadata already exists)", video_path.name)
            with metadata_path.open("r", encoding="utf-8") as f:
                summary[video_path.name] = json.load(f)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error("Failed to open video: %s", video_path)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        stride = max(1, int(args.frame_stride))
        total_processed_target = ((frame_total - 1) // stride + 1) if frame_total else None
        write_progress(args.progress_file, video_path.name, 0, total_processed_target, "running")

        output_video_path = video_out_dir / "annotated.mp4"
        writer = None
        if not args.skip_annotated_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        metadata: Dict[str, object] = {
            "video": video_path.name,
            "fps": fps,
            "frame_width": width,
            "frame_height": height,
            "total_frames": 0,
            "processed_frames": 0,
            "frame_stride": stride,
            "detections_per_class": {name: 0 for name in TARGET_CLASSES.values()},
            "annotated_video": str(output_video_path),
            "sample_frames": [],
            "frames": [],
        }

        saved_samples = 0
        frame_index = 0
        samples_dir = video_out_dir / "samples"
        ensure_dir(samples_dir)

        frame_iter = tqdm(total=frame_total, desc=video_path.name, unit="frame", leave=False)
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                metadata["total_frames"] = int(metadata["total_frames"]) + 1  # type: ignore[arg-type]

                if frame_index % stride != 0:
                    if writer is not None:
                        writer.write(frame_bgr)
                    frame_index += 1
                    frame_iter.update(1)
                    continue

                metadata["processed_frames"] = int(metadata["processed_frames"]) + 1  # type: ignore[arg-type]

                with torch.inference_mode():
                    yolo_result = yolo_model.predict(
                        source=frame_bgr,
                        conf=args.confidence_threshold,
                        verbose=False,
                        device=yolo_device,
                    )[0]

                filtered = filter_detections(yolo_result, args.confidence_threshold)
                detection_results: List[DetectionResult] = []
                if filtered:
                    detection_results = run_sam_masks(
                        sam_model,
                        frame_bgr,
                        filtered,
                        device=sam_device,
                    )

                annotated = overlay_detections(frame_bgr, detection_results)
                if writer is not None:
                    writer.write(annotated)

                if detection_results:
                    frame_record: Dict[str, object] = {"frame_index": frame_index, "detections": []}
                    frame_hsv = None
                    for det in detection_results:
                        class_name = TARGET_CLASSES.get(det.class_id, str(det.class_id))
                        detections_per_class = metadata["detections_per_class"]
                        detections_per_class[class_name] = int(detections_per_class[class_name]) + 1  # type: ignore[index]

                        det_record: Dict[str, object] = {
                            "class_id": det.class_id,
                            "class_name": class_name,
                            "confidence": round(det.confidence, 4),
                            "bbox_xyxy": [round(x, 2) for x in det.bbox],
                            "bbox_width_px": round(det.bbox[2] - det.bbox[0], 2),
                            "mask_width_px": det.mask_width,
                            "mask_x_span_px": det.mask_width,
                            "mask_area": int(det.mask.sum()),
                        }

                        if det.class_id == 0:  # Comb
                            mask_u8 = det.mask.astype(np.uint8) * 255
                            b, g, r, _ = cv2.mean(frame_bgr, mask=mask_u8)
                            if frame_hsv is None:
                                frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                            h, s, v, _ = cv2.mean(frame_hsv, mask=mask_u8)
                            denom = b + g + r
                            det_record["mean_bgr"] = [round(b, 2), round(g, 2), round(r, 2)]
                            det_record["mean_hsv"] = [round(h, 2), round(s, 2), round(v, 2)]
                            det_record["red_ratio"] = round(r / denom, 6) if denom > 0 else None
                        elif det.class_id == 1:  # Shank
                            if det.principal_axis_perpendicular_median_width is not None:
                                det_record["principal_axis_perpendicular_median_width_px"] = round(
                                    det.principal_axis_perpendicular_median_width, 2
                                )
                            if det.principal_axis_angle_deg is not None:
                                det_record["principal_axis_angle_deg"] = round(det.principal_axis_angle_deg, 2)

                        frame_record["detections"].append(det_record)  # type: ignore[union-attr]

                    metadata["frames"].append(frame_record)  # type: ignore[union-attr]

                if saved_samples < args.sample_frames:
                    sample_path = samples_dir / f"frame_{frame_index:05d}.png"
                    cv2.imwrite(str(sample_path), annotated)
                    metadata["sample_frames"].append(str(sample_path))  # type: ignore[union-attr]
                    saved_samples += 1

                frame_index += 1
                frame_iter.update(1)
                write_progress(
                    args.progress_file,
                    video_path.name,
                    int(metadata["processed_frames"]),
                    total_processed_target,
                    "running",
                )
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            frame_iter.close()

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logging.info(
            "Finished %s | frames: %d | detections: %s",
            video_path.name,
            metadata["total_frames"],
            metadata["detections_per_class"],
        )
        summary[video_path.name] = metadata
        write_progress(
            args.progress_file,
            video_path.name,
            int(metadata["processed_frames"]),
            total_processed_target,
            "done",
        )

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("All done. Summary saved to %s", summary_path)


def compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
    }


def select_primary_shank_detection(detections: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    shank_detections = [det for det in detections if det.get("class_name") == "Shank"]
    if not shank_detections:
        return None

    def shank_sort_key(det: Dict[str, object]) -> Tuple[float, float]:
        area = det.get("mask_area")
        confidence = det.get("confidence")
        area_value = float(area) if area is not None else -1.0
        confidence_value = float(confidence) if confidence is not None else -1.0
        return area_value, confidence_value

    return max(shank_detections, key=shank_sort_key)


def fmt(value: Optional[float], decimals: int = 2) -> str:
    if value is None or np.isnan(value):
        return ""
    return f"{value:.{decimals}f}"


def cmd_export_metrics(args: argparse.Namespace) -> None:
    outputs_dir = Path(args.outputs_dir)
    video_paths = sorted(outputs_dir.glob("*/metadata.json"))
    if not video_paths:
        raise FileNotFoundError(f"No metadata.json found under {outputs_dir}")

    shank_metric_field = shank_metric_detection_field(args.shank_metric_method)
    shank_prefix = shank_metric_video_prefix(args.shank_metric_method)

    rows: List[Dict[str, object]] = []
    missing_shank_metric = 0
    for meta_path in video_paths:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        video_name = data.get("video", meta_path.parent.name)
        video_id = Path(str(video_name)).stem

        comb_areas: List[float] = []
        comb_b: List[float] = []
        comb_g: List[float] = []
        comb_r: List[float] = []
        comb_h: List[float] = []
        comb_s: List[float] = []
        comb_v: List[float] = []
        comb_red_ratio: List[float] = []
        shank_widths: List[float] = []

        for frame in data.get("frames", []):
            frame_detections = frame.get("detections", [])
            for detection in frame_detections:
                class_name = detection.get("class_name")
                if class_name == "Comb":
                    area = detection.get("mask_area")
                    if area is not None:
                        comb_areas.append(float(area))
                    mean_bgr = detection.get("mean_bgr")
                    if mean_bgr and len(mean_bgr) == 3:
                        comb_b.append(float(mean_bgr[0]))
                        comb_g.append(float(mean_bgr[1]))
                        comb_r.append(float(mean_bgr[2]))
                    mean_hsv = detection.get("mean_hsv")
                    if mean_hsv and len(mean_hsv) == 3:
                        comb_h.append(float(mean_hsv[0]))
                        comb_s.append(float(mean_hsv[1]))
                        comb_v.append(float(mean_hsv[2]))
                    red_ratio = detection.get("red_ratio")
                    if red_ratio is not None:
                        comb_red_ratio.append(float(red_ratio))
            primary_shank = select_primary_shank_detection(frame_detections)
            if primary_shank is not None:
                width = primary_shank.get(shank_metric_field)
                if width is None and shank_metric_field != "mask_width_px":
                    missing_shank_metric += 1
                if width is None:
                    width = primary_shank.get("mask_width_px")
                if width is None:
                    bbox = primary_shank.get("bbox_xyxy")
                    if bbox and len(bbox) == 4:
                        width = float(bbox[2]) - float(bbox[0])
                if width is not None:
                    shank_widths.append(float(width))

        comb_stats = compute_stats(comb_areas)
        shank_stats = compute_stats(shank_widths)
        b_stats = compute_stats(comb_b)
        g_stats = compute_stats(comb_g)
        r_stats = compute_stats(comb_r)
        h_stats = compute_stats(comb_h)
        s_stats = compute_stats(comb_s)
        v_stats = compute_stats(comb_v)
        rr_stats = compute_stats(comb_red_ratio)

        if args.minimal:
            rows.append(
                {
                    "video_id": video_id,
                    "comb_area_px2_median": fmt(comb_stats["median"]),
                    f"{shank_prefix}_median": fmt(shank_stats["median"]),
                    "comb_color_red_ratio_mean": fmt(rr_stats["mean"], decimals=6),
                }
            )
        else:
            rows.append(
                {
                    "video_id": video_id,
                    "video": video_name,
                    "comb_count": comb_stats["count"],
                    "comb_area_px2_mean": fmt(comb_stats["mean"]),
                    "comb_area_px2_median": fmt(comb_stats["median"]),
                    "comb_area_px2_min": fmt(comb_stats["min"]),
                    "comb_area_px2_max": fmt(comb_stats["max"]),
                    "comb_area_px2_std": fmt(comb_stats["std"]),
                    "comb_color_b_mean": fmt(b_stats["mean"]),
                    "comb_color_g_mean": fmt(g_stats["mean"]),
                    "comb_color_r_mean": fmt(r_stats["mean"]),
                    "comb_color_h_mean": fmt(h_stats["mean"]),
                    "comb_color_s_mean": fmt(s_stats["mean"]),
                    "comb_color_v_mean": fmt(v_stats["mean"]),
                    "comb_red_ratio_mean": fmt(rr_stats["mean"], decimals=6),
                    "comb_red_ratio_median": fmt(rr_stats["median"], decimals=6),
                    "shank_count": shank_stats["count"],
                    f"{shank_prefix}_mean": fmt(shank_stats["mean"]),
                    f"{shank_prefix}_median": fmt(shank_stats["median"]),
                    f"{shank_prefix}_min": fmt(shank_stats["min"]),
                    f"{shank_prefix}_max": fmt(shank_stats["max"]),
                    f"{shank_prefix}_std": fmt(shank_stats["std"]),
                }
            )

    if not rows:
        logging.warning("No metrics found to export.")
        return

    if args.shank_metric_method != SHANK_METRIC_X_SPAN and missing_shank_metric > 0:
        raise RuntimeError(
            "Selected shank metric is missing from metadata.json. "
            "Re-run run-inference to backfill principal-axis perpendicular widths before export."
        )

    csv_path = Path(args.csv_path) if args.csv_path else outputs_dir / default_metrics_csv_name(args.shank_metric_method)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Saved %d rows to %s", len(rows), csv_path)


def cmd_convert_to_cm(args: argparse.Namespace) -> None:
    input_path = (
        Path(args.input_csv)
        if args.input_csv
        else default_outputs_dir() / default_metrics_csv_name(args.shank_metric_method)
    )
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else default_outputs_dir() / default_metrics_csv_name(args.shank_metric_method, cm=True)
    )

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("Input CSV is empty.")

    converted_rows: List[Dict[str, str]] = []
    for row in rows:
        converted = row.copy()
        for key, value in list(row.items()):
            if not value or not value.strip():
                continue
            if key.startswith("comb_area_px2"):
                converted[key.replace("_px2", "_cm2")] = f"{float(value) * (args.cm_per_pixel**2):.6f}"
            elif key.startswith("shank_") and "_px" in key:
                converted[key.replace("_px", "_cm")] = f"{float(value) * args.cm_per_pixel:.6f}"
        converted_rows.append(converted)

    base_fields = list(rows[0].keys())
    extra_fields: List[str] = []
    for key in base_fields:
        if key.startswith("comb_area_px2"):
            extra_key = key.replace("_px2", "_cm2")
            if extra_key not in base_fields:
                extra_fields.append(extra_key)
        elif key.startswith("shank_") and "_px" in key:
            extra_key = key.replace("_px", "_cm")
            if extra_key not in base_fields:
                extra_fields.append(extra_key)
    fieldnames = base_fields + extra_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted_rows)

    logging.info(
        "Wrote %d rows to %s (1 px = %s cm).",
        len(converted_rows),
        output_path,
        args.cm_per_pixel,
    )


def default_outputs_dir() -> Path:
    return outputs_default_dir()


def default_videos_dir() -> Path:
    return data_best_videos_dir()


def default_yolo_weights() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    yolov11_dir = repo_root / "Yolov11"
    candidates = [
        yolov11_dir / "best.pt",
        yolov11_dir / "runs" / "detect" / "train" / "weights" / "best.pt",
        local_yolo_weights_path(),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "best.pt"


def default_sam_weights() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    yolov11_dir = repo_root / "Yolov11"
    candidates = [
        local_sam_weight_path("sam2_b.pt"),
        local_sam_weight_path("sam_b.pt"),
        local_sam_weight_path("mobile_sam.pt"),
        yolov11_dir / "sam2_b.pt",
        yolov11_dir / "sam_b.pt",
        yolov11_dir / "mobile_sam.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "sam2_b.pt"


def cmd_merge_dlc(args: argparse.Namespace) -> None:
    dlc_csv = Path(args.dlc_csv)
    sam_csv = Path(args.sam_csv)
    output_csv = Path(args.output_csv)

    with dlc_csv.open("r", encoding="utf-8") as f:
        dlc_reader = csv.DictReader(f)
        dlc_rows = list(dlc_reader)
        dlc_fields = dlc_reader.fieldnames or []

    with sam_csv.open("r", encoding="utf-8") as f:
        sam_reader = csv.DictReader(f)
        sam_rows = list(sam_reader)
        sam_fields = sam_reader.fieldnames or []

    if not dlc_rows:
        raise RuntimeError(f"DLC CSV is empty: {dlc_csv}")
    if not sam_rows:
        raise RuntimeError(f"SAM metrics CSV is empty: {sam_csv}")

    key = args.key
    sam_by_key: Dict[str, Dict[str, str]] = {}
    for row in sam_rows:
        k = str(row.get(key, "")).strip()
        if not k:
            continue
        sam_by_key[k] = row

    extra_fields = [f for f in sam_fields if f not in dlc_fields]
    merged_fields = dlc_fields + extra_fields

    merged_rows: List[Dict[str, str]] = []
    missing = 0
    for row in dlc_rows:
        k = str(row.get(key, "")).strip()
        merged = dict(row)
        sam_row = sam_by_key.get(k)
        if sam_row is None:
            missing += 1
            for f in extra_fields:
                merged[f] = ""
        else:
            for f in extra_fields:
                merged[f] = sam_row.get(f, "")
        merged_rows.append(merged)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    logging.info(
        "Merged %d DLC row(s) with SAM metrics (%d missing matches) -> %s",
        len(merged_rows),
        missing,
        output_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO + SAM mask measurement pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-inference
    p_run = subparsers.add_parser("run-inference", help="Run YOLO + SAM inference on videos.")
    p_run.add_argument("--videos-dir", default=str(default_videos_dir()))
    p_run.add_argument("--output-dir", default=str(default_outputs_dir()))
    p_run.add_argument("--yolo-weights", default=default_yolo_weights())
    p_run.add_argument("--sam-weights", default=default_sam_weights())
    p_run.add_argument("--confidence-threshold", type=float, default=0.25)
    p_run.add_argument("--sample-frames", type=int, default=10)
    p_run.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run inference every N frames. Default 1 processes every frame.",
    )
    p_run.add_argument("--device", default=None)
    p_run.add_argument("--yolo-device", default=None)
    p_run.add_argument("--sam-device", default=None)
    p_run.add_argument("--max-videos", type=int, default=None)
    p_run.add_argument("--skip-existing", action="store_true")
    p_run.add_argument(
        "--skip-annotated-video",
        action="store_true",
        help="Do not rewrite annotated.mp4; useful when only refreshing metadata-derived metrics.",
    )
    p_run.add_argument("--progress-file", default=None)
    p_run.set_defaults(func=cmd_run_inference)

    # export metrics
    p_export = subparsers.add_parser("export-metrics", help="Export Comb/Shank statistics from metadata.")
    p_export.add_argument("--outputs-dir", default=str(default_outputs_dir()))
    p_export.add_argument("--csv-path", default=None)
    p_export.add_argument(
        "--shank-metric-method",
        choices=tuple(SHANK_METRIC_CONFIG.keys()),
        default=SHANK_METRIC_PRINCIPAL_AXIS_PERP_MEDIAN,
        help="Which Shank metric to export. The principal-axis option uses median cross-sectional width.",
    )
    p_export.add_argument(
        "--minimal",
        action="store_true",
        help="Only export 3 features: comb_area_median, selected shank metric median, comb_color_red_ratio_mean.",
    )
    p_export.set_defaults(func=cmd_export_metrics)

    # convert to cm
    p_convert = subparsers.add_parser("convert-cm", help="Convert pixel measurements to centimetres.")
    p_convert.add_argument("--input-csv", default=None)
    p_convert.add_argument("--output-csv", default=None)
    p_convert.add_argument("--cm-per-pixel", type=float, required=True)
    p_convert.add_argument(
        "--shank-metric-method",
        choices=tuple(SHANK_METRIC_CONFIG.keys()),
        default=SHANK_METRIC_PRINCIPAL_AXIS_PERP_MEDIAN,
        help="Used only when input/output paths are omitted, to choose the matching default filenames.",
    )
    p_convert.set_defaults(func=cmd_convert_to_cm)

    # merge DLC
    p_merge = subparsers.add_parser("merge-dlc", help="Merge YOLO/SAM metrics with DeepLabCut features.")
    p_merge.add_argument("--dlc-csv", required=True)
    p_merge.add_argument("--sam-csv", required=True)
    p_merge.add_argument("--output-csv", required=True)
    p_merge.add_argument("--key", default="video_id")
    p_merge.set_defaults(func=cmd_merge_dlc)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
