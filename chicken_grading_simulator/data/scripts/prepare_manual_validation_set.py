#!/usr/bin/env python3
"""
Build a manual-validation package from YOLOv11 + SAM outputs.

The package contains:
- images_raw: original frames extracted from source videos
- sam_overlay: SAM overlay images exported by the pipeline
- manual_labels: where you save ImageJ binary masks
- manifest.csv: mapping table for all samples
- prediction_summary.csv: per-sample model stats from metadata
- README.md: quick labeling instructions
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from paths import data_best_videos_dir, outputs_best_videos_dir, validation_manual_dir

FRAME_RE = re.compile(r"frame_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


@dataclass
class Candidate:
    video_name: str
    video_id: str
    frame_index: int
    overlay_path: Path
    detections: List[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default=str(outputs_best_videos_dir()),
        help="Directory containing per-video metadata.json and samples/",
    )
    parser.add_argument(
        "--videos-dir",
        default=str(data_best_videos_dir()),
        help="Directory containing original videos.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(validation_manual_dir()),
        help="Output directory for the manual validation package.",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def parse_frame_index(path: Path) -> Optional[int]:
    match = FRAME_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def resolve_overlay_path(raw_path: str, meta_parent: Path, outputs_dir: Path) -> Optional[Path]:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        meta_parent / path,
        outputs_dir / path,
        outputs_dir.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def collect_candidates(outputs_dir: Path) -> List[Candidate]:
    metadata_paths = sorted(outputs_dir.glob("*/metadata.json"))
    candidates: List[Candidate] = []

    for meta_path in metadata_paths:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = str(data.get("video") or f"{meta_path.parent.name}.mp4")
        video_id = Path(video_name).stem

        frames_map: Dict[int, List[dict]] = {}
        for frame_record in data.get("frames", []):
            frame_idx = frame_record.get("frame_index")
            if isinstance(frame_idx, int):
                frames_map[frame_idx] = frame_record.get("detections", [])

        for raw_sample_path in data.get("sample_frames", []):
            overlay_path = resolve_overlay_path(str(raw_sample_path), meta_path.parent, outputs_dir)
            if overlay_path is None:
                continue
            frame_index = parse_frame_index(Path(str(raw_sample_path)))
            if frame_index is None:
                continue
            detections = frames_map.get(frame_index, [])
            candidates.append(
                Candidate(
                    video_name=video_name,
                    video_id=video_id,
                    frame_index=frame_index,
                    overlay_path=overlay_path,
                    detections=detections,
                )
            )
    return candidates


def select_candidates(candidates: List[Candidate], num_samples: int, seed: int) -> List[Candidate]:
    if not candidates:
        return []

    rng = random.Random(seed)
    by_video: Dict[str, List[Candidate]] = {}
    for item in candidates:
        by_video.setdefault(item.video_id, []).append(item)

    selected: List[Candidate] = []
    used_keys = set()

    video_ids = list(by_video.keys())
    rng.shuffle(video_ids)

    for video_id in video_ids:
        if len(selected) >= num_samples:
            break
        choice = rng.choice(by_video[video_id])
        key = (choice.video_id, choice.frame_index, str(choice.overlay_path))
        selected.append(choice)
        used_keys.add(key)

    if len(selected) < num_samples:
        leftovers = []
        for item in candidates:
            key = (item.video_id, item.frame_index, str(item.overlay_path))
            if key not in used_keys:
                leftovers.append(item)
        rng.shuffle(leftovers)
        for item in leftovers:
            if len(selected) >= num_samples:
                break
            selected.append(item)

    return selected[:num_samples]


def resolve_video_path(videos_dir: Path, video_name: str, video_id: str) -> Optional[Path]:
    exact = videos_dir / video_name
    if exact.exists():
        return exact

    matches = sorted(videos_dir.glob(f"{video_id}.*"))
    if matches:
        return matches[0]
    return None


def extract_frame(video_path: Path, frame_index: int) -> Optional[object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def write_readme(readme_path: Path) -> None:
    content = """# Manual Validation Set (ImageJ)

## Folder Layout
- `images_raw/`: original frames for manual annotation
- `sam_overlay/`: YOLOv11 + SAM overlay result for reference
- `manual_labels/`: save your ground-truth binary masks here
- `manifest.csv`: mapping table
- `prediction_summary.csv`: model-predicted stats from metadata

## Naming Rule
For each sample, save mask as:
- `manual_labels/<sample_id>_mask.png`

Example:
- raw image: `images_raw/S001_video_xxx_f00053.png`
- label mask: `manual_labels/S001_video_xxx_f00053_mask.png`

## ImageJ Suggested Steps
1. Open one image in `images_raw/`.
2. Segment target region(s) manually.
3. Export a binary mask (white=target, black=background).
4. Save using the exact naming rule above.

## Notes
- Keep mask size exactly the same as the raw image.
- Use PNG format to avoid compression artifacts.
"""
    readme_path.write_text(content, encoding="utf-8")


def build_package(
    selected: List[Candidate],
    videos_dir: Path,
    out_dir: Path,
) -> None:
    raw_dir = out_dir / "images_raw"
    overlay_dir = out_dir / "sam_overlay"
    labels_dir = out_dir / "manual_labels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    pred_rows: List[Dict[str, str]] = []

    for idx, item in enumerate(selected, start=1):
        sample_id = f"S{idx:03d}_{item.video_id}_f{item.frame_index:05d}"
        image_name = f"{sample_id}.png"
        label_name = f"{sample_id}_mask.png"

        dst_overlay = overlay_dir / image_name
        shutil.copy2(item.overlay_path, dst_overlay)

        video_path = resolve_video_path(videos_dir, item.video_name, item.video_id)
        if video_path is None:
            raise FileNotFoundError(f"Missing source video for {item.video_name}")

        frame = extract_frame(video_path, item.frame_index)
        if frame is None:
            raise RuntimeError(f"Failed to extract frame {item.frame_index} from {video_path}")
        dst_raw = raw_dir / image_name
        cv2.imwrite(str(dst_raw), frame)

        class_names = sorted({str(det.get("class_name", "")) for det in item.detections if det})
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "video_id": item.video_id,
                "video_name": item.video_name,
                "frame_index": str(item.frame_index),
                "raw_image": str(dst_raw.relative_to(out_dir)),
                "sam_overlay_image": str(dst_overlay.relative_to(out_dir)),
                "manual_label_mask": str((labels_dir / label_name).relative_to(out_dir)),
                "num_detections": str(len(item.detections)),
                "detected_classes": ";".join(class_names),
            }
        )

        if item.detections:
            for det_i, det in enumerate(item.detections, start=1):
                bbox = det.get("bbox_xyxy")
                pred_rows.append(
                    {
                        "sample_id": sample_id,
                        "detection_index": str(det_i),
                        "class_name": str(det.get("class_name", "")),
                        "confidence": str(det.get("confidence", "")),
                        "mask_area_px": str(det.get("mask_area", "")),
                        "mask_width_px": str(det.get("mask_width_px", "")),
                        "bbox_xyxy": json.dumps(bbox) if bbox is not None else "",
                    }
                )
        else:
            pred_rows.append(
                {
                    "sample_id": sample_id,
                    "detection_index": "",
                    "class_name": "",
                    "confidence": "",
                    "mask_area_px": "",
                    "mask_width_px": "",
                    "bbox_xyxy": "",
                }
            )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    pred_path = out_dir / "prediction_summary.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    write_readme(out_dir / "README.md")


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    videos_dir = Path(args.videos_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs-dir not found: {outputs_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos-dir not found: {videos_dir}")
    if args.num_samples <= 0:
        raise ValueError("num-samples must be > 0")

    candidates = collect_candidates(outputs_dir)
    if len(candidates) < args.num_samples:
        raise RuntimeError(
            f"Not enough candidates: requested {args.num_samples}, found {len(candidates)}"
        )

    selected = select_candidates(candidates, args.num_samples, args.seed)
    if len(selected) < args.num_samples:
        raise RuntimeError(
            f"Unable to select enough samples: requested {args.num_samples}, got {len(selected)}"
        )

    build_package(selected, videos_dir=videos_dir, out_dir=out_dir)
    print(f"Prepared manual validation set: {out_dir}")
    print(f"Samples: {len(selected)}")


if __name__ == "__main__":
    main()
