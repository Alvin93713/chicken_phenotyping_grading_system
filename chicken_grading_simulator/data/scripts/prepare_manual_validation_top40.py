#!/usr/bin/env python3
"""
Prepare a manual-label package with top-N largest Comb and top-N largest Shank frames.

Selection source:
- metadata in results/outputs_best_videos (all detected frames, not only sample frames)

Output:
- images_raw/
- sam_overlay/ (only when a matching sample overlay exists)
- manual_labels/
- manifest.csv
- manifest_for_labelers.csv
- prediction_summary.csv
- LABELER_GUIDE_ZH.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from paths import data_best_videos_dir, outputs_best_videos_dir, validation_manual_dir

FRAME_RE = re.compile(r"frame_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
TARGET_CLASSES = ("Comb", "Shank")


@dataclass
class ClassEntry:
    class_name: str
    size_value: float
    video_name: str
    video_id: str
    frame_index: int
    detections: List[dict]
    overlay_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=str(outputs_best_videos_dir()))
    parser.add_argument("--videos-dir", default=str(data_best_videos_dir()))
    parser.add_argument("--out-dir", default=str(validation_manual_dir()))
    parser.add_argument("--comb-top", type=int, default=20)
    parser.add_argument("--shank-top", type=int, default=20)
    parser.add_argument(
        "--max-per-video",
        type=int,
        default=4,
        help="Maximum selected frames per video for each class group.",
    )
    parser.add_argument(
        "--min-frame-gap",
        type=int,
        default=10,
        help="Minimum frame-index gap for selections from the same video.",
    )
    parser.add_argument(
        "--overlay-demo-per-class",
        type=int,
        default=4,
        help="Number of demo overlay images to export for each class (Comb/Shank).",
    )
    parser.add_argument(
        "--overlay-demo-max-per-video",
        type=int,
        default=2,
        help="Maximum demo overlay images from the same video for each class.",
    )
    parser.add_argument(
        "--overlay-demo-min-frame-gap",
        type=int,
        default=10,
        help="Minimum frame gap between demo overlays from the same video.",
    )
    parser.add_argument("--size-key", default="mask_area", help="Detection key used to rank size.")
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow Comb/Shank selected frames to overlap. Default is no overlap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove out-dir before writing new files.",
    )
    return parser.parse_args()


def parse_frame_index(path: Path) -> Optional[int]:
    m = FRAME_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def resolve_overlay_path(raw_path: str, meta_parent: Path, outputs_dir: Path) -> Optional[Path]:
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p.resolve()

    candidates = [
        meta_parent / p,
        outputs_dir / p,
        outputs_dir.parent / p,
        Path.cwd() / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def as_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def collect_entries(outputs_dir: Path, size_key: str) -> Dict[str, List[ClassEntry]]:
    entries: Dict[str, List[ClassEntry]] = {"Comb": [], "Shank": []}
    metadata_paths = sorted(outputs_dir.glob("*/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata.json found under {outputs_dir}")

    for meta_path in metadata_paths:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = str(data.get("video") or f"{meta_path.parent.name}.mp4")
        video_id = Path(video_name).stem

        overlay_map: Dict[int, Path] = {}
        for sample in data.get("sample_frames", []):
            frame_idx = parse_frame_index(Path(str(sample)))
            if frame_idx is None:
                continue
            overlay = resolve_overlay_path(str(sample), meta_path.parent, outputs_dir)
            if overlay is not None:
                overlay_map[frame_idx] = overlay

        for frame_record in data.get("frames", []):
            frame_index = frame_record.get("frame_index")
            if not isinstance(frame_index, int):
                continue

            detections = frame_record.get("detections", [])
            if not detections:
                continue

            for cls_name in TARGET_CLASSES:
                cls_dets = [d for d in detections if str(d.get("class_name", "")) == cls_name]
                if not cls_dets:
                    continue
                best_det = max(cls_dets, key=lambda d: as_float(d.get(size_key)))
                entries[cls_name].append(
                    ClassEntry(
                        class_name=cls_name,
                        size_value=as_float(best_det.get(size_key)),
                        video_name=video_name,
                        video_id=video_id,
                        frame_index=frame_index,
                        detections=detections,
                        overlay_path=overlay_map.get(frame_index),
                    )
                )

    for cls_name in TARGET_CLASSES:
        entries[cls_name].sort(
            key=lambda x: (-x.size_value, x.video_id, x.frame_index),
        )
    return entries


def pick_top(
    items: List[ClassEntry],
    top_k: int,
    used_frame_keys: set,
    used_frame_indices_by_video: Dict[str, List[int]],
    allow_overlap: bool,
    max_per_video: int,
    min_frame_gap: int,
    existing_group_counts: Optional[Counter] = None,
) -> Tuple[List[ClassEntry], set, Dict[str, List[int]]]:
    selected: List[ClassEntry] = []
    new_used = set(used_frame_keys)
    selected_keys = set()
    selected_per_video: Counter = Counter(existing_group_counts or {})
    selected_frame_indices: Dict[str, List[int]] = {
        k: list(v) for k, v in used_frame_indices_by_video.items()
    }

    def is_far_enough(video_id: str, frame_index: int) -> bool:
        if min_frame_gap <= 0:
            return True
        prev = selected_frame_indices.get(video_id, [])
        for idx in prev:
            if abs(frame_index - idx) < min_frame_gap:
                return False
        return True

    for item in items:
        if len(selected) >= top_k:
            break
        frame_key = (item.video_id, item.frame_index)
        if selected_per_video[item.video_id] >= max_per_video:
            continue
        if frame_key in selected_keys:
            continue
        if not allow_overlap and frame_key in used_frame_keys:
            continue
        if not is_far_enough(item.video_id, item.frame_index):
            continue
        selected.append(item)
        selected_keys.add(frame_key)
        new_used.add(frame_key)
        selected_per_video[item.video_id] += 1
        selected_frame_indices.setdefault(item.video_id, []).append(item.frame_index)
    return selected, new_used, selected_frame_indices


def max_selectable_with_cap(
    items: List[ClassEntry],
    max_per_video: int,
    min_frame_gap: int,
) -> int:
    per_video_indices: Dict[str, set] = defaultdict(set)
    for item in items:
        per_video_indices[item.video_id].add(item.frame_index)

    total = 0
    for indices in per_video_indices.values():
        sorted_idx = sorted(indices)
        take = 0
        last = None
        for idx in sorted_idx:
            if last is None or min_frame_gap <= 0 or (idx - last) >= min_frame_gap:
                take += 1
                last = idx
        total += min(take, max_per_video)
    return total


def resolve_video_path(videos_dir: Path, video_name: str, video_id: str) -> Optional[Path]:
    exact = videos_dir / video_name
    if exact.exists():
        return exact
    matches = sorted(videos_dir.glob(f"{video_id}.*"))
    if matches:
        return matches[0]
    return None


def extract_frame(video_path: Path, frame_index: int):
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


def write_labeler_guide(path: Path) -> None:
    content = """# 標註說明（給標註者）

## 你會用到的檔案
- `images_raw/`：待標註原圖
- `sam_overlay/`：示範圖（Comb 4 + Shank 4，只做參考）
- `manifest_for_labelers.csv`：每張圖對應的輸出檔名
- `manual_labels/`：請把標註結果存到這裡

## 標註工具
- 建議使用 ImageJ / Fiji。

## 標註目標
- 目標區域標成白色（255）。
- 背景保持黑色（0）。
- 產出二值遮罩（binary mask）。

## 每張圖流程
1. 開啟 `images_raw/` 內影像。
2. 手動圈選目標區域。
3. 匯出二值 mask。
4. 檔名必須與 `manifest_for_labelers.csv` 的 `manual_label_mask` 完全一致。

## 必須遵守
- 輸出 mask 尺寸要和原圖完全一樣。
- 請用 PNG。
- 檔名大小寫不可更改。
"""
    path.write_text(content, encoding="utf-8")


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write for manifest.")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pick_demo_overlays(
    items: List[ClassEntry],
    top_k: int,
    max_per_video: int,
    min_frame_gap: int,
) -> List[ClassEntry]:
    candidates = [x for x in items if x.overlay_path is not None and x.overlay_path.exists()]
    selected: List[ClassEntry] = []
    used_keys = set()
    per_video = Counter()
    per_video_indices: Dict[str, List[int]] = defaultdict(list)

    def can_take(item: ClassEntry, relax_video_cap: bool, relax_gap: bool) -> bool:
        key = (item.video_id, item.frame_index)
        if key in used_keys:
            return False
        if not relax_video_cap and per_video[item.video_id] >= max_per_video:
            return False
        if not relax_gap and min_frame_gap > 0:
            for idx in per_video_indices[item.video_id]:
                if abs(item.frame_index - idx) < min_frame_gap:
                    return False
        return True

    def try_pick(relax_video_cap: bool, relax_gap: bool) -> None:
        for item in candidates:
            if len(selected) >= top_k:
                break
            if not can_take(item, relax_video_cap=relax_video_cap, relax_gap=relax_gap):
                continue
            selected.append(item)
            used_keys.add((item.video_id, item.frame_index))
            per_video[item.video_id] += 1
            per_video_indices[item.video_id].append(item.frame_index)

    # strict -> relax frame gap -> relax both constraints
    try_pick(relax_video_cap=False, relax_gap=False)
    if len(selected) < top_k:
        try_pick(relax_video_cap=False, relax_gap=True)
    if len(selected) < top_k:
        try_pick(relax_video_cap=True, relax_gap=True)
    return selected[:top_k]


def build_overlay_demos(
    entries_by_class: Dict[str, List[ClassEntry]],
    out_dir: Path,
    per_class: int,
    max_per_video: int,
    min_frame_gap: int,
) -> None:
    overlay_dir = out_dir / "sam_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for cls_name in TARGET_CLASSES:
        picked = pick_demo_overlays(
            entries_by_class[cls_name],
            top_k=per_class,
            max_per_video=max_per_video,
            min_frame_gap=min_frame_gap,
        )
        if len(picked) < per_class:
            raise RuntimeError(
                f"Not enough overlay demos for {cls_name}. "
                f"Need {per_class}, got {len(picked)}."
            )

        cls_tag = cls_name.upper()
        for i, item in enumerate(picked, start=1):
            demo_name = f"DEMO_{cls_tag}_{i:03d}_{item.video_id}_f{item.frame_index:05d}.png"
            dst = overlay_dir / demo_name
            shutil.copy2(item.overlay_path, dst)  # type: ignore[arg-type]
            rows.append(
                {
                    "demo_id": f"DEMO_{cls_tag}_{i:03d}",
                    "class_name": cls_name,
                    "video_id": item.video_id,
                    "video_name": item.video_name,
                    "frame_index": str(item.frame_index),
                    "size_metric": "mask_area",
                    "size_value": f"{item.size_value:.2f}",
                    "overlay_image": str((Path("sam_overlay") / demo_name).as_posix()),
                }
            )

    write_manifest(out_dir / "sam_overlay_manifest.csv", rows)


def build_package(
    selected_comb: List[ClassEntry],
    selected_shank: List[ClassEntry],
    entries_by_class: Dict[str, List[ClassEntry]],
    videos_dir: Path,
    out_dir: Path,
    overlay_demo_per_class: int,
    overlay_demo_max_per_video: int,
    overlay_demo_min_frame_gap: int,
) -> None:
    raw_dir = out_dir / "images_raw"
    overlay_dir = out_dir / "sam_overlay"
    labels_dir = out_dir / "manual_labels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    pred_rows: List[Dict[str, str]] = []

    def add_rows(group: str, rows: List[ClassEntry]) -> None:
        for i, item in enumerate(rows, start=1):
            prefix = "COMB" if group == "Comb" else "SHANK"
            sample_id = f"{prefix}_{i:03d}_{item.video_id}_f{item.frame_index:05d}"
            img_name = f"{sample_id}.png"
            label_name = f"{sample_id}_mask.png"

            video_path = resolve_video_path(videos_dir, item.video_name, item.video_id)
            if video_path is None:
                raise FileNotFoundError(f"Missing source video: {item.video_name}")
            frame = extract_frame(video_path, item.frame_index)
            if frame is None:
                raise RuntimeError(
                    f"Failed to extract frame {item.frame_index} from {video_path}"
                )

            raw_path = raw_dir / img_name
            cv2.imwrite(str(raw_path), frame)

            overlay_rel = ""

            class_names = sorted(
                {str(det.get("class_name", "")) for det in item.detections if det}
            )
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "group": group,
                    "rank": str(i),
                    "size_metric": "mask_area",
                    "size_value": f"{item.size_value:.2f}",
                    "video_id": item.video_id,
                    "video_name": item.video_name,
                    "frame_index": str(item.frame_index),
                    "raw_image": str(raw_path.relative_to(out_dir)),
                    "sam_overlay_image": overlay_rel,
                    "manual_label_mask": str((labels_dir / label_name).relative_to(out_dir)),
                    "num_detections": str(len(item.detections)),
                    "detected_classes": ";".join(class_names),
                }
            )

            for j, det in enumerate(item.detections, start=1):
                bbox = det.get("bbox_xyxy")
                pred_rows.append(
                    {
                        "sample_id": sample_id,
                        "group": group,
                        "detection_index": str(j),
                        "class_name": str(det.get("class_name", "")),
                        "confidence": str(det.get("confidence", "")),
                        "mask_area_px": str(det.get("mask_area", "")),
                        "mask_width_px": str(det.get("mask_width_px", "")),
                        "bbox_xyxy": json.dumps(bbox) if bbox is not None else "",
                    }
                )

    add_rows("Comb", selected_comb)
    add_rows("Shank", selected_shank)

    write_manifest(out_dir / "manifest.csv", manifest_rows)
    write_manifest(out_dir / "prediction_summary.csv", pred_rows)

    label_rows = [
        {
            "sample_id": r["sample_id"],
            "group": r["group"],
            "raw_image": r["raw_image"],
            "manual_label_mask": r["manual_label_mask"],
        }
        for r in manifest_rows
    ]
    write_manifest(out_dir / "manifest_for_labelers.csv", label_rows)
    write_labeler_guide(out_dir / "LABELER_GUIDE_ZH.md")
    build_overlay_demos(
        entries_by_class=entries_by_class,
        out_dir=out_dir,
        per_class=overlay_demo_per_class,
        max_per_video=overlay_demo_max_per_video,
        min_frame_gap=overlay_demo_min_frame_gap,
    )


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    videos_dir = Path(args.videos_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs-dir not found: {outputs_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos-dir not found: {videos_dir}")
    if args.comb_top <= 0 or args.shank_top <= 0:
        raise ValueError("comb-top and shank-top must be > 0")
    if args.max_per_video <= 0:
        raise ValueError("max-per-video must be > 0")
    if args.min_frame_gap < 0:
        raise ValueError("min-frame-gap must be >= 0")
    if args.overlay_demo_per_class <= 0:
        raise ValueError("overlay-demo-per-class must be > 0")
    if args.overlay_demo_max_per_video <= 0:
        raise ValueError("overlay-demo-max-per-video must be > 0")
    if args.overlay_demo_min_frame_gap < 0:
        raise ValueError("overlay-demo-min-frame-gap must be >= 0")

    entries = collect_entries(outputs_dir=outputs_dir, size_key=args.size_key)
    if len(entries["Comb"]) < args.comb_top:
        raise RuntimeError(f"Not enough Comb entries: need {args.comb_top}, got {len(entries['Comb'])}")
    if len(entries["Shank"]) < args.shank_top:
        raise RuntimeError(
            f"Not enough Shank entries: need {args.shank_top}, got {len(entries['Shank'])}"
        )

    comb_cap = max_selectable_with_cap(entries["Comb"], args.max_per_video, args.min_frame_gap)
    shank_cap = max_selectable_with_cap(entries["Shank"], args.max_per_video, args.min_frame_gap)
    if comb_cap < args.comb_top:
        raise RuntimeError(
            f"Comb cannot satisfy comb-top={args.comb_top} with "
            f"max-per-video={args.max_per_video} and min-frame-gap={args.min_frame_gap}. "
            f"Max selectable={comb_cap}."
        )
    if shank_cap < args.shank_top:
        raise RuntimeError(
            f"Shank cannot satisfy shank-top={args.shank_top} with "
            f"max-per-video={args.max_per_video} and min-frame-gap={args.min_frame_gap}. "
            f"Max selectable={shank_cap}."
        )

    selected_comb, _, used_frame_indices = pick_top(
        entries["Comb"],
        top_k=args.comb_top,
        used_frame_keys=set(),
        used_frame_indices_by_video={},
        allow_overlap=False,
        max_per_video=args.max_per_video,
        min_frame_gap=args.min_frame_gap,
    )
    comb_frame_keys = {(x.video_id, x.frame_index) for x in selected_comb}
    shank_used_keys = set() if args.allow_overlap else comb_frame_keys
    selected_shank, _, used_after_shank = pick_top(
        entries["Shank"],
        top_k=args.shank_top,
        used_frame_keys=shank_used_keys,
        used_frame_indices_by_video=used_frame_indices,
        allow_overlap=args.allow_overlap,
        max_per_video=args.max_per_video,
        min_frame_gap=args.min_frame_gap,
    )

    if len(selected_shank) < args.shank_top and not args.allow_overlap:
        remaining = args.shank_top - len(selected_shank)
        shank_used_keys = {(x.video_id, x.frame_index) for x in selected_shank}
        shank_counts = Counter(x.video_id for x in selected_shank)
        extra, _, _ = pick_top(
            entries["Shank"],
            top_k=remaining,
            used_frame_keys=shank_used_keys,
            used_frame_indices_by_video=used_after_shank,
            allow_overlap=True,
            max_per_video=args.max_per_video,
            min_frame_gap=args.min_frame_gap,
            existing_group_counts=shank_counts,
        )
        selected_shank.extend(extra[:remaining])

    if len(selected_comb) < args.comb_top or len(selected_shank) < args.shank_top:
        raise RuntimeError(
            f"Selection failed: Comb={len(selected_comb)}, Shank={len(selected_shank)}"
        )

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_package(
        selected_comb=selected_comb,
        selected_shank=selected_shank,
        entries_by_class=entries,
        videos_dir=videos_dir,
        out_dir=out_dir,
        overlay_demo_per_class=args.overlay_demo_per_class,
        overlay_demo_max_per_video=args.overlay_demo_max_per_video,
        overlay_demo_min_frame_gap=args.overlay_demo_min_frame_gap,
    )

    comb_unique = len({(x.video_id, x.frame_index) for x in selected_comb})
    shank_unique = len({(x.video_id, x.frame_index) for x in selected_shank})
    overlap = len(
        {(x.video_id, x.frame_index) for x in selected_comb}
        & {(x.video_id, x.frame_index) for x in selected_shank}
    )

    print(f"Prepared: {out_dir}")
    print(f"Comb selected: {len(selected_comb)} (unique frames: {comb_unique})")
    print(f"Shank selected: {len(selected_shank)} (unique frames: {shank_unique})")
    print(f"Comb/Shank frame overlap: {overlap}")
    comb_video_max = max(Counter(x.video_id for x in selected_comb).values(), default=0)
    shank_video_max = max(Counter(x.video_id for x in selected_shank).values(), default=0)
    print(f"Comb max frames in one video: {comb_video_max}")
    print(f"Shank max frames in one video: {shank_video_max}")


if __name__ == "__main__":
    main()
