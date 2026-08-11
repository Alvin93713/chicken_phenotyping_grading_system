#!/usr/bin/env python3
"""Render frame-synced Comb mask/bbox and Comb-size chart videos."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/nas2/Workspace/Hank")
DEFAULT_VIDEO_ID = "video_20250908_083233"
CM_PER_PX_SAM2 = 0.02
CM_PER_PX_GAIT = 0.0406


@dataclass
class CombMask:
    frame_index: int
    bbox: np.ndarray
    coords_yx: np.ndarray
    area_px2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument(
        "--source-video",
        type=Path,
        default=None,
        help="Defaults to SAM2/data/videos/Best_Videos/<video_id>.mp4.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Defaults to SAM2/results/outputs_best_videos/<video_id>/metadata.json.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Defaults to SAM2/results/outputs_best_videos/comb_color/mask_pixels/<video_id>.",
    )
    parser.add_argument(
        "--manual-labels",
        type=Path,
        default=None,
        help="Manual polygon label JSON from manual_comb_polygon_label_tool.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to SAM2/results/outputs_best_videos/analysis/comb_size_timeseries.",
    )
    parser.add_argument("--fps-scale", type=float, default=0.4)
    parser.add_argument("--plot-width", type=int, default=1440)
    parser.add_argument("--plot-height", type=int, default=720)
    parser.add_argument("--video-height", type=int, default=900)
    parser.add_argument("--mask-alpha", type=float, default=0.42)
    parser.add_argument(
        "--p98-override-cm2",
        type=float,
        default=None,
        help="Draw the P98 reference line at this cm2 value instead of computing it.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional filename tag inserted after video_id, e.g. 'demo'.",
    )
    parser.add_argument(
        "--crop-to-mask-interval",
        action="store_true",
        help="Render only from the first to last frame that has a Comb mask.",
    )
    return parser.parse_args()


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    source_video = args.source_video or ROOT / "SAM2/data/videos/Best_Videos" / f"{args.video_id}.mp4"
    metadata = (
        args.metadata
        or ROOT / "SAM2/results/outputs_best_videos" / args.video_id / "metadata.json"
    )
    mask_dir = (
        args.mask_dir
        or ROOT
        / "SAM2/results/outputs_best_videos/comb_color/mask_pixels"
        / args.video_id
    )
    output_dir = (
        args.output_dir
        or ROOT / "SAM2/results/outputs_best_videos/analysis/comb_size_timeseries"
    )
    return source_video, metadata, mask_dir, output_dir


def load_masks(mask_dir: Path) -> dict[int, CombMask]:
    masks: dict[int, CombMask] = {}
    for path in sorted(mask_dir.glob("frame_*_det_*.npz")):
        data = np.load(path)
        frame_index = int(data["frame_index"])
        coords = data["coords"].astype(np.int32)
        area = int(coords.shape[0])
        mask = CombMask(
            frame_index=frame_index,
            bbox=data["bbox"].astype(float),
            coords_yx=coords,
            area_px2=area,
        )
        existing = masks.get(frame_index)
        if existing is None or mask.area_px2 > existing.area_px2:
            masks[frame_index] = mask
    return masks


def load_manual_label_masks(labels_path: Path) -> tuple[dict[int, CombMask], set[int]]:
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    width = int(data["width"])
    height = int(data["height"])
    masks: dict[int, CombMask] = {}
    no_comb_frames: set[int] = set()
    for key, label in data.get("labels", {}).items():
        frame_index = int(label.get("frame_index", key))
        if label.get("status") == "no_comb":
            no_comb_frames.add(frame_index)
            continue
        if label.get("status") != "labeled":
            continue
        polygon = label.get("polygon") or []
        if len(polygon) < 3:
            continue
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        draw.polygon([(float(x), float(y)) for x, y in polygon], fill=255)
        mask_array = np.asarray(mask_image) > 0
        coords = np.column_stack(np.where(mask_array)).astype(np.int32)
        if coords.size == 0:
            continue
        bbox = label.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            y = coords[:, 0]
            x = coords[:, 1]
            bbox = [float(x.min()), float(y.min()), float(x.max()), float(y.max())]
        masks[frame_index] = CombMask(
            frame_index=frame_index,
            bbox=np.asarray(bbox, dtype=float),
            coords_yx=coords,
            area_px2=int(coords.shape[0]),
        )
    return masks, no_comb_frames


def write_series_csv(
    output_path: Path,
    video_id: str,
    fps: float,
    frame_indices: list[int],
    masks: dict[int, CombMask],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "frame_index",
        "time_sec",
        "comb_detected",
        "comb_area_px2",
        "comb_area_cm2_sam2_0p02",
        "comb_area_cm2_gait_0p0406",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in frame_indices:
            mask = masks.get(frame_index)
            if mask is None:
                writer.writerow(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "time_sec": frame_index / fps,
                        "comb_detected": 0,
                    }
                )
                continue
            x1, y1, x2, y2 = mask.bbox
            writer.writerow(
                {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                    "comb_detected": 1,
                    "comb_area_px2": mask.area_px2,
                    "comb_area_cm2_sam2_0p02": mask.area_px2 * CM_PER_PX_SAM2**2,
                    "comb_area_cm2_gait_0p0406": mask.area_px2 * CM_PER_PX_GAIT**2,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                }
            )


def contain_resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(frame)
    scale = min(width / image.width, height / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    canvas.paste(image, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
    return np.asarray(canvas)


def draw_mask_and_bbox(frame: np.ndarray, mask: CombMask | None, frame_index: int, alpha: float) -> np.ndarray:
    out = frame.copy()
    if mask is not None:
        coords = mask.coords_yx
        y = coords[:, 0]
        x = coords[:, 1]
        valid = (y >= 0) & (y < out.shape[0]) & (x >= 0) & (x < out.shape[1])
        y = y[valid]
        x = x[valid]
        color = np.array([255, 0, 0], dtype=np.float32)
        out[y, x] = (out[y, x].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)

    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    if mask is not None:
        x1, y1, x2, y2 = [float(v) for v in mask.bbox]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=7)
    return np.asarray(image)


def make_chart_renderer(
    video_id: str,
    frame_indices: list[int],
    masks: dict[int, CombMask],
    width: int,
    height: int,
    p98_override_cm2: float | None = None,
) -> tuple[plt.Figure, FigureCanvasAgg, callable]:
    dpi = 120
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    frames = np.asarray(frame_indices, dtype=int)
    area_cm2 = np.full(len(frames), np.nan)
    for i, frame_index in enumerate(frames):
        mask = masks.get(int(frame_index))
        if mask is not None:
            area_cm2[i] = mask.area_px2 * CM_PER_PX_SAM2**2

    detected = np.isfinite(area_cm2)
    p98 = (
        float(p98_override_cm2)
        if p98_override_cm2 is not None
        else float(np.nanpercentile(area_cm2, 98)) if detected.any() else 0.0
    )
    y_max = max(5.0, float(np.nanmax(area_cm2)) * 1.12 if detected.any() else 5.0)
    ax.set_xlim(float(frames.min()), float(frames.max()))
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Comb mask area (cm2)")
    ax.set_title(f"{video_id} Comb size")
    ax.grid(True, alpha=0.22)

    area_line, = ax.plot([], [], color="#d43f3a", linewidth=2.5, marker="o", markersize=4, label="Comb size")
    p98_line = ax.axhline(p98, color="#333333", linestyle="--", linewidth=1.8, label=f"P98 {p98:.2f} cm2")
    current_dot, = ax.plot([], [], marker="o", markersize=9, color="#1f77b4", label="Current frame")
    current_vline = ax.axvline(float(frames[0]), color="#1f77b4", linewidth=1.4, alpha=0.55)
    frame_text = ax.text(
        0.985,
        0.06,
        "",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=16,
        color="#2d2d2d",
        bbox={"facecolor": "white", "edgecolor": "#cfcfcf", "alpha": 0.92, "pad": 6},
    )
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()

    def render(i: int) -> np.ndarray:
        upto = i + 1
        visible = np.isfinite(area_cm2[:upto])
        area_line.set_data(frames[:upto][visible], area_cm2[:upto][visible])
        frame_index = int(frames[i])
        current_vline.set_xdata([frame_index, frame_index])
        if np.isfinite(area_cm2[i]):
            current_dot.set_data([frame_index], [area_cm2[i]])
            frame_text.set_text(f"Frame {frame_index} | {area_cm2[i]:.2f} cm2")
        else:
            current_dot.set_data([], [])
            frame_text.set_text(f"Frame {frame_index} | no Comb")
        p98_line.set_ydata([p98, p98])
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    return fig, canvas, render


def main() -> int:
    args = parse_args()
    source_video, metadata_path, mask_dir, output_dir = paths(args)
    data = json.loads(metadata_path.read_text())
    total_frames = int(data.get("total_frames", data.get("frame_count")))
    source_fps = float(data["fps"])
    output_fps = source_fps * args.fps_scale
    masks = load_masks(mask_dir)
    if args.manual_labels:
        manual_masks, manual_no_comb = load_manual_label_masks(args.manual_labels)
        for frame_index in manual_no_comb:
            masks.pop(frame_index, None)
        masks.update(manual_masks)
    if args.crop_to_mask_interval and masks:
        frame_indices = list(range(min(masks), max(masks) + 1))
    else:
        frame_indices = list(range(total_frames))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / f"{args.video_id}_comb_size_masknpz_per_frame.csv"
    output_prefix = args.video_id
    if args.output_tag:
        output_prefix = f"{args.video_id}_{args.output_tag}"
    overlay_output = output_dir / f"{output_prefix}_comb_bbox_mask_0p4x.mp4"
    chart_output = output_dir / f"{output_prefix}_comb_size_chart_0p4x.mp4"
    stacked_output = output_dir / f"{output_prefix}_comb_bbox_mask_stacked_chart_0p4x.mp4"

    write_series_csv(csv_output, args.video_id, source_fps, frame_indices, masks)
    fig, _, render_chart = make_chart_renderer(
        args.video_id,
        frame_indices,
        masks,
        args.plot_width,
        args.plot_height,
        args.p98_override_cm2,
    )

    reader = imageio.get_reader(source_video)
    with imageio.get_writer(overlay_output, fps=output_fps, codec="libx264", quality=9, macro_block_size=2) as overlay_writer, imageio.get_writer(
        chart_output, fps=output_fps, codec="libx264", quality=9, macro_block_size=2
    ) as chart_writer, imageio.get_writer(
        stacked_output, fps=output_fps, codec="libx264", quality=9, macro_block_size=2
    ) as stacked_writer:
        try:
            for i, frame_index in enumerate(frame_indices):
                source = reader.get_data(frame_index)
                overlay = draw_mask_and_bbox(source, masks.get(frame_index), frame_index, args.mask_alpha)
                overlay = contain_resize(overlay, args.plot_width, args.video_height)
                chart = render_chart(i)
                overlay_writer.append_data(overlay)
                chart_writer.append_data(chart)
                stacked_writer.append_data(np.vstack([overlay, chart]))
        finally:
            reader.close()
            plt.close(fig)

    print(csv_output)
    print(overlay_output)
    print(chart_output)
    print(stacked_output)
    print(f"frames={len(frame_indices)}, source_fps={source_fps}, output_fps={output_fps}")
    print(f"render_interval={frame_indices[0]}-{frame_indices[-1]}")
    print(f"mask_frames={len(masks)}, first={min(masks) if masks else ''}, last={max(masks) if masks else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
