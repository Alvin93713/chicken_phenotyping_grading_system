#!/usr/bin/env python3
"""Common project paths for the SAM2 workspace."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def _preferred_path(new_relative: str, old_relative: str | None = None) -> Path:
    new_path = ROOT_DIR / new_relative
    if new_path.exists():
        return new_path
    if old_relative is not None:
        old_path = ROOT_DIR / old_relative
        if old_path.exists():
            return old_path
    return new_path


def data_best_videos_dir() -> Path:
    return _preferred_path("data/videos/Best_Videos", "Best_Videos")


def data_good_videos_dir() -> Path:
    return _preferred_path("data/videos/Good_Videos_untils9_10", "Good_Videos_untils9_10")


def validation_manual_dir() -> Path:
    return _preferred_path("data/validation/manual_validation_20", "manual_validation_20")


def outputs_default_dir() -> Path:
    return _preferred_path("results/outputs", "outputs")


def outputs_best_videos_dir() -> Path:
    return _preferred_path("results/outputs_best_videos", "outputs_best_videos")


def outputs_good_videos_dir() -> Path:
    return _preferred_path("results/outputs_good_videos_untils9_10", "outputs_good_videos_untils9_10")


def local_yolo_weights_path() -> Path:
    return _preferred_path("resources/weights/best.pt", "best.pt")


def local_sam_weight_path(filename: str) -> Path:
    return _preferred_path(f"resources/weights/{filename}", filename)


def reference_sample_image_path() -> Path:
    return _preferred_path("resources/reference/sample.png", "sample.png")


def reference_ground_truth_csv_path() -> Path:
    return _preferred_path("resources/reference/Ground_truth.csv", "Ground_truth.csv")


def vendor_sam2_dir() -> Path:
    return _preferred_path("vendor/segment-anything-2", "segment-anything-2")
