"""Shared paths and small image helpers for the dual-camera snapshot inference.

The old single-camera video + YOLO/SAM2 pipeline that used to live here was removed (unused since the
dual-camera, one-photo-per-camera flow). The pre-cleanup file is kept in backup_before_code_review_fixes_2026-11-26.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def is_frozen_app() -> bool:
    """判斷目前是否在 PyInstaller exe 中執行。"""
    return bool(getattr(sys, "frozen", False))


BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(sys.executable).resolve().parent if is_frozen_app() else BASE_DIR
DATA_DIR = APP_DIR / "data" if is_frozen_app() else BASE_DIR / "data"
OUTPUTS_DIR = DATA_DIR / "sam2_inference_outputs"  # folder name kept so existing results stay readable


def open_camera(camera_index: int):
    """Open a camera with the native backend for the current operating system."""
    import cv2

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(camera_index), backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_ANY)
    return cap


def _mask_sides_px(mask) -> tuple[Optional[float], Optional[float]]:
    """Compute short and long sides of a mask using its minimum-area rectangle."""
    import cv2
    import numpy as np

    mask_bool = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None, None
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    if len(points) < 3:
        return 1.0, 1.0
    (_, _), (width, height), _ = cv2.minAreaRect(points)
    short_side = min(float(width), float(height))
    long_side = max(float(width), float(height))
    return (
        short_side if short_side > 0 else None,
        long_side if long_side > 0 else None,
    )
