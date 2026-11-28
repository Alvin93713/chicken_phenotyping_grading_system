"""ArUco calibration: find the printed marker (assets/aruco_calibration_markers.pdf) in a calibration photo and
turn its known side length into cm/pixel. No UI here, so it can be tested on synthetic images.

The comb camera looks for marker ID 0, the shank camera for ID 1 (dictionary DICT_4X4_50). The marker's side
length is the edge of its black outer frame, entered by the user (default 4 cm; the measured size when the
print was scaled)."""
from __future__ import annotations

import math

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_IDS = {"comb": 0, "shank": 1}
DEFAULT_MARKER_CM = 4.0
MIN_SIDE_PX = 40  # smaller markers give an imprecise scale (1 px ≈ 2.5 %)
MAX_SIDE_RATIO = 1.15  # longest / shortest side: above this the marker is too tilted for a reliable scale

# Reasons (Chinese source strings; the UI passes them through t()).
NOT_FOUND = "找不到 ID {marker_id} 標記，請確認標記在畫面中、沒有被遮住或反光"
WRONG_MARKER = "畫面中只找到 ID {found} 標記，這台相機應使用 ID {marker_id}"
SEVERAL = "畫面中有 {count} 個 ID {marker_id} 標記，請只留一個"
TOO_SMALL = "標記在畫面中太小（邊長 {side:.0f} px，至少需 {minimum} px），請改用較大的標記或靠近相機"
TOO_TILTED = "標記太斜（最長邊／最短邊 = {ratio:.2f}），請讓標記正對相機"


def _detector():
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # corners to a fraction of a pixel
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), parameters)


def detect(image_bgr, marker_id: int, marker_cm: float) -> dict:
    """Find `marker_id` and compute cm/pixel. Always returns a dict:
    ok, error (template) + error_values, marker_id, corners (4×[x, y]), side_px (mean of the 4 sides),
    side_ratio, cm_per_pixel (None unless ok), other_ids."""
    if marker_cm is None or not math.isfinite(float(marker_cm)) or float(marker_cm) <= 0:
        raise ValueError("標記邊長必須大於 0")
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)
    found = [] if ids is None else [int(i) for i in ids.flatten()]
    result = {"ok": False, "error": None, "error_values": {}, "marker_id": int(marker_id), "corners": None,
              "side_px": None, "side_ratio": None, "cm_per_pixel": None, "marker_cm": float(marker_cm),
              "other_ids": sorted({i for i in found if i != marker_id})}
    matches = [c[0] for c, i in zip(corners, found) if i == marker_id]
    if not matches:
        if result["other_ids"]:
            result.update(error=WRONG_MARKER, error_values={"found": "、".join(map(str, result["other_ids"])),
                                                            "marker_id": marker_id})
        else:
            result.update(error=NOT_FOUND, error_values={"marker_id": marker_id})
        return result
    if len(matches) > 1:
        result.update(error=SEVERAL, error_values={"count": len(matches), "marker_id": marker_id})
        return result
    points = np.asarray(matches[0], dtype=np.float64)
    sides = [float(np.linalg.norm(points[k] - points[(k + 1) % 4])) for k in range(4)]
    side_px = sum(sides) / 4
    ratio = max(sides) / min(sides)
    result.update(corners=points.round(3).tolist(), side_px=round(side_px, 3), side_ratio=round(ratio, 4))
    if side_px < MIN_SIDE_PX:
        result.update(error=TOO_SMALL, error_values={"side": side_px, "minimum": MIN_SIDE_PX})
        return result
    if ratio > MAX_SIDE_RATIO:
        result.update(error=TOO_TILTED, error_values={"ratio": ratio})
        return result
    result.update(ok=True, cm_per_pixel=float(marker_cm) / side_px)
    return result


def annotate(image_bgr, result: dict):
    """A copy of the photo with the found marker outlined (green = usable, red = rejected)."""
    image = image_bgr.copy()
    if result.get("corners"):
        points = np.asarray(result["corners"], dtype=np.float32)
        color = (60, 200, 60) if result.get("ok") else (40, 40, 230)
        thickness = max(2, round(max(image.shape[:2]) / 300))
        cv2.polylines(image, [points.round().astype(np.int32)], True, color, thickness, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(image, (round(float(x)), round(float(y))), thickness * 2, color, -1, cv2.LINE_AA)
    return image
