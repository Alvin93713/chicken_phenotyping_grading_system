#!/usr/bin/env python3
"""
Interactive pixel distance measurement tool.
Usage:
    ./measure_pixels.py /path/to/frame.png
Left-click two points to measure; press 'r' to reset, 'q' or ESC to quit.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import cv2

Point = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure pixel distance between two points.")
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument(
        "--real-distance-cm",
        type=float,
        default=None,
        help="If provided, also print cm-per-pixel using this real-world distance (cm).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    clone = image.copy()
    points: List[Point] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal points, image
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                image,
                f"({x}, {y})",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                distance = math.hypot(x2 - x1, y2 - y1)
                mid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.putText(
                    image,
                    f"{distance:.2f} px",
                    (mid[0] + 10, mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                print(f"Distance between points: {distance:.3f} pixels")
                if args.real_distance_cm is not None:
                    cm_per_px = args.real_distance_cm / distance if distance > 0 else float("inf")
                    print(f"Scale: 1 px = {cm_per_px:.6f} cm (real distance: {args.real_distance_cm} cm)")

    window = "Pixel Measurement"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    instructions = (
        "Left-click two points to measure. Press 'r' to reset, 'q' or ESC to exit."
    )
    print(instructions)

    while True:
        cv2.imshow(window, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("r"):
            image = clone.copy()
            points = []
            print("Reset measurement.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
