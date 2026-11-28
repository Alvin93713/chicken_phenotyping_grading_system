import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aruco_calibration as ac  # noqa: E402


def scene(marker_id=0, side=200, angle=0.0, size=(720, 1280), at=(500, 300), blur=True, noise=6, seed=0,
          squash=1.0, quiet=None):
    """A grey camera-like frame with one printed marker (white quiet zone) of `side` px, rotated by `angle`
    degrees; squash < 1 shortens it vertically (a tilted card)."""
    rng = np.random.default_rng(seed)
    frame = rng.normal(120, 18, size).clip(0, 255).astype(np.uint8)
    big = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ac.DICTIONARY), marker_id, 600)
    quiet = quiet if quiet is not None else 100
    card = cv2.copyMakeBorder(big, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255)
    scale = side / 600
    h = card.shape[0]
    center = (h / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    m[1] *= squash
    m[:, 2] += (at[0] - h / 2 + h * (1 - scale) / 2 * 0, at[1] - h / 2)
    warped = cv2.warpAffine(card, m, (size[1], size[0]), flags=cv2.INTER_AREA, borderValue=0)
    mask = cv2.warpAffine(np.full_like(card, 255), m, (size[1], size[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    frame = np.where(mask > 0, warped, frame).astype(np.uint8)
    if blur:
        frame = cv2.GaussianBlur(frame, (3, 3), 0.8)
    if noise:
        frame = (frame.astype(np.int16) + rng.normal(0, noise, size).round().astype(np.int16)).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


class ArucoCalibrationTests(unittest.TestCase):
    def test_scale_is_accurate_for_sizes_and_rotations(self):
        for side in (80, 150, 300):
            for angle in (0, 17, 45):
                result = ac.detect(scene(side=side, angle=angle), 0, 4.0)
                self.assertTrue(result["ok"], (side, angle, result))
                error = abs(result["side_px"] / side - 1)
                self.assertLess(error, 0.01, (side, angle, result["side_px"]))  # under 1 % (clicking: 1–3 %)
                self.assertLess(abs(result["cm_per_pixel"] * result["side_px"] / 4.0 - 1), 1e-4)

    def test_each_camera_uses_its_own_marker(self):
        result = ac.detect(scene(marker_id=1), 0, 4.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], ac.WRONG_MARKER)
        self.assertTrue(ac.detect(scene(marker_id=1), ac.MARKER_IDS["shank"], 4.0)["ok"])

    def test_rejections(self):
        empty = np.full((720, 1280, 3), 120, np.uint8)
        self.assertEqual(ac.detect(empty, 0, 4.0)["error"], ac.NOT_FOUND)
        small = ac.detect(scene(side=30, blur=False, noise=0), 0, 4.0)
        self.assertEqual(small["error"], ac.TOO_SMALL)
        tilted = ac.detect(scene(side=200, squash=0.75), 0, 4.0)
        self.assertEqual(tilted["error"], ac.TOO_TILTED)
        with self.assertRaises(ValueError):
            ac.detect(empty, 0, 0)

    def test_annotate_keeps_the_photo_size(self):
        frame = scene()
        out = ac.annotate(frame, ac.detect(frame, 0, 4.0))
        self.assertEqual(out.shape, frame.shape)
        self.assertFalse(np.array_equal(out, frame))


if __name__ == "__main__":
    unittest.main()
