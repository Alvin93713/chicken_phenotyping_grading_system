import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dual_camera_ui
from scale_control import parse_scale_packets


class ScaleControlTests(unittest.TestCase):
    def test_packet_parser_uses_valid_grams_and_rejects_error_code(self):
        data = b"noise\x020100\x03\x022500\x03\x029999\x03"
        result = parse_scale_packets(data)
        self.assertEqual(result.samples, [100, 2500])
        self.assertAlmostEqual(result.average_g, 1300.0)
        self.assertTrue(any("9999" in error for error in result.errors))

    def test_final_weight_is_collected_before_camera_capture(self):
        events = []
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        def read_average(port, duration_seconds):
            events.append(("weight", port, duration_seconds))
            return SimpleNamespace(average_g=2450.0, samples=[2400, 2500])

        def capture(indices):
            events.append(("capture", dict(indices)))
            return {"comb": frame, "shank": frame}

        calibrations = {
            "comb": {"camera": 0, "shape": [100, 200], "cm_per_pixel": 0.1},
            "shank": {"camera": 1, "shape": [100, 200], "cm_per_pixel": 0.2},
        }
        with patch.object(dual_camera_ui, "capture_pair", side_effect=capture), \
             patch.object(dual_camera_ui, "save_pair", return_value={"comb": "c.png", "shank": "s.png"}):
            weight, result, paths, scales = dual_camera_ui.acquire_measurement_inputs(
                {"comb": 0, "shank": 1}, calibrations, "COM6", 0, read_average
            )

        self.assertEqual([event[0] for event in events], ["weight", "capture"])
        self.assertEqual(weight, 2450.0)
        self.assertEqual(len(result.samples), 2)
        self.assertEqual(paths["shank"], "s.png")
        self.assertEqual(scales, {"comb": 0.1, "shank": 0.2})


if __name__ == "__main__":
    unittest.main()
