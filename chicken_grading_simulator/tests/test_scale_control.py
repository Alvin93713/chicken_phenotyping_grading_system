import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"  # never start the real background uploader in tests
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

    def test_camera_capture_uses_calibrated_scales(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        calibrations = {
            "comb": {"camera": 0, "shape": [100, 200], "cm_per_pixel": 0.1},
            "shank": {"camera": 1, "shape": [100, 200], "cm_per_pixel": 0.2},
        }
        with patch.object(dual_camera_ui, "capture_pair", return_value={"comb": frame, "shank": frame}), \
             patch.object(dual_camera_ui, "save_pair", return_value={"comb": "c.png", "shank": "s.png"}):
            paths, scales = dual_camera_ui.acquire_measurement_inputs({"comb": 0, "shank": 1}, calibrations)
        self.assertEqual(paths["shank"], "s.png")
        self.assertEqual(scales, {"comb": 0.1, "shank": 0.2})

    def test_weight_reuses_stable_live_window_without_rereading(self):
        now = 100.0
        samples = [(99.0, 2440), (99.5, 2450), (100.0, 2460), (96.0, 9999)]  # last one is too old
        weight, count = dual_camera_ui.resolve_weight("COM6", False, 0, samples, now=now)
        self.assertEqual((weight, count), (2450.0, 3))

    def test_weight_requires_retry_instead_of_silent_manual_fallback(self):
        with self.assertRaisesRegex(ValueError, "使用手動重量"):
            dual_camera_ui.resolve_weight("COM6", False, 2000, [], now=100.0)  # no readings
        unstable = [(99.0, 2000), (99.5, 2300), (100.0, 2600)]
        with self.assertRaisesRegex(ValueError, "穩定"):
            dual_camera_ui.resolve_weight("COM6", False, 2000, unstable, now=100.0)

    def test_weight_below_minimum_is_rejected(self):
        empty_scale = [(99.0, 0), (99.5, 1), (100.0, 0)]
        with self.assertRaisesRegex(ValueError, "300"):
            dual_camera_ui.resolve_weight("COM6", False, 0, empty_scale, now=100.0)
        with self.assertRaisesRegex(ValueError, "300"):
            dual_camera_ui.resolve_weight("COM6", True, 0, [], now=100.0)  # manual 0 g

    def test_manual_weight_only_when_chosen_or_no_scale(self):
        self.assertEqual(dual_camera_ui.resolve_weight("COM6", True, 2100, [], now=1.0), (2100.0, 0))
        self.assertEqual(dual_camera_ui.resolve_weight("", False, 1900, [], now=1.0), (1900.0, 0))

    def test_blank_manual_weight_asks_for_input(self):
        with self.assertRaisesRegex(ValueError, "請輸入手動重量"):
            dual_camera_ui.resolve_weight("", False, None, [], now=1.0)  # field starts blank, nothing carried over

    def test_uploaded_photos_bypass_camera_capture(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frames = {"comb": frame, "shank": frame.copy()}
        supplied_scales = {"comb": 0.03, "shank": 0.04}

        with patch.object(dual_camera_ui, "capture_pair") as capture, \
             patch.object(dual_camera_ui, "save_pair", return_value={"comb": "c.png", "shank": "s.png"}) as save:
            paths, scales = dual_camera_ui.acquire_measurement_inputs(
                {"comb": 0, "shank": 1},
                {},
                supplied_frames=frames,
                supplied_scales=supplied_scales,
            )

        capture.assert_not_called()
        save.assert_called_once_with(frames)
        self.assertEqual(paths["comb"], "c.png")
        self.assertEqual(scales, supplied_scales)


class ScaleMonitorTests(unittest.TestCase):
    def test_background_reader_keeps_packets_split_across_reads(self):
        import time as _time
        import scale_control

        chunks = [b"\x022", b"450\x03\x022", b"460\x03", b"\x029999\x03"]

        class FakePort:
            in_waiting = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, size):
                _time.sleep(0.01)
                return chunks.pop(0) if chunks else b""

        monitor = scale_control.ScaleMonitor("FAKE", opener=lambda port: FakePort())
        try:
            deadline = _time.monotonic() + 2
            while _time.monotonic() < deadline and not monitor.snapshot(5).errors:  # the 9999 packet comes last
                _time.sleep(0.02)
            snap = monitor.snapshot(5)
            self.assertEqual([value for _, value in snap.samples], [2450, 2460])
            self.assertEqual(snap.error, "")
            self.assertTrue(any("9999" in e for e in snap.errors))
        finally:
            monitor.stop.set()
            monitor.thread.join(2)

    def test_split_complete_packets(self):
        from scale_control import split_complete_packets
        self.assertEqual(split_complete_packets(b"\x021\x03\x022"), (b"\x021\x03", b"\x022"))
        self.assertEqual(split_complete_packets(b"\x0212"), (b"", b"\x0212"))


class MaskSelectionAndLightTests(unittest.TestCase):
    def test_mask_selection_uses_confidence_and_area(self):
        from dual_camera import select_mask_index
        # big but doubtful (0.40) loses to a confident one; among confident ones the largest wins
        self.assertEqual(select_mask_index([5000, 1200, 1500], [0.40, 0.92, 0.85]), 2)
        self.assertEqual(select_mask_index([5000, 1200], [0.90, 0.95]), 0)
        self.assertEqual(select_mask_index([10, 30], []), 1)  # no confidences → largest

    def test_light_for_does_not_block(self):
        import arduino_control
        sent = []
        with patch.object(arduino_control, "send_arduino_commands",
                          side_effect=lambda port, cmds, **kw: sent.append(list(cmds)) or ["OK"]), \
             patch.object(arduino_control.time, "sleep", side_effect=AssertionError("must not sleep")):
            arduino_control.light_for("COM3", ["LIGHT:RED", "BUZZ:250"], 1.0)
        self.assertEqual(sent, [["LIGHT_FOR:RED:1000", "BUZZ:250"]])

    def test_light_for_falls_back_for_old_firmware(self):
        import arduino_control
        sent = []

        def fake_send(port, cmds, **kw):
            sent.append(list(cmds))
            return ["ERR:UNKNOWN:LIGHT_FOR:GREEN:3000"] if cmds[0].startswith("LIGHT_FOR") else ["OK:LIGHT:GREEN"]

        with patch.object(arduino_control, "send_arduino_commands", side_effect=fake_send), \
             patch.object(arduino_control.threading, "Timer") as timer:
            arduino_control.light_for("COM3", ["LIGHT:GREEN"], 3.0)
        self.assertEqual(sent, [["LIGHT_FOR:GREEN:3000"], ["LIGHT:GREEN"]])
        timer.assert_called_once()
        timer.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
