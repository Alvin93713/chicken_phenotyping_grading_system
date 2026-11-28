import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dual_camera  # noqa: E402


class PreviewPauseTests(unittest.TestCase):
    def test_snapshot_job_pauses_preview_inference_and_resumes_after(self):
        self.assertFalse(dual_camera.preview_inference_paused())
        with dual_camera.snapshot_inference():
            self.assertTrue(dual_camera.preview_inference_paused())
            with dual_camera.snapshot_inference():  # two jobs at once
                pass
            self.assertTrue(dual_camera.preview_inference_paused())
        self.assertFalse(dual_camera.preview_inference_paused())

    def test_pause_is_released_when_the_job_fails(self):
        with self.assertRaises(ValueError):
            with patch.object(dual_camera, "_infer_pair", side_effect=ValueError("no mask")):
                dual_camera.infer_pair({}, {}, 0.5, lambda *a: None)
        self.assertFalse(dual_camera.preview_inference_paused())

    def test_preview_shows_the_plain_picture_while_a_job_runs(self):
        frame = np.full((4, 4, 3), 7, np.uint8)
        calls = []

        class Camera:
            def latest(self):
                return frame, time.monotonic()

        def annotate(role, source, confidence):
            calls.append(role)
            return source + 1

        with patch.object(dual_camera, "reader", return_value=Camera()), \
                patch.object(dual_camera, "annotate_preview_frame", side_effect=annotate):
            with dual_camera.snapshot_inference():
                preview = dual_camera.InferenceReader("comb", 0, 0.5)
                time.sleep(0.4)
                paused_frame, _ = preview.latest()
                self.assertEqual(calls, [])  # no YOLO during the job
                self.assertTrue(np.array_equal(paused_frame, frame))
            time.sleep(0.4)
            resumed_frame, _ = preview.latest()
            preview.stop.set()
        self.assertTrue(calls)  # inference is back
        self.assertTrue(np.array_equal(resumed_frame, frame + 1))


class PreloadTests(unittest.TestCase):
    def setUp(self):
        dual_camera._preload_started = False
        dual_camera.preload_state.clear()

    def tearDown(self):
        dual_camera._preload_started = False

    def test_both_models_are_loaded_and_warmed_once(self):
        warmed = []
        done = threading.Event()

        def predict(role, frame, confidence):
            warmed.append((role, frame.shape))
            if len(warmed) == 2:
                done.set()

        with patch.dict(os.environ, {"CHICKEN_PRELOAD_MODELS": "1"}), \
                patch.object(dual_camera, "predict_role", side_effect=predict):
            self.assertTrue(dual_camera.preload_models())
            self.assertFalse(dual_camera.preload_models())  # once per process
            self.assertTrue(done.wait(5))
        self.assertEqual([role for role, _ in warmed], ["comb", "shank"])
        time.sleep(0.05)
        self.assertEqual(dual_camera.preload_state, {"comb": "ready", "shank": "ready"})

    def test_a_missing_model_does_not_stop_the_other(self):
        done = threading.Event()

        def predict(role, frame, confidence):
            if role == "comb":
                raise FileNotFoundError("best_comb.pt")
            done.set()

        with patch.dict(os.environ, {"CHICKEN_PRELOAD_MODELS": "1"}), \
                patch.object(dual_camera, "predict_role", side_effect=predict):
            dual_camera.preload_models()
            self.assertTrue(done.wait(5))
        time.sleep(0.05)
        self.assertTrue(dual_camera.preload_state["comb"].startswith("error"))
        self.assertEqual(dual_camera.preload_state["shank"], "ready")

    def test_switch_off(self):
        with patch.dict(os.environ, {"CHICKEN_PRELOAD_MODELS": "0"}):
            self.assertFalse(dual_camera.preload_models())


if __name__ == "__main__":
    unittest.main()
