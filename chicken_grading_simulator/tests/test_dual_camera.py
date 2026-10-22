import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dual_camera as dual
import inference_queue as queue


class Tensor:
    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class DualCameraTests(unittest.TestCase):
    def test_camera_and_resolution_must_match_calibration(self):
        calibration = dict(camera=1, shape=[100, 200], cm_per_pixel=0.2)
        self.assertEqual(dual.validate_calibration(calibration, 1, (100,200,3)), 0.2)
        for camera, shape in [(0,(100,200,3)), (1,(200,100,3))]:
            with self.assertRaises(ValueError):
                dual.validate_calibration(calibration, camera, shape)

    def test_same_camera_rejected_before_capture(self):
        with self.assertRaises(ValueError):
            dual.capture_pair(dict(comb=0, shank=0))

    def test_distinct_models_and_scales(self):
        import cv2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = np.zeros((1,100,200), dtype=np.float32)
            masks[0,10:31,20:61] = 1
            result = SimpleNamespace(masks=SimpleNamespace(data=Tensor(masks)), boxes=[1],
                                     plot=lambda: np.zeros((100,200,3), dtype=np.uint8))
            models = {r: Mock(names={0:r}, predict=Mock(return_value=[result])) for r in dual.ROLES}
            paths = {}
            for role in dual.ROLES:
                path = root / f"{role}.png"
                ok, encoded = cv2.imencode(".png", result.plot())
                self.assertTrue(ok)
                encoded.tofile(str(path))
                paths[role] = str(path)
            with patch.object(dual, "model_for", side_effect=lambda r: models[r]), patch.object(dual, "OUTPUTS_DIR", root / "output"):
                values = dual.infer_pair(paths, dict(comb=0.1, shank=0.2), 0.5, Mock())
            self.assertAlmostEqual(values["comb_area_cm2"], 21 * 41 * 0.01)
            self.assertAlmostEqual(values["shank_width_cm"], 4.0)
            self.assertAlmostEqual(values["shank_length_cm"], 8.0)
            for model in models.values():
                self.assertEqual(model.predict.call_count, 1)
                self.assertEqual(model.predict.call_args.kwargs["classes"], [0])
                self.assertTrue(model.predict.call_args.kwargs["retina_masks"])

    def test_failure_does_not_write_database_or_light(self):
        job = SimpleNamespace(job_id="test", snapshot_paths={}, snapshot_scales={}, confidence_threshold=0.5)
        with patch.object(queue, "_update_job"), patch.object(queue, "infer_pair", side_effect=ValueError("missing shank")), patch.object(queue, "update_chicken") as update, patch.object(queue, "add_chicken") as add, patch.object(queue, "_run_preliminary_grading_and_light") as light:
            with self.assertRaises(ValueError):
                queue._process_snapshot(job)
            update.assert_not_called()
            add.assert_not_called()
            light.assert_not_called()

    def test_enqueue_freezes_photo_and_scale_settings(self):
        paths = dict(comb="/capture/abc/comb.png", shank="/capture/abc/shank.png")
        scales = dict(comb=0.01, shank=0.02)
        with patch.object(queue, "ensure_worker_running"), patch.object(queue, "_queue"):
            job_id = queue.enqueue_snapshot_job("C001", 2500, paths, scales, 0.5, False, "")
        scales["shank"] = 9
        paths["comb"] = "other.png"
        job = queue.get_job(job_id)
        self.assertEqual(job.snapshot_scales["shank"], 0.02)
        self.assertEqual(job.snapshot_paths["comb"], "/capture/abc/comb.png")
        with queue._lock:
            del queue._jobs[job_id]


if __name__ == "__main__":
    unittest.main()
