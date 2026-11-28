import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dual_camera  # noqa: E402


class FakeModel:
    def __init__(self, path, fail_predict=False):
        self.path, self.fail_predict, self.names = path, fail_predict, {0: "comb"}

    def predict(self, *args, **kwargs):
        if self.fail_predict:
            raise RuntimeError("engine shape mismatch")
        return [f"result from {Path(self.path).suffix}"]


class TensorRTRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.pt = Path(self.folder.name) / "best_comb.pt"
        self.engine = self.pt.with_suffix(".engine")
        self.pt.write_bytes(b"pt")
        os.utime(self.pt, ns=(1_000_000_000, 1_000_000_000))
        self.paths = patch.object(dual_camera, "MODEL_PATHS", {"comb": self.pt})
        self.paths.start()
        dual_camera._engine_failures.clear()
        dual_camera.load_model.cache_clear()

    def tearDown(self):
        self.paths.stop()
        dual_camera._engine_failures.clear()
        dual_camera.load_model.cache_clear()
        self.folder.cleanup()

    def make_engine(self, mtime_s):
        self.engine.write_bytes(b"engine")
        os.utime(self.engine, ns=(mtime_s * 1_000_000_000, mtime_s * 1_000_000_000))

    def test_pt_without_engine(self):
        self.assertEqual(dual_camera.runtime_model("comb"), (self.pt, "PyTorch"))

    def test_engine_is_preferred(self):
        self.make_engine(2)
        self.assertEqual(dual_camera.runtime_model("comb"), (self.engine, "TensorRT"))

    def test_stale_engine_is_skipped_until_rebuilt(self):
        self.make_engine(0)  # built from an older .pt
        self.assertEqual(dual_camera.runtime_model("comb")[1], "PyTorch")
        self.assertIn("舊", dual_camera.runtime_status("comb")["reason"])
        self.make_engine(3)  # rebuilt after the pull
        self.assertEqual(dual_camera.runtime_model("comb")[1], "TensorRT")

    def test_disable_switch(self):
        self.make_engine(2)
        with patch.dict(os.environ, {"CHICKEN_DISABLE_TENSORRT": "1"}):
            self.assertEqual(dual_camera.runtime_model("comb")[1], "PyTorch")

    def test_engine_that_cannot_load_falls_back_to_pt(self):
        self.make_engine(2)

        def load(path, modified):
            if path.endswith(".engine"):
                raise RuntimeError("TensorRT version mismatch")
            return FakeModel(path)

        with patch.object(dual_camera, "load_model", side_effect=load):
            model = dual_camera.model_for("comb")
        self.assertTrue(model.path.endswith(".pt"))
        self.assertEqual(dual_camera.runtime_model("comb")[1], "PyTorch")  # not retried every frame
        self.make_engine(5)  # a rebuilt engine is tried again
        self.assertEqual(dual_camera.runtime_model("comb")[1], "TensorRT")

    def test_engine_that_fails_to_predict_is_retried_with_pt(self):
        self.make_engine(2)
        models = {".engine": FakeModel("x.engine", fail_predict=True), ".pt": FakeModel("x.pt")}
        with patch.object(dual_camera, "load_model", side_effect=lambda path, m: models[Path(path).suffix]):
            result, model = dual_camera.predict_role("comb", object(), 0.5)
        self.assertEqual(result, "result from .pt")
        self.assertEqual(dual_camera.runtime_status("comb")["runtime"], "PyTorch")


if __name__ == "__main__":
    unittest.main()
