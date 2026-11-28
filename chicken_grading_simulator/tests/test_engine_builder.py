import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import engine_builder as eb  # noqa: E402


def fake_export(pt, role, workdir):
    engine = workdir / (pt.stem + ".engine")
    engine.write_bytes(b"engine for " + pt.read_bytes())
    return engine


class EngineBuilderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.models = Path(self.folder.name)
        self.pt = {role: self.models / f"best_{role}_rayshing_20261128.pt" for role in eb.ROLES}
        for path in self.pt.values():
            path.write_bytes(b"v1")
        eb.status.clear()
        eb._failed.clear()
        self.warm = patch.object(eb, "_warm")
        self.warm.start()

    def tearDown(self):
        self.warm.stop()
        self.folder.cleanup()

    def test_missing_and_stale_engines_are_built(self):
        self.assertTrue(eb.needs_engine(self.pt["comb"]))
        with patch.object(eb, "export_engine", side_effect=fake_export), patch.object(eb, "check_engine"):
            self.assertEqual(eb.check_and_build(self.pt), ["comb", "shank"])
        engine = self.pt["comb"].with_suffix(".engine")
        self.assertEqual(engine.read_bytes(), b"engine for v1")
        self.assertGreaterEqual(engine.stat().st_mtime_ns, self.pt["comb"].stat().st_mtime_ns)
        self.assertFalse(eb.needs_engine(self.pt["comb"]))
        self.assertEqual(eb.status["comb"]["state"], "ready")
        # nothing to do the second time
        with patch.object(eb, "export_engine", side_effect=AssertionError("rebuilt")):
            self.assertEqual(eb.check_and_build(self.pt), [])

    def test_a_new_model_with_the_same_name_is_rebuilt(self):
        with patch.object(eb, "export_engine", side_effect=fake_export), patch.object(eb, "check_engine"):
            eb.check_and_build(self.pt)
            engine = self.pt["shank"].with_suffix(".engine")
            self.pt["shank"].write_bytes(b"v2")
            os.utime(self.pt["shank"], ns=(engine.stat().st_mtime_ns + 10**9,) * 2)  # newer than the engine
            self.assertEqual(eb.check_and_build(self.pt), ["shank"])
        self.assertEqual(self.pt["shank"].with_suffix(".engine").read_bytes(), b"engine for v2")

    def test_an_engine_that_fails_its_check_is_never_installed_or_retried_blindly(self):
        with patch.object(eb, "export_engine", side_effect=fake_export), \
                patch.object(eb, "check_engine", side_effect=RuntimeError("TensorRT mismatch")):
            self.assertEqual(eb.check_and_build(self.pt), [])
        self.assertFalse(self.pt["comb"].with_suffix(".engine").exists())  # the .pt stays in use
        self.assertEqual(eb.status["comb"]["state"], "failed")
        self.assertIn("TensorRT mismatch", eb.status["comb"]["message"])
        with patch.object(eb, "export_engine", side_effect=AssertionError("retried")):
            eb.check_and_build(self.pt)  # same model: not retried every minute
        eb.retry("comb")  # the UI button forgets the failure
        with patch.object(eb, "export_engine", side_effect=fake_export), patch.object(eb, "check_engine"):
            self.assertEqual(eb.check_and_build(self.pt), ["comb"])

    def test_an_existing_engine_is_kept_when_a_rebuild_fails(self):
        with patch.object(eb, "export_engine", side_effect=fake_export), patch.object(eb, "check_engine"):
            eb.check_and_build(self.pt)
        old = self.pt["comb"].with_suffix(".engine").read_bytes()
        os.utime(self.pt["comb"], ns=(self.pt["comb"].with_suffix(".engine").stat().st_mtime_ns + 10**9,) * 2)
        with patch.object(eb, "export_engine", side_effect=RuntimeError("out of memory")):
            eb.check_and_build(self.pt)
        self.assertEqual(self.pt["comb"].with_suffix(".engine").read_bytes(), old)  # untouched (and stale → PT used)

    def test_off_without_tensorrt_or_when_disabled(self):
        with patch.dict(os.environ, {"CHICKEN_AUTO_BUILD_ENGINE": "0"}):
            self.assertFalse(eb.enabled())
        with patch.dict(os.environ, {"CHICKEN_DISABLE_TENSORRT": "1"}):
            self.assertFalse(eb.enabled())
        with patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(eb.enabled())
            self.assertFalse(eb.start())
        self.assertEqual(eb.status["comb"]["state"], "unavailable")


if __name__ == "__main__":
    unittest.main()
