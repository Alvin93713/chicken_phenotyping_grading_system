import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"
import os
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model_registry  # noqa: E402


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.legacy = {role: root / f"best_{role}.pt" for role in model_registry.ROLES}
        self.patches = [
            patch.object(model_registry, "MODELS_DIR", root / "models"),
            patch.object(model_registry, "SELECTION_PATH", root / "data" / "model_versions.json"),
            patch.object(model_registry, "LEGACY_PATHS", self.legacy),
        ]
        for item in self.patches:
            item.start()
        model_registry._forget_cache()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        model_registry._forget_cache()
        self.temp.cleanup()

    def test_original_file_is_used_until_a_version_is_imported(self):
        self.legacy["comb"].write_bytes(b"orig")
        self.assertEqual(model_registry.active_path("comb"), self.legacy["comb"])
        self.assertTrue(model_registry.list_versions("comb")[0].legacy)

    def test_file_name_carries_farm_and_date(self):
        self.assertEqual(model_registry.version_file_name("comb", "NTU lab/雞舍A", date(2026, 11, 26)),
                         "best_comb_NTU-lab-雞舍A_20261126.pt")
        path = model_registry.import_model("shank", b"model", "farm_a", "20261126")
        version = model_registry.parse_version(path)
        self.assertEqual((version.role, version.farm, version.updated), ("shank", "farm-a", "20261126"))

    def test_newest_date_wins_unless_pinned_and_nothing_is_overwritten(self):
        self.legacy["comb"].write_bytes(b"orig")
        old = model_registry.import_model("comb", b"v1", "farm", date(2026, 10, 1))
        new = model_registry.import_model("comb", b"v2", "farm", date(2026, 11, 26))
        again = model_registry.import_model("comb", b"v3", "farm", date(2026, 11, 26))
        self.assertNotEqual(new, again)  # same name → _2 suffix, both kept
        self.assertEqual(new.read_bytes(), b"v2")
        model_registry._forget_cache()
        self.assertIn(model_registry.active_path("comb"), (new, again))
        model_registry.set_active("comb", old.name)  # roll back
        self.assertEqual(model_registry.active_path("comb"), old)
        self.assertEqual(model_registry.ActiveModels()["comb"], old)
        model_registry.set_active("comb", model_registry.AUTO)
        self.assertNotEqual(model_registry.active_path("comb"), old)
        with self.assertRaises(ValueError):
            model_registry.set_active("comb", "best_comb_nope_20200101.pt")

    def test_rejected_model_leaves_nothing_behind(self):
        def reject(path, role):
            raise ValueError("not a segmentation model")

        with self.assertRaises(ValueError):
            model_registry.import_model("comb", b"bad", "farm", date(2026, 11, 26), reject)
        self.assertEqual(list(model_registry.MODELS_DIR.iterdir()), [])

    def test_register_original_copies_with_file_date(self):
        self.legacy["shank"].write_bytes(b"orig")
        stamp = time.mktime((2026, 9, 5, 12, 0, 0, 0, 0, -1))
        os.utime(self.legacy["shank"], (stamp, stamp))
        path = model_registry.register_original("shank", "ntu-lab")
        self.assertEqual(path.name, "best_shank_ntu-lab_20260905.pt")
        self.assertTrue(self.legacy["shank"].is_file())  # original stays for older tools


if __name__ == "__main__":
    unittest.main()
