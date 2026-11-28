import os as _preload_os
_preload_os.environ["CHICKEN_PRELOAD_MODELS"] = "0"  # tests never load the real models
_preload_os.environ["CHICKEN_AUTO_BUILD_ENGINE"] = "0"  # … nor build TensorRT engines
import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"
_os.environ["CHICKEN_UI_PREFS_PATH"] = _os.devnull
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import database  # noqa: E402
import dual_camera_ui  # noqa: E402
import product_review  # noqa: E402


class CameraSettingsTests(unittest.TestCase):
    def test_every_calibration_is_appended_with_its_change(self):
        with tempfile.TemporaryDirectory() as folder,              patch.dict(_os.environ, {"CHICKEN_CAMERA_SETTINGS_PATH": str(Path(folder) / "cams.json")}):
            points = {"comb": [(0, 0), (100, 0)], "shank": [(0, 0), (0, 200)]}
            photos = {"comb": "c.png", "shank": "s.png"}
            first = {"comb": {"camera": 0, "shape": [480, 640], "cm_per_pixel": 0.2},
                     "shank": {"camera": 1, "shape": [480, 640], "cm_per_pixel": 0.1}}
            entry = dual_camera_ui.record_calibration(first, points, {"comb": 20.0, "shank": 20.0}, photos, now=0)
            self.assertIsNone(entry["roles"]["comb"]["change_ratio"])  # nothing to compare with yet
            self.assertEqual(entry["roles"]["shank"]["pixel_distance"], 200.0)
            second = {"comb": {"camera": 0, "shape": [480, 640], "cm_per_pixel": 0.22},
                      "shank": {"camera": 1, "shape": [480, 640], "cm_per_pixel": 0.1}}
            entry = dual_camera_ui.record_calibration(second, points, {"comb": 22.0, "shank": 20.0}, photos)
            self.assertAlmostEqual(entry["roles"]["comb"]["change_ratio"], 0.10)  # +10 % -> warned on the page
            self.assertAlmostEqual(entry["roles"]["shank"]["change_ratio"], 0.0)
            history = dual_camera_ui.load_calibration_history()
            self.assertEqual(len(history), 2)  # appended, the first one is kept
            self.assertEqual(history[0]["roles"]["comb"]["cm_per_pixel"], 0.2)
            self.assertEqual(history[1]["roles"]["comb"]["ruler_cm"], 22.0)

    def test_camera_ids_follow_from_adjustment_to_calibration_and_survive_a_new_session(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(_os.environ, {"CHICKEN_CAMERA_SETTINGS_PATH": str(Path(folder) / "cams.json")}), \
             patch.object(database, "DB_PATH", Path(folder) / "g.db"), \
             patch.object(product_review, "REVIEW_DB_PATH", Path(folder) / "r.db"), \
             patch.object(dual_camera_ui, "ensure_preview_server", return_value=8765):
            at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=30).run()
            nav = lambda page: at.segmented_control(key="main_page_navigation_segmented").set_value(page).run()
            nav("相機調整")
            at.number_input(key="adjustment_comb_camera").set_value(3).run()
            at.number_input(key="adjustment_shank_camera").set_value(2).run()
            nav("比例尺校正")
            self.assertEqual((at.number_input(key="calibration_comb_camera").value,
                              at.number_input(key="calibration_shank_camera").value), (3, 2))
            saved = json.loads((Path(folder) / "cams.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["cameras"], {"comb": 3, "shank": 2})
            del at

            # a brand-new session (page refresh, new tab or restart) starts from the saved IDs
            fresh = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=30).run()
            fresh.segmented_control(key="main_page_navigation_segmented").set_value("比例尺校正").run()
            self.assertEqual((fresh.number_input(key="calibration_comb_camera").value,
                              fresh.number_input(key="calibration_shank_camera").value), (3, 2))
            self.assertEqual(len(fresh.exception), 0, str(fresh.exception))
            del fresh
            gc.collect()

    def test_feature_checkboxes_show_units(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(database, "DB_PATH", Path(folder) / "g.db"), \
             patch.object(product_review, "REVIEW_DB_PATH", Path(folder) / "r.db"):
            at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=30).run()
            at.segmented_control(key="main_page_navigation_segmented").set_value("分級門檻設定").run()
            labels = [box.label for box in at.checkbox if box.key and box.key.startswith("use_feature_")]
            self.assertEqual(labels, ["體重 (g)", "雞冠面積 (cm²)", "腳脛寬度 (cm)", "腳脛長度 (cm)"])
            del at
            gc.collect()


if __name__ == "__main__":
    unittest.main()
