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
