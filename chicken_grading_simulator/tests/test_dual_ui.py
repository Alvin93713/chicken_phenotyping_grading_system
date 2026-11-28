import os as _preload_os
_preload_os.environ["CHICKEN_PRELOAD_MODELS"] = "0"  # tests never load the real models
_preload_os.environ["CHICKEN_AUTO_BUILD_ENGINE"] = "0"  # … nor build TensorRT engines
import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"  # never start the real background uploader in tests
_os.environ["CHICKEN_UI_PREFS_PATH"] = _os.devnull  # tests always use the default 原版+深色 look
import tempfile as _tempfile
# never touch the station's real data/camera_settings.json
_os.environ["CHICKEN_CAMERA_SETTINGS_PATH"] = _os.path.join(_tempfile.mkdtemp(), "camera_settings.json")
import sys
import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import database
import dual_camera_ui


class DualUITests(unittest.TestCase):
    def test_aruco_calibration_is_the_default_and_needs_no_clicks(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_aruco_calibration import scene
        frames = {"comb": scene(marker_id=0, side=250), "shank": scene(marker_id=0, side=250)}
        with tempfile.TemporaryDirectory() as folder, patch.object(database, "DB_PATH", Path(folder)/"test.db"),                 patch.object(dual_camera_ui, "ensure_preview_server", return_value=8765),                 patch.object(dual_camera_ui, "capture_one", side_effect=lambda index: frames["comb" if index == 0 else "shank"]),                 patch("dual_camera.BASE_DIR", Path(folder)):
            at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=20).run()
            at.segmented_control(key="main_page_navigation_segmented").set_value("比例尺校正").run()
            self.assertEqual(at.segmented_control(key="calibration_method_comb").value, "aruco")
            self.assertTrue(at.button(key="calibration_confirm_comb").disabled)  # nothing to confirm before a photo
            at.number_input(key="dual_comb_marker_cm").set_value(3.6).run()  # measured side of a scaled print
            at.button(key="calibration_capture_comb").click().run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
            self.assertFalse(at.button(key="calibration_confirm_comb").disabled)
            at.button(key="calibration_confirm_comb").click().run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
            comb = at.session_state["dual_calibrations"]["comb"]
            self.assertEqual(comb["method"], "aruco")
            self.assertAlmostEqual(comb["cm_per_pixel"], 3.6 / 250, delta=3.6 / 250 * 0.01)
            history = dual_camera_ui.load_calibration_history()[-1]["roles"]["comb"]
            self.assertEqual((history["method"], history["marker_id"], history["ruler_cm"]), ("aruco", 0, 3.6))
            # the shank camera sees ID 0 instead of its own ID 1: refused, with the reason shown
            at.button(key="calibration_capture_shank").click().run()
            self.assertTrue(at.button(key="calibration_confirm_shank").disabled)
            self.assertTrue(any("ID 1" in e.value for e in at.error), [e.value for e in at.error])
            # the measured side length is remembered for the next session
            fresh = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=20).run()
            fresh.segmented_control(key="main_page_navigation_segmented").set_value("比例尺校正").run()
            self.assertAlmostEqual(fresh.number_input(key="dual_comb_marker_cm").value, 3.6)

    def test_pages_and_independent_calibration(self):
        frame = np.zeros((480,640,3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder, patch.object(database, "DB_PATH", Path(folder)/"test.db"), patch.object(dual_camera_ui, "ensure_preview_server", return_value=8765), patch.object(dual_camera_ui, "capture_pair", return_value={r:frame for r in ("comb","shank")}), patch.object(dual_camera_ui, "capture_one", return_value=frame), patch("dual_camera.BASE_DIR", Path(folder)):
            at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=20).run()
            self.assertEqual(len(at.exception), 0)
            for page in ["相機調整", "比例尺校正", "掃描 QR code 與表型輸入"]:
                at.segmented_control(key="main_page_navigation_segmented").set_value(page).run()
                self.assertEqual(len(at.exception), 0, str(at.exception))
                labels = [n.label for n in at.number_input]
                self.assertIn("雞冠相機編號", labels)
                self.assertIn("腳脛相機編號", labels)
            at.segmented_control(key="main_page_navigation_segmented").set_value("比例尺校正").run()
            # ArUco is the default method; this test uses manual points on both cameras
            for role in ("comb", "shank"):
                at.segmented_control(key=f"calibration_method_{role}").set_value("manual").run()
            # each camera is calibrated on its own: comb first ...
            at.button(key="calibration_capture_comb").click().run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
            self.assertNotIn("dual_calibration_photo_shank", at.session_state)  # shank untouched
            at.session_state["dual_comb_points"] = [(0,0),(100,0)]
            at.run()
            at.button(key="calibration_confirm_comb").click().run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
            self.assertAlmostEqual(at.session_state["dual_calibrations"]["comb"]["cm_per_pixel"], 0.2)
            self.assertNotIn("shank", at.session_state["dual_calibrations"])
            # ... then shank, which keeps comb's calibration
            at.button(key="calibration_capture_shank").click().run()
            at.session_state["dual_shank_points"] = [(0,0),(200,0)]
            at.run()
            at.button(key="calibration_confirm_shank").click().run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
            calibration = at.session_state["dual_calibrations"]
            self.assertAlmostEqual(calibration["comb"]["cm_per_pixel"], 0.2)
            self.assertAlmostEqual(calibration["shank"]["cm_per_pixel"], 0.1)
            at.segmented_control(key="main_page_navigation_segmented").set_value("掃描 QR code 與表型輸入").run()
            self.assertEqual(at.session_state["dual_calibrations"], calibration)
            confidence = next(item for item in at.number_input if item.label == "辨識信心門檻")
            self.assertAlmostEqual(confidence.value, 0.5)
            at.toggle[0].set_value(True).run()
            self.assertEqual(len(at.file_uploader), 2)
            self.assertEqual(
                {item.label for item in at.file_uploader},
                {"上傳雞冠照片", "上傳腳脛照片"},
            )
            self.assertTrue(
                {"雞冠比例 (cm/像素)", "腳脛比例 (cm/像素)"}
                <= {item.label for item in at.number_input}
            )
            at.session_state["phenotype_entry_confirmed_manual_id"] = "C001"
            at.run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
            self.assertTrue(any(b.label == "計算雞冠及腳脛數據" for b in at.button))
            gc.collect()


if __name__ == "__main__":
    unittest.main()
