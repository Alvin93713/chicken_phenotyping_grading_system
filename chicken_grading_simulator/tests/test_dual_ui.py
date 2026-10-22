import sys
import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import database
import dual_camera_ui


class DualUITests(unittest.TestCase):
    def test_pages_and_independent_calibration(self):
        frame = np.zeros((480,640,3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder, patch.object(database, "DB_PATH", Path(folder)/"test.db"), patch.object(dual_camera_ui, "reader", return_value=SimpleNamespace(latest=lambda:(frame,0))), patch.object(dual_camera_ui, "capture_pair", return_value={r:frame for r in ("comb","shank")}), patch("dual_camera.BASE_DIR", Path(folder)):
            at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=20).run()
            self.assertEqual(len(at.exception), 0)
            for page in ["相機調整", "比例尺校正", "掃描 QR code 與表型輸入"]:
                at.segmented_control[0].set_value(page).run()
                self.assertEqual(len(at.exception), 0, str(at.exception))
                labels = [n.label for n in at.number_input]
                self.assertIn("Comb 雞冠 camera ID", labels)
                self.assertIn("Shank 腳脛 camera ID", labels)
            at.segmented_control[0].set_value("比例尺校正").run()
            next(b for b in at.button if b.label == "拍攝雙相機校正照片").click().run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
            at.session_state["dual_comb_points"] = [(0,0),(100,0)]
            at.session_state["dual_shank_points"] = [(0,0),(200,0)]
            at.run()
            next(b for b in at.button if b.label == "確定雙相機校正").click().run()
            self.assertEqual(len(at.exception), 0)
            calibration = at.session_state["dual_calibrations"]
            self.assertAlmostEqual(calibration["comb"]["cm_per_pixel"], 0.1)
            self.assertAlmostEqual(calibration["shank"]["cm_per_pixel"], 0.05)
            at.segmented_control[0].set_value("掃描 QR code 與表型輸入").run()
            self.assertEqual(at.session_state["dual_calibrations"], calibration)
            at.session_state["phenotype_entry_confirmed_manual_id"] = "C001"
            at.run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
            self.assertTrue(any(b.label == "計算雞冠及腳脛數據" for b in at.button))
            gc.collect()


if __name__ == "__main__":
    unittest.main()
