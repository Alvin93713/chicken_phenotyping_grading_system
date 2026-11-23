import sys
import tempfile
import unittest
import zipfile
import gc
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import database
import inference_queue
import product_review


class ProductReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.review_dir = self.root / "review"
        self.db_path = self.review_dir / "review.db"
        self.patches = (
            patch.object(product_review, "REVIEW_DIR", self.review_dir),
            patch.object(product_review, "REVIEW_DB_PATH", self.db_path),
        )
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def _record(self, job_id, confidences, *, error="", sample_rate=0.0):
        return product_review.record_case(
            job_id=job_id,
            chicken_id="C001",
            pipeline_status="失敗" if error else "已完成",
            confidence_threshold=0.25,
            role_confidences=confidences,
            measurements={"weight_g": 2500},
            source_paths={},
            result_paths={},
            models={"comb": "comb.pt", "shank": "shank.pt"},
            error=error,
            sample_rate=sample_rate,
        )

    def test_only_low_confidence_failure_or_sample_is_collected(self):
        self.assertFalse(self._record("high", {"comb": 0.91, "shank": 0.88}))
        self.assertTrue(self._record("low", {"comb": 0.48, "shank": 0.82}))
        self.assertTrue(self._record("failed", {"comb": None, "shank": None}, error="no mask"))
        cases = product_review.list_cases()
        self.assertEqual({case["job_id"] for case in cases}, {"low", "failed"})

    def test_review_and_export_bundle(self):
        image = self.root / "result.png"
        image.write_bytes(b"image")
        product_review.record_case(
            job_id="export", chicken_id="C002", pipeline_status="已完成",
            confidence_threshold=0.25, role_confidences={"comb": 0.4, "shank": 0.8},
            measurements={}, source_paths={}, result_paths={"comb": str(image)},
            models={}, force=True,
        )
        product_review.review_case("export", "不正確", "遮罩錯誤")
        case = product_review.get_case("export")
        self.assertEqual(case["review_state"], "不正確")
        with zipfile.ZipFile(BytesIO(product_review.export_review_bundle())) as archive:
            self.assertIn("manifest.csv", archive.namelist())
            self.assertIn("manifest.json", archive.namelist())
            self.assertTrue(any(name.endswith("result_comb.png") for name in archive.namelist()))

    def test_quality_capture_failure_never_changes_core_result(self):
        job = SimpleNamespace(snapshot_paths={"comb": str(self.root / "capture" / "comb.png")})
        with patch.object(inference_queue, "record_snapshot_review", side_effect=RuntimeError("db locked")):
            inference_queue._capture_product_review(job, measurements={"weight_g": 2500})

    def test_product_pages_render_and_seed_demo_cases(self):
        app_path = Path(__file__).resolve().parents[1] / "src" / "app.py"
        with patch.object(database, "DB_PATH", self.root / "grading.db"):
            app = AppTest.from_file(str(app_path), default_timeout=20).run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            for page in ("模型品質監控", "案例審核", "資料與模型維護"):
                app.segmented_control[0].set_value(page).run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
            next(button for button in app.button if button.label == "載入三筆示範案例").click().run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(product_review.review_summary()["總案例"], 3)
            del app
            gc.collect()


if __name__ == "__main__":
    unittest.main()
