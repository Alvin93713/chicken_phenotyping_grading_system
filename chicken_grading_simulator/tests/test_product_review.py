import os as _preload_os
_preload_os.environ["CHICKEN_PRELOAD_MODELS"] = "0"  # tests never load the real models
import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"  # never start the real background uploader in tests
_os.environ["CHICKEN_UI_PREFS_PATH"] = _os.devnull  # tests always use the default 原版+深色 look
import json
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
import product_review_ui


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
        self.assertEqual(case["comb_review_state"], "不正確")
        self.assertEqual(case["shank_review_state"], "不正確")
        with zipfile.ZipFile(BytesIO(product_review.export_review_bundle())) as archive:
            self.assertIn("manifest.csv", archive.namelist())
            self.assertIn("manifest.json", archive.namelist())
            self.assertIn("comb/to_relabel/review_notes.csv", archive.namelist())

    def _case_with_files(self, job_id, label_roles=("comb", "shank")):
        folder = self.root / job_id
        folder.mkdir()
        sources, results = {}, {}
        for role in ("comb", "shank"):
            (folder / f"{role}_src.jpg").write_bytes(f"{job_id}-{role}-photo".encode())
            (folder / f"{role}.png").write_bytes(b"overlay")
            if role in label_roles:
                (folder / f"{role}.txt").write_text("0 0.1 0.1 0.5 0.1 0.3 0.6\n", encoding="ascii")
            sources[role] = str(folder / f"{role}_src.jpg")
            results[role] = str(folder / f"{role}.png")
        product_review.record_case(
            job_id=job_id, chicken_id=job_id, pipeline_status="已完成",
            confidence_threshold=0.25, role_confidences={"comb": 0.4, "shank": 0.8},
            measurements={}, source_paths=sources, result_paths=results, models={}, force=True,
        )

    def test_export_bundle_is_a_fine_tuning_dataset(self):
        self._case_with_files("good")
        self._case_with_files("bad")
        self._case_with_files("unsure")
        self._case_with_files("nolabel", label_roles=())
        self._case_with_files("partial")
        self._case_with_files("pending")
        product_review.review_case("good", "正確")
        product_review.review_case("bad", "不正確", "遮罩漏掉尖端")
        product_review.review_case("unsure", "無法判斷")
        product_review.review_case("nolabel", "正確")
        product_review.review_case_role("partial", "comb", "正確")

        with zipfile.ZipFile(BytesIO(product_review.export_review_bundle())) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            notes = archive.read("comb/to_relabel/review_notes.csv").decode("utf-8-sig").splitlines()
            data_yaml = archive.read("comb/data.yaml").decode()
            image_bytes = archive.read("comb/train_ready/images/good_comb.jpg")

        # correct + label -> train_ready, the ORIGINAL photo (not the overlay) with its label
        self.assertEqual(image_bytes, b"good-comb-photo")
        self.assertIn("comb/train_ready/labels/good_comb.txt", names)
        self.assertIn("shank/train_ready/images/good_shank.jpg", names)
        # incorrect / uncertain -> to_relabel with the model prediction as a pre-annotation
        self.assertIn("comb/to_relabel/images/bad_comb.jpg", names)
        self.assertIn("comb/to_relabel/labels/bad_comb.txt", names)
        self.assertIn("shank/to_relabel/images/unsure_shank.jpg", names)
        # correct but unlabelled must NOT be train-ready (YOLO would read it as "nothing here")
        self.assertIn("comb/to_relabel/images/nolabel_comb.jpg", names)
        self.assertNotIn("comb/train_ready/images/nolabel_comb.jpg", names)
        # per-role review: only the reviewed half of "partial"; nothing from "pending"
        self.assertIn("comb/train_ready/images/partial_comb.jpg", names)
        self.assertFalse(any("partial_shank" in name or "pending" in name for name in names))
        # no overlay images, no per-case folders
        self.assertFalse(any(name.endswith(".png") for name in names))
        self.assertTrue({"README.txt", "manifest.csv", "shank/data.yaml"} <= names)
        self.assertIn("0: comb", data_yaml)
        # review notes: worst first (incorrect, then uncertain, then missing labels)
        self.assertIn("bad_comb.jpg", notes[1])
        self.assertIn("遮罩漏掉尖端", notes[1])
        self.assertIn("missing_label", notes[-1])
        self.assertEqual(len(manifest), 9)
        self.assertEqual(
            product_review.bundle_summary(),
            {"train_ready": 3, "to_relabel": 6, "missing_label": 2, "missing_image": 0},
        )

    @staticmethod
    def _review_page_app():
        def page():
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path.cwd() / "src"))
            from product_review_ui import review_cases_page

            review_cases_page(lambda title, description="": None)

        return AppTest.from_function(page, default_timeout=30)

    def test_undo_last_decision_on_pending_queue(self):
        self._record("first", {"comb": 0.4, "shank": 0.5})
        app = self._review_page_app().run()
        app.button(key="decision_comb_不正確_first").click().run()
        self.assertEqual(product_review.get_case("first")["comb_review_state"], "不正確")
        self.assertEqual(len(app.exception), 0, str(app.exception))
        app.button(key="undo_last_review").click().run()
        self.assertEqual(product_review.get_case("first")["comb_review_state"], "待審核")
        self.assertTrue(any("已復原" in item.value for item in app.success))
        self.assertFalse([b for b in app.button if b.key == "undo_last_review"])  # undo is one-shot

    def test_revise_decision_from_gallery(self):
        self._record("first", {"comb": 0.4, "shank": 0.5})
        product_review.review_case_role("first", "comb", "正確", "看起來不錯")
        app = self._review_page_app().run()
        app.segmented_control[0].set_value("正確").run()
        app.radio(key="revise_choice_first_comb").set_value("不正確")
        app.button(key="revise_save_first_comb").click().run()
        case = product_review.get_case("first")
        self.assertEqual(case["comb_review_state"], "不正確")
        self.assertEqual(case["reviewer_note"], "看起來不錯")  # note kept unless edited
        self.assertEqual(len(app.exception), 0, str(app.exception))

        app = self._review_page_app().run()
        app.segmented_control[0].set_value("不正確").run()
        app.radio(key="revise_choice_first_comb").set_value("待審核")
        app.button(key="revise_save_first_comb").click().run()
        self.assertEqual(product_review.get_case("first")["comb_review_state"], "待審核")

    def test_comb_and_shank_can_be_reviewed_independently(self):
        self._record("separate", {"comb": 0.4, "shank": 0.5})
        product_review.review_case_role("separate", "comb", "正確")
        case = product_review.get_case("separate")
        self.assertEqual(case["comb_review_state"], "正確")
        self.assertEqual(case["shank_review_state"], "待審核")
        self.assertEqual(case["review_state"], "待審核")

        product_review.review_case_role("separate", "shank", "不正確")
        case = product_review.get_case("separate")
        self.assertEqual(case["comb_review_state"], "正確")
        self.assertEqual(case["shank_review_state"], "不正確")
        self.assertEqual(case["review_state"], "不正確")

    def test_shank_review_advances_to_next_shank(self):
        self._record("first", {"comb": 0.4, "shank": 0.5})
        self._record("second", {"comb": 0.4, "shank": 0.5})
        shank_items = [
            case["case_id"]
            for case, role in product_review_ui._role_items("待審核")
            if role == "shank"
        ]
        first_id = shank_items[0]
        product_review.review_case_role(first_id, "shank", "正確")
        remaining_shank_items = [
            case["case_id"]
            for case, role in product_review_ui._role_items("待審核")
            if role == "shank"
        ]
        self.assertNotIn(first_id, remaining_shank_items)
        self.assertEqual(len(remaining_shank_items), 1)
        self.assertEqual(len(product_review_ui._role_items("全部")), 4)
        self.assertEqual(len(product_review_ui._role_items("全部", "comb")), 2)
        self.assertEqual(len(product_review_ui._role_items("全部", "shank")), 2)

    def test_quality_capture_failure_never_changes_core_result(self):
        job = SimpleNamespace(snapshot_paths={"comb": str(self.root / "capture" / "comb.png")})
        with patch.object(inference_queue, "record_snapshot_review", side_effect=RuntimeError("db locked")):
            inference_queue._capture_product_review(job, measurements={"weight_g": 2500})

    def test_product_pages_render_and_seed_demo_cases(self):
        app_path = Path(__file__).resolve().parents[1] / "src" / "app.py"
        import ui_style
        with patch.object(database, "DB_PATH", self.root / "grading.db"):
            app = AppTest.from_file(str(app_path), default_timeout=20).run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            for page in ("模型品質監控", "案例審核", "資料與模型維護"):
                app.segmented_control(key="main_page_navigation_segmented").set_value(page).run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
            # demo buttons are hidden in the product …
            self.assertFalse(any(button.label == "載入五筆真實測試案例" for button in app.button))
            # The demo photos live in data/product_review/demo_assets (not in git), so a fresh clone (e.g. the
            # Orin) has only the page checks above; the demo-seeding part runs where the photos exist.
            if not (product_review.DEMO_ASSET_DIR / "DEMO-REAL-01_comb_source.jpg").is_file():
                del app
                gc.collect()
                self.skipTest("demo photos not present (data/product_review/demo_assets is not in git)")
            with patch.object(ui_style, "SHOW_DEMO_TOOLS", True):  # … and shown with CHICKEN_SHOW_DEMO_TOOLS=1
                app.run()
            with patch.object(ui_style, "SHOW_DEMO_TOOLS", True):
                next(button for button in app.button if button.label == "載入五筆真實測試案例").click().run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(product_review.review_summary()["總案例"], 5)
            app.segmented_control(key="main_page_navigation_segmented").set_value("案例審核").run()
            next(control for control in app.segmented_control if control.label == "審核篩選").set_value("全部").run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertTrue(any(selectbox.label == "頁數" for selectbox in app.selectbox))
            del app
            gc.collect()


if __name__ == "__main__":
    unittest.main()
