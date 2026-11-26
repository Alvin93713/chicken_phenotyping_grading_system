import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"  # never start the real background uploader in tests
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import database  # noqa: E402
import grading  # noqa: E402
import product_review  # noqa: E402


class DatabaseEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db = patch.object(database, "DB_PATH", Path(self.folder.name) / "grading.db")
        self.db.start()
        database.init_db()

    def tearDown(self):
        self.db.stop()
        self.folder.cleanup()

    def test_init_db_runs_schema_once_per_database(self):
        with patch.object(database, "_init_db_schema") as schema:
            database.init_db()
            database.init_db()
        schema.assert_not_called()  # already initialised in setUp

    def test_csv_import_upserts_in_one_pass_and_is_all_or_nothing(self):
        database.add_chicken("A1", 1000, 1, 1, 1)
        csv = io.StringIO("chicken_id,weight_g,comb_area_cm2,shank_width_cm\nA1,2000,2,2\nB2,1500,1,1\nB2,1600,1,1\n")
        self.assertEqual(database.import_csv(csv), (1, 2))  # B2 inserted, then A1 and the 2nd B2 updated
        self.assertEqual(database.get_chicken("A1")["weight_g"], 2000)
        self.assertEqual(database.get_chicken("B2")["weight_g"], 1600)

        bad = io.StringIO("chicken_id,weight_g,comb_area_cm2,shank_width_cm\nC3,1000,1,1\nC4,-5,1,1\n")
        with self.assertRaises(ValueError):
            database.import_csv(bad)
        self.assertIsNone(database.get_chicken("C3"))  # nothing written when one row is invalid

    def test_batch_grading_matches_single_grading(self):
        features = list(grading.FEATURE_COLUMNS)
        pass_t = {f: 0.0 for f in features}
        standby_t = {f: 0.0 for f in features}
        pass_t["weight_g"], standby_t["weight_g"] = 2000.0, 1500.0
        for cid, weight in (("P", 2100), ("S", 1600), ("F", 900)):
            database.add_chicken(cid, weight, 1, 1, 1)
        enabled = grading.active_features(features, pass_t, standby_t)
        results = []
        for chicken in database.get_all_chickens():
            grade, light, below_pass, below_standby = grading.grade_details(chicken, enabled, pass_t, standby_t)
            self.assertEqual((grade, light), grading.grade_chicken(chicken, features, pass_t, standby_t))
            results.append((chicken["chicken_id"], grade, light))
        database.update_grading_results(results, features, pass_t, standby_t)
        self.assertEqual({c["chicken_id"]: c["grade_result"] for c in database.get_all_chickens()},
                         {"P": "通過", "S": "備用", "F": "不通過"})

    def test_grade_and_store_with_latest_config(self):
        database.add_chicken("X", 1000, 1, 1, 1)
        self.assertEqual(grading.grade_and_store_with_latest_config(database.get_chicken("X")), (None, None))
        features = list(grading.FEATURE_COLUMNS)
        pass_t = {f: 0.0 for f in features}
        standby_t = {f: 0.0 for f in features}
        pass_t["weight_g"] = 500.0
        database.update_grading_result("X", features, pass_t, standby_t, "不通過", "RED")  # saves a config
        self.assertEqual(grading.grade_and_store_with_latest_config(database.get_chicken("X")), ("通過", "GREEN"))
        self.assertEqual(database.get_chicken("X")["grade_result"], "通過")


class CsvUploadLoopTests(unittest.TestCase):
    def test_uploader_is_reset_after_import(self):
        source = (ROOT / "src" / "app.py").read_text(encoding="utf-8-sig")
        # the uploader key carries a version that is bumped after a successful import (no import→rerun loop)
        self.assertIn('key=f"csv_uploader_{uploader_version}"', source)
        self.assertIn("st.session_state.csv_uploader_version = uploader_version + 1", source)


class HotkeyGuardTests(unittest.TestCase):
    def test_every_hotkey_checks_for_typing(self):
        source = (ROOT / "src" / "app.py").read_text(encoding="utf-8-sig")
        handlers = source.count('event.code !== "Key')
        self.assertEqual(handlers, 4)  # r submit, n focus, n click, p cycle
        self.assertEqual(source.count("|| ignoreHotkey(event)"), handlers)


class ReviewSchemaOnceTests(unittest.TestCase):
    def test_review_schema_created_once_per_file(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(product_review, "REVIEW_DIR", Path(folder)), \
             patch.object(product_review, "REVIEW_DB_PATH", Path(folder) / "review.db"):
            with patch.object(product_review, "_ensure_schema", wraps=product_review._ensure_schema) as schema:
                for _ in range(3):
                    product_review.list_cases()
            self.assertEqual(schema.call_count, 1)


class DataPageRenderTests(unittest.TestCase):
    def test_data_page_renders_with_lazy_export(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(_os.environ, {"CHICKEN_UI_PREFS_PATH": _os.devnull}), \
             patch.object(database, "DB_PATH", Path(folder) / "grading.db"), \
             patch.object(product_review, "REVIEW_DB_PATH", Path(folder) / "review.db"):
            database.init_db()
            database.add_chicken("G1", 2100, 1, 1, 1)
            app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=30).run()
            app.segmented_control(key="main_page_navigation_segmented").set_value("雞隻資料管理").run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            # tables show 通過 / 備用 / 不通過 only: no GREEN / YELLOW / RED light column
            columns = list(app.dataframe[0].value.columns)
            self.assertIn("分級結果", columns)
            self.assertNotIn("light_result", columns)
            app.segmented_control(key="main_page_navigation_segmented").set_value("硬體連接設定").run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            app.segmented_control(key="main_page_navigation_segmented").set_value("分級門檻設定").run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            del app


if __name__ == "__main__":
    unittest.main()
