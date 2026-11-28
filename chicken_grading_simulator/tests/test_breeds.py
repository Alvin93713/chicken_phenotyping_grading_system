import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import database  # noqa: E402
from grading import grade_and_store_with_latest_config  # noqa: E402


class BreedTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db_path = Path(self.folder.name) / "grading.db"
        self.db = patch.object(database, "DB_PATH", self.db_path)
        self.db.start()

    def tearDown(self):
        self.db.stop()
        self.folder.cleanup()

    def test_legacy_rows_and_thresholds_migrate_to_default_breed(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE chickens (id INTEGER PRIMARY KEY AUTOINCREMENT, chicken_id TEXT UNIQUE NOT NULL,
            weight_g REAL, comb_area_cm2 REAL, shank_width_cm REAL, shank_length_cm REAL, selected_features TEXT,
            weight_threshold REAL, comb_area_threshold REAL, shank_width_threshold REAL, shank_length_threshold REAL,
            weight_standby_threshold REAL, comb_area_standby_threshold REAL, shank_width_standby_threshold REAL,
            shank_length_standby_threshold REAL, grade_result TEXT, light_result TEXT, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, graded_at TEXT, final_scanned_at TEXT)""")
        conn.execute("""INSERT INTO chickens (chicken_id, weight_g, selected_features, weight_threshold,
            weight_standby_threshold, grade_result, created_at, updated_at, graded_at)
            VALUES ('C001', 3000, 'weight_g', 2800, 2500, '通過', 't', 't', '2026-01-01 00:00:00')""")
        conn.commit()
        conn.close()
        database.init_db()
        self.assertEqual(database.get_chicken("C001")["breed"], database.DEFAULT_BREED)
        self.assertEqual(database.list_breeds(), [database.DEFAULT_BREED])
        history = database.breed_threshold_history(database.DEFAULT_BREED)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["weight_threshold"], 2800)
        _, pass_thresholds, _ = database.get_latest_grading_config()
        self.assertEqual(pass_thresholds["weight_g"], 2800)

    def test_each_breed_keeps_its_own_thresholds_and_records(self):
        database.init_db()
        database.add_breed("黑羽")
        with self.assertRaises(ValueError):
            database.add_breed("黑羽")
        database.add_chicken("R1", 3000, 10, 1, 9)  # current breed = default
        database.set_current_breed("黑羽")
        database.add_chicken("B1", 2000, 5, 1, 8)
        self.assertEqual(database.get_chicken("B1")["breed"], "黑羽")
        self.assertEqual([c["chicken_id"] for c in database.get_all_chickens("黑羽")], ["B1"])

        database.save_breed_thresholds(database.DEFAULT_BREED, ["weight_g"], {"weight_g": 2800}, {"weight_g": 2500})
        database.save_breed_thresholds("黑羽", ["weight_g"], {"weight_g": 1800}, {"weight_g": 1500}, graded_count=1)
        database.save_breed_thresholds("黑羽", ["weight_g"], {"weight_g": 2100}, {"weight_g": 1500}, graded_count=1)
        self.assertEqual(len(database.breed_threshold_history("黑羽")), 2)  # history is appended, not replaced
        self.assertEqual(database.get_latest_grading_config("黑羽")[1]["weight_g"], 2100)
        self.assertEqual(database.get_latest_grading_config(database.DEFAULT_BREED)[1]["weight_g"], 2800)

        # Grading uses the chicken's own breed, not the current one.
        grade, _ = grade_and_store_with_latest_config({"chicken_id": "B1", "weight_g": 2000})
        self.assertEqual(grade, "備用")
        grade, _ = grade_and_store_with_latest_config({"chicken_id": "R1", "weight_g": 3000})
        self.assertEqual(grade, "通過")

        stats = database.breed_stats("黑羽")
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["features"]["weight_g"]["mean"], 2000)
        self.assertEqual(database.breed_stats()["count"], 2)

    def test_delete_breed_rules(self):
        database.init_db()
        with self.assertRaises(ValueError):  # the last breed stays
            database.delete_breed(database.DEFAULT_BREED)
        database.add_breed("黑羽")
        database.set_current_breed("黑羽")
        database.add_chicken("B1", 2000, 5, 1, 8)
        database.save_breed_thresholds("黑羽", ["weight_g"], {"weight_g": 1800}, {"weight_g": 1500})
        database.set_current_breed(database.DEFAULT_BREED)
        database.add_chicken("R1", 3000, 10, 1, 9)
        self.assertEqual(database.delete_breed("黑羽"), 1)  # its chickens go with it
        self.assertIsNone(database.get_chicken("B1"))
        self.assertIsNotNone(database.get_chicken("R1"))  # other breeds untouched
        self.assertEqual(database.list_breeds(), [database.DEFAULT_BREED])
        self.assertEqual(database.get_current_breed(), database.DEFAULT_BREED)  # current falls back
        self.assertEqual(len(database.breed_threshold_history("黑羽")), 1)  # history kept

        # A deleted default breed is not re-created on the next start.
        database.add_breed("烏骨雞")
        database.delete_breed(database.DEFAULT_BREED)
        database._INITIALIZED_DB_PATHS.clear()
        database.init_db()
        self.assertEqual(database.list_breeds(), ["烏骨雞"])
        self.assertEqual(database.get_current_breed(), "烏骨雞")

    def test_other_breed_chicken_is_refused_by_current_breed_pages(self):
        database.init_db()
        database.add_chicken("R1", 3000, 10, 1, 9)
        database.add_breed("紅羽_G06")
        self.assertEqual(database.breed_mismatch_message("R1"), "")
        database.set_current_breed("紅羽_G06")
        self.assertIn("紅羽_G06", database.breed_mismatch_message("R1"))
        self.assertEqual(database.breed_mismatch_message("NEW1"), "")  # a new ID joins the current breed

    def test_csv_breed_column(self):
        database.init_db()
        csv = ("chicken_id,breed,weight_g,comb_area_cm2,shank_width_cm\n"
               "X1,烏骨雞,1500,3,1\nX2,,1600,3,1\n")
        database.import_csv(io.StringIO(csv))
        self.assertIn("烏骨雞", database.list_breeds())
        self.assertEqual(database.get_chicken("X1")["breed"], "烏骨雞")
        self.assertEqual(database.get_chicken("X2")["breed"], database.DEFAULT_BREED)


if __name__ == "__main__":
    unittest.main()
