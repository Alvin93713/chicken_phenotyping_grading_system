import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"  # never start the real background uploader in tests
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit import config as st_config
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import database  # noqa: E402
import product_review  # noqa: E402
import ui_style  # noqa: E402


class UIStyleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.prefs = Path(self.folder.name) / "ui_style.json"
        self.env = patch.dict(_os.environ, {"CHICKEN_UI_PREFS_PATH": str(self.prefs)})
        self.env.start()
        # never touch the real review database (data/product_review) from these AppTests
        self.review_db = patch.object(product_review, "REVIEW_DB_PATH", Path(self.folder.name) / "review.db")
        self.review_db.start()

    def tearDown(self):
        self.env.stop()
        self.review_db.stop()
        ui_style._ensure_streamlit_theme(ui_style.THEME_DARK)
        gc.collect()
        self.folder.cleanup()

    def test_default_is_original_look(self):
        self.assertEqual(ui_style.load_prefs(), {"style": "原版", "theme": "深色", "lang": "中文"})
        # 深色 is exactly .streamlit/config.toml, so the original look never changes.
        config_text = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        for name, value in ui_style.STREAMLIT_THEMES["深色"].items():
            self.assertIn(f'{name} = "{value}"', config_text)

    def test_invalid_prefs_fall_back_to_default(self):
        self.prefs.write_text('{"style": "x", "theme": 3}', encoding="utf-8")
        self.assertEqual(ui_style.load_prefs(), ui_style.DEFAULT_PREFS)
        self.prefs.write_text("not json", encoding="utf-8")
        self.assertEqual(ui_style.load_prefs(), ui_style.DEFAULT_PREFS)

    def test_result_band_html(self):
        band = ui_style.result_band_html("C<1>", "通過", [("重量 (g)", "2150.4", "weight"), ("雞冠面積 (cm²)", "-", "comb")])
        self.assertIn("mo-pass", band)
        self.assertIn("C&lt;1&gt;", band)
        self.assertIn("mo-n2", band)
        self.assertIn(">cm²<", band)
        self.assertEqual(band.count('class="mo-card mo-cd'), 2)

    def test_switching_style_and_theme_and_restoring(self):
        with patch.object(database, "DB_PATH", Path(self.folder.name) / "grading.db"):
            app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(st_config.get_option("theme.base"), "dark")

            app.segmented_control(key="ui_style_choice").set_value("動態").run()
            app.segmented_control(key="ui_theme_choice").set_value("米白").run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(json.loads(self.prefs.read_text(encoding="utf-8")),
                             {"style": "動態", "theme": "米白", "lang": "中文"})
            self.assertEqual(st_config.get_option("theme.base"), "light")
            self.assertEqual(st_config.get_option("theme.backgroundColor"), "#f6f1e7")
            self.assertTrue(any("mo-card" in m.value for m in app.markdown))

            for page in ("掃描 QR code 與表型輸入", "再次掃描與亮燈判定", "案例審核", "資料與模型維護"):
                app.segmented_control(key="main_page_navigation_segmented").set_value(page).run()
                self.assertEqual(len(app.exception), 0, str(app.exception))

            app.button(key="ui_restore_classic").click().run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(ui_style.load_prefs(), ui_style.DEFAULT_PREFS)
            self.assertEqual(st_config.get_option("theme.base"), "dark")
            self.assertEqual(st_config.get_option("theme.backgroundColor"), "#0b1018")
            self.assertFalse(any("mo-card" in m.value for m in app.markdown))
            del app

    def test_language_switch(self):
        with patch.object(database, "DB_PATH", Path(self.folder.name) / "grading.db"):
            app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=30).run()
            nav = app.segmented_control(key="main_page_navigation_segmented")
            self.assertIn("硬體連接設定", nav.options)

            app.segmented_control(key="ui_lang_choice").set_value("English").run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(ui_style.load_prefs()["lang"], "English")
            nav = app.segmented_control(key="main_page_navigation_segmented")
            self.assertNotIn("硬體連接設定", nav.options)  # displayed in English
            nav.set_value("案例審核").run()  # values stay Chinese
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(app.session_state["main_page_navigation"], "案例審核")

            app.segmented_control(key="ui_lang_choice").set_value("中文").run()
            self.assertIn("硬體連接設定", app.segmented_control(key="main_page_navigation_segmented").options)
            del app

    def test_t_falls_back_and_formats(self):
        import i18n
        self.assertEqual(i18n.t("通過"), "通過")  # Chinese by default
        with patch.object(i18n, "current_language", return_value=i18n.LANG_EN):
            self.assertEqual(i18n.t("通過"), "Pass")
            self.assertEqual(i18n.t("沒有翻譯的字 {n}", n=3), "沒有翻譯的字 3")
            self.assertIn("沒有翻譯的字 {n}", i18n.missing_translations())


if __name__ == "__main__":
    unittest.main()
