import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"  # never start the real background uploader in tests
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import product_review
import secure_sync


class _Reply:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class _FakeProvider:
    """In-process stand-in for provider_sync_server's snapshot protocol."""

    def __init__(self, fail_on_part=None, complete_status=None):
        self.parts = {}
        self.calls = []
        self.fail_on_part = fail_on_part
        self.complete_status = complete_status
        self.stored = None

    def post(self, url, data=b"", headers=None, timeout=None, verify=None):
        path = urlsplit(url).path
        self.calls.append(path)
        assert headers["X-Content-SHA256"] == hashlib.sha256(data).hexdigest()
        assert headers["X-Signature"] and headers["Authorization"].startswith("Bearer ")
        assert verify is True
        if "/parts/" in path:
            index = int(path.rsplit("/", 1)[1])
            if self.fail_on_part == index:
                self.fail_on_part = None  # fail once, like a dropped connection
                raise requests.ConnectionError("connection reset")
            self.parts[index] = data
            return _Reply(200, {"received": index})
        if path.endswith("/complete"):
            if self.complete_status:
                status, self.complete_status = self.complete_status, None
                return _Reply(status, {"detail": "Missing parts"})
            details = json.loads(data)
            blob = b"".join(self.parts[i] for i in range(details["parts"]))
            assert len(blob) == details["size"]
            assert hashlib.sha256(blob).hexdigest() == details["sha256"]
            self.stored = blob
            return _Reply(201, {"stored_as": "farm-a_20261125/orin-001_103000.zip"})
        return _Reply(404)


class SecureSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.review_dir = self.root / "review"
        self.db_path = self.review_dir / "review.db"
        self.image = self.root / "comb.jpg"
        self.image.write_bytes(b"compressed-image" * 50)
        self.patches = (
            patch.object(product_review, "REVIEW_DIR", self.review_dir),
            patch.object(product_review, "REVIEW_DB_PATH", self.db_path),
            patch.dict(
                os.environ,
                {
                    "CHICKEN_SYNC_ENDPOINT": "https://provider.example/v1/review-packages",
                    "CHICKEN_SYNC_DEVICE_ID": "orin-001",
                    "CHICKEN_SYNC_SITE_ID": "farm-a",
                    "CHICKEN_SYNC_API_KEY": "api-key",
                    "CHICKEN_SYNC_HMAC_SECRET": "x" * 32,
                },
                clear=False,
            ),
        )
        for item in self.patches:
            item.start()
        result = self.root / "comb.png"
        result.write_bytes(b"overlay")
        (self.root / "comb.txt").write_bytes(b"0 0.1 0.1 0.5 0.1 0.3 0.6\n")
        product_review.record_case(
            job_id="case-1",
            chicken_id="C001",
            pipeline_status="已完成",
            confidence_threshold=0.25,
            role_confidences={"comb": 0.4, "shank": 0.8},
            measurements={"weight_g": 2500, "comb_area_cm2": 3.2},
            source_paths={"comb": str(self.image)},
            result_paths={"comb": str(result)},
            models={"comb": "best_comb.pt"},
            force=True,
        )

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def _set_due(self):
        with closing(product_review.get_connection()) as connection, connection:
            connection.execute("UPDATE snapshot_uploads SET next_attempt_at = '2000-01-01 00:00:00'")

    # --- snapshot upload -----------------------------------------------------
    def test_reviewing_alone_uploads_nothing(self):
        product_review.review_case_role("case-1", "comb", "正確")
        status = secure_sync.snapshot_status()
        self.assertIsNone(status["active"])
        self.assertIsNone(status["last_done"])
        self.assertFalse(secure_sync.process_next_snapshot(session=_FakeProvider()))

    def test_snapshot_contains_every_reviewed_image_and_hides_chicken_ids(self):
        product_review.review_case_role("case-1", "comb", "正確", "ok")
        secure_sync.request_snapshot()
        provider = _FakeProvider()
        self.assertTrue(secure_sync.process_next_snapshot(session=provider))
        with zipfile.ZipFile(io.BytesIO(provider.stored)) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            info = json.loads(archive.read("snapshot_info.json"))
        self.assertIn("comb/train_ready/images/case-1_comb.jpg", names)
        self.assertIn("comb/train_ready/labels/case-1_comb.txt", names)
        self.assertEqual(manifest[0]["decision"], "correct")
        self.assertNotEqual(manifest[0]["chicken_id"], "C001")  # pseudonymised for the network
        self.assertNotIn(b"C001", provider.stored)
        self.assertEqual((info["site_id"], info["device_id"]), ("farm-a", "orin-001"))
        done = secure_sync.snapshot_status()["last_done"]
        self.assertEqual(done["summary"]["train_ready"], 1)
        self.assertEqual(done["stored_as"], "farm-a_20261125/orin-001_103000.zip")
        self.assertFalse(list((self.review_dir / "outgoing").glob("*.zip")))  # local copy removed
        # the manual on-site download keeps real IDs
        with zipfile.ZipFile(io.BytesIO(product_review.export_review_bundle())) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))[0]["chicken_id"], "C001")

    def test_every_press_sends_everything_again(self):
        product_review.review_case_role("case-1", "comb", "正確")
        for _ in range(2):
            secure_sync.request_snapshot()
            provider = _FakeProvider()
            self.assertTrue(secure_sync.process_next_snapshot(session=provider))
            with zipfile.ZipFile(io.BytesIO(provider.stored)) as archive:
                self.assertIn("comb/train_ready/images/case-1_comb.jpg", archive.namelist())

    def test_upload_resumes_from_the_failed_part(self):
        product_review.review_case_role("case-1", "comb", "不正確")
        with patch.object(secure_sync, "PART_SIZE_BYTES", 256):
            secure_sync.request_snapshot()
            provider = _FakeProvider(fail_on_part=2)
            self.assertFalse(secure_sync.process_next_snapshot(session=provider))
            active = secure_sync.snapshot_status()["active"]
            self.assertEqual((active["status"], active["parts_done"]), ("failed", 2))
            self.assertIn("無法連線", secure_sync.describe_error(active["last_error"]))
            self.assertGreater(active["parts_total"], 3)
            self.assertFalse(secure_sync.process_next_snapshot(session=provider))  # waits for back-off
            self._set_due()
            first_new_part = len(provider.calls)
            self.assertTrue(secure_sync.process_next_snapshot(session=provider))
        resent = [call for call in provider.calls[first_new_part:] if "/parts/" in call]
        self.assertTrue(resent[0].endswith("/parts/2"))  # resumed, parts 0-1 not re-sent
        self.assertIsNone(secure_sync.snapshot_status()["active"])

    def test_server_missing_parts_triggers_full_resend(self):
        product_review.review_case_role("case-1", "comb", "正確")
        secure_sync.request_snapshot()
        provider = _FakeProvider(complete_status=409)
        self.assertFalse(secure_sync.process_next_snapshot(session=provider))
        self.assertEqual(secure_sync.snapshot_status()["active"]["parts_done"], 0)
        self._set_due()
        self.assertTrue(secure_sync.process_next_snapshot(session=provider))

    def test_new_press_replaces_unfinished_snapshot(self):
        product_review.review_case_role("case-1", "comb", "正確")
        with patch.object(secure_sync, "PART_SIZE_BYTES", 256):
            first = secure_sync.request_snapshot()
            secure_sync.process_next_snapshot(session=_FakeProvider(fail_on_part=1))
            old_file = Path(secure_sync._job(first)["file_path"])
            self.assertTrue(old_file.is_file())
            second = secure_sync.request_snapshot()
        self.assertEqual(secure_sync._job(first)["status"], "superseded")
        self.assertFalse(old_file.exists())
        self.assertEqual(secure_sync.snapshot_status()["active"]["upload_id"], second)

    def test_long_outage_keeps_retrying_hourly(self):
        product_review.review_case_role("case-1", "comb", "正確")
        secure_sync.request_snapshot()
        with patch.object(secure_sync, "_build", side_effect=requests.ConnectionError("down")):
            for _ in range(secure_sync.STALLED_AFTER_ATTEMPTS + 3):
                self._set_due()
                self.assertFalse(secure_sync.process_next_snapshot(session=_FakeProvider()))
        active = secure_sync.snapshot_status()["active"]
        self.assertTrue(active["stalled"])
        delay = secure_sync.datetime.strptime(active["next_attempt_at"], "%Y-%m-%d %H:%M:%S") - secure_sync._now()
        self.assertLessEqual(delay.total_seconds(), secure_sync.MAX_RETRY_DELAY_SECONDS + 1)
        self.assertGreater(delay.total_seconds(), secure_sync.MAX_RETRY_DELAY_SECONDS - 60)
        self.assertTrue(secure_sync.retry_now())
        self.assertTrue(secure_sync.process_next_snapshot(session=_FakeProvider()))  # heals when back

    def test_plain_http_endpoint_is_rejected(self):
        config = secure_sync.SyncConfig.from_environment()
        insecure = secure_sync.SyncConfig(**{**config.__dict__, "endpoint": "http://provider.example"})
        self.assertIn("HTTPS", insecure.validation_error())

    def test_snapshot_urls_sit_next_to_the_configured_endpoint(self):
        self.assertEqual(
            secure_sync.snapshot_url("https://h/v1/review-packages", "abc", "parts/3"), "https://h/v1/snapshots/abc/parts/3"
        )
        self.assertEqual(secure_sync.ping_url("https://h/v1/review-packages"), "https://h/v1/ping")

    # --- configuration from the UI --------------------------------------------
    def _without_env(self):
        return patch.dict(os.environ, {name: "" for name in secure_sync.ENV_NAMES.values()})

    def test_setup_code_round_trip_and_saved_config(self):
        values = {
            "endpoint": "https://vendor.example/v1/review-packages",
            "device_id": "orin-002",
            "site_id": "farm-b",
            "api_key": "k" * 20,
            "hmac_secret": "s" * 40,
        }
        code = secure_sync.encode_setup_code(values)
        self.assertTrue(code.startswith("CHKSYNC1:"))
        decoded = secure_sync.decode_setup_code("  " + code[:20] + "\n" + code[20:] + "  ")
        self.assertEqual({k: decoded[k] for k in values}, values)
        with self._without_env():
            self.assertTrue(secure_sync.SyncConfig.from_environment().validation_error())
            secure_sync.save_config(decoded)
            config = secure_sync.SyncConfig.from_environment()
            self.assertEqual(config.validation_error(), "")
            self.assertEqual(config.device_id, "orin-002")
            with patch.dict(os.environ, {"CHICKEN_SYNC_DEVICE_ID": "env-wins"}):
                self.assertEqual(secure_sync.SyncConfig.from_environment().device_id, "env-wins")
            secure_sync.clear_saved_config()
            self.assertTrue(secure_sync.SyncConfig.from_environment().validation_error())

    def test_invalid_setup_codes_are_rejected_clearly(self):
        for code in ("", "hello", "CHKSYNC1:not-base64!!"):
            with self.assertRaises(ValueError):
                secure_sync.decode_setup_code(code)
        incomplete = secure_sync.encode_setup_code(
            {"endpoint": "https://x/v1/review-packages", "device_id": "d", "site_id": "s", "api_key": "a", "hmac_secret": "h"}
        )
        with self._without_env(), self.assertRaises(ValueError):
            secure_sync.save_config(secure_sync.decode_setup_code(incomplete))  # secret too short

    def test_friendly_status_explains_missing_settings(self):
        with self._without_env():
            message = secure_sync.SyncConfig.from_environment().friendly_status()
        self.assertIn("連線設定碼", message)

    def test_check_connection_maps_responses(self):
        class _PingSession:
            def __init__(self, reply):
                self.reply, self.calls = reply, []

            def post(self, url, **kwargs):
                self.calls.append((url, kwargs))
                return self.reply

        session = _PingSession(_Reply(200))
        ok, _ = secure_sync.check_connection(session=session)
        self.assertTrue(ok)
        self.assertEqual(session.calls[0][0], "https://provider.example/v1/ping")
        self.assertIn("X-Signature", session.calls[0][1]["headers"])
        ok, message = secure_sync.check_connection(session=_PingSession(_Reply(401, {"detail": "Invalid credentials"})))
        self.assertFalse(ok)
        self.assertIn("認證失敗", message)
        ok, message = secure_sync.check_connection(session=_PingSession(_Reply(401, {"detail": "Expired request"})))
        self.assertIn("時間", message)

    # --- vendor tools ------------------------------------------------------------
    def test_vendor_issued_code_is_accepted_by_the_farm(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import provider_issue_device

        credentials = self.root / "creds.json"
        code = provider_issue_device.issue("https://vendor.example/v1/review-packages", "farm-c", "orin-009", credentials)
        decoded = secure_sync.decode_setup_code(code)
        stored = json.loads(credentials.read_text(encoding="utf-8"))["orin-009"]
        self.assertEqual(decoded["api_key"], stored["api_key"])
        self.assertEqual(decoded["hmac_secret"], stored["hmac_secret"])
        self.assertEqual(secure_sync.SyncConfig.from_values(decoded).validation_error(), "")

    def test_yolo_label_helpers(self):
        import training_labels

        text = training_labels.format_yolo_seg([0, 0], [[[0.1, 0.2], [0.5, 0.2], [0.3, 1.2]], [[0.1, 0.1], [0.2, 0.2]]])
        self.assertEqual(text, "0 0.100000 0.200000 0.500000 0.200000 0.300000 1.000000\n")  # clipped, 2-point dropped
        self.assertEqual(training_labels.label_path_for("out/job/samples/comb.png"), Path("out/job/samples/comb.txt"))

        class _Tensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        fake = type("R", (), {
            "boxes": type("B", (), {"cls": _Tensor([0.0])})(),
            "masks": type("M", (), {"xyn": [[[0.1, 0.1], [0.9, 0.1], [0.5, 0.8]]]})(),
        })()
        written = training_labels.write_result_labels(fake, self.root / "shank.png")
        self.assertEqual(written.read_text(encoding="ascii").split()[0], "0")
        self.assertIsNone(training_labels.write_result_labels(object(), self.root / "shank.png"))  # never raises
        training_labels.save_class_names(self.review_dir, "shank", {0: "shank"})
        self.assertEqual(training_labels.load_class_names(self.review_dir, "shank"), {0: "shank"})


if __name__ == "__main__":
    unittest.main()
