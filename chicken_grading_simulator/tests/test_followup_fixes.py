"""Tests for the follow-up review fixes: snapshot races, per-image notes, review filters,
provider server hardening (path traversal, replay, signed site ID) and the vendor export of snapshots."""
import os as _os
_os.environ["CHICKEN_SYNC_DISABLE_WORKER"] = "1"  # never start the real background uploader in tests
import hashlib
import hmac
import importlib
import io
import json
import sys
import tempfile
import time
import types
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import product_review  # noqa: E402
import secure_sync  # noqa: E402
import test_secure_sync  # noqa: E402
from test_secure_sync import _FakeProvider  # noqa: E402


class SnapshotRaceTests(unittest.TestCase):
    """Borrows SecureSyncTests' temp review DB + sync settings (without re-running its tests)."""

    setUp = test_secure_sync.SecureSyncTests.setUp
    tearDown = test_secure_sync.SecureSyncTests.tearDown

    def test_press_during_packing_does_not_revive_the_old_snapshot(self):
        product_review.review_case_role("case-1", "comb", "正確")
        first = secure_sync.request_snapshot()
        real_write = product_review.write_review_bundle
        second = {}

        def write_and_press_again(*args, **kwargs):
            summary = real_write(*args, **kwargs)
            second["id"] = secure_sync.request_snapshot()  # operator presses「回傳」while packing
            return summary

        provider = _FakeProvider()
        with patch.object(product_review, "write_review_bundle", side_effect=write_and_press_again):
            self.assertFalse(secure_sync.process_next_snapshot(session=provider))
        self.assertEqual(provider.calls, [])  # nothing of the replaced snapshot was sent
        self.assertEqual(secure_sync._job(first)["status"], "superseded")
        self.assertFalse(Path(secure_sync.outgoing_dir() / f"{first}.zip").exists())
        self.assertEqual(secure_sync._job(second["id"])["status"], "pending")

    def test_locked_old_zip_does_not_break_a_new_press(self):
        product_review.review_case_role("case-1", "comb", "正確")
        secure_sync.request_snapshot()
        locked = secure_sync.outgoing_dir()
        locked.mkdir(parents=True, exist_ok=True)
        with closing(product_review.get_connection()) as connection, connection:
            connection.execute("UPDATE snapshot_uploads SET file_path = ?", (str(locked / "old.zip"),))
        (locked / "old.zip").write_bytes(b"zip")
        with patch.object(Path, "unlink", side_effect=PermissionError("WinError 32")):
            new_id = secure_sync.request_snapshot()  # used to raise and lose the request
        self.assertEqual(secure_sync._job(new_id)["status"], "pending")
        secure_sync._cleanup_superseded()  # the worker removes it once it is no longer open
        self.assertFalse((locked / "old.zip").exists())

    def test_signature_covers_site_id(self):
        config = secure_sync.SyncConfig.from_environment()
        headers = secure_sync._signed_headers(config, "https://provider.example/v1/ping", "d" * 64)
        self.assertEqual(headers["X-Signature-Version"], "2")
        expected = secure_sync._canonical_signature(
            config.hmac_secret, "/v1/ping", headers["X-Timestamp"], headers["X-Nonce"], "d" * 64, "farm-a")
        self.assertEqual(headers["X-Signature"], expected)
        other_site = secure_sync._canonical_signature(
            config.hmac_secret, "/v1/ping", headers["X-Timestamp"], headers["X-Nonce"], "d" * 64, "farm-b")
        self.assertNotEqual(headers["X-Signature"], other_site)

    def test_each_image_keeps_its_own_note(self):
        product_review.review_case_role("case-1", "comb", "不正確", "mask misses comb tip")
        product_review.review_case_role("case-1", "shank", "正確", "")
        case = product_review.get_case("case-1")
        self.assertEqual(product_review.role_note(case, "comb"), "mask misses comb tip")
        self.assertEqual(product_review.role_note(case, "shank"), "")
        buffer = io.BytesIO()
        product_review.write_review_bundle(buffer)
        with zipfile.ZipFile(buffer) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        comb_rows = [row for row in manifest if row["role"] == "comb"]
        self.assertEqual(comb_rows[0]["reviewer_note"], "mask misses comb tip")

    def test_vendor_export_reads_snapshots(self):
        product_review.review_case_role("case-1", "comb", "正確", "ok")
        secure_sync.request_snapshot()
        provider = _FakeProvider()
        self.assertTrue(secure_sync.process_next_snapshot(session=provider))
        inbox = self.root / "inbox" / "farm-a_20261126"
        inbox.mkdir(parents=True)
        (inbox / "orin-001_103000.zip").write_bytes(provider.stored)
        sys.path.insert(0, str(ROOT / "scripts"))
        export_tool = importlib.import_module("provider_export_dataset")
        out = self.root / "dataset"
        self.assertEqual(export_tool.export(self.root / "inbox", out), 1)
        images = list((out / "comb" / "train_ready" / "images").iterdir())
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].name.startswith("farm-a_orin-001_case-1_comb"))
        self.assertTrue((out / "comb" / "train_ready" / "labels" / (images[0].stem + ".txt")).is_file())


# --- provider server -------------------------------------------------------------------------------
class _HTTPException(Exception):
    def __init__(self, status_code, detail=""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _FastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def post(self, *args, **kwargs):
        return lambda function: function


def _load_server(temp: Path):
    fake = types.ModuleType("fastapi")
    fake.FastAPI, fake.HTTPException, fake.Request = _FastAPI, _HTTPException, object
    env = {"CHICKEN_PROVIDER_DATA_DIR": str(temp / "inbox"), "CHICKEN_PROVIDER_DB_PATH": str(temp / "db" / "r.db"),
           "CHICKEN_PROVIDER_API_KEY": "api-key", "CHICKEN_PROVIDER_HMAC_SECRET": "s" * 32}
    with patch.dict(sys.modules, {"fastapi": fake}), patch.dict(_os.environ, env):
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.modules.pop("provider_sync_server", None)
        server = importlib.import_module("provider_sync_server")
    return server, env


class _Request:
    def __init__(self, headers, path):
        self.headers = headers
        self.url = types.SimpleNamespace(path=path, query="")


class ProviderServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server, env = _load_server(Path(self.temp.name))
        self.env = patch.dict(_os.environ, env)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def _signed(self, body=b"", site="farm-a", version="2", nonce="n1"):
        timestamp = str(int(time.time()))
        digest = hashlib.sha256(body).hexdigest()
        canonical = f"POST\n/v1/ping\n{timestamp}\n{nonce}\n{digest}" + (f"\n{site}" if version == "2" else "")
        headers = {"Authorization": "Bearer api-key", "X-Timestamp": timestamp, "X-Nonce": nonce,
                   "X-Content-SHA256": digest, "X-Site-ID": site,
                   "X-Signature": hmac.new(b"s" * 32, canonical.encode(), hashlib.sha256).hexdigest()}
        if version:
            headers["X-Signature-Version"] = version
        return headers

    def test_dot_identifiers_and_traversal_are_rejected(self):
        for bad in (".", "..", ".hidden", "../x", "a/b"):
            with self.assertRaises(_HTTPException):
                self.server._safe_identifier(bad, "upload ID")
        self.assertEqual(self.server._safe_identifier("20261126-ab.cd", "upload ID"), "20261126-ab.cd")
        with self.assertRaises(_HTTPException):
            self.server._staging_folder("dev", "..")
        self.assertEqual(self.server._staging_folder("dev", "u1").name, "u1")

    def test_replayed_nonce_is_rejected(self):
        request = _Request(self._signed(nonce="same"), "/v1/ping")
        self.server._verify_signed_request(request, "orin-001", b"")
        with self.assertRaises(_HTTPException) as caught:
            self.server._verify_signed_request(_Request(self._signed(nonce="same"), "/v1/ping"), "orin-001", b"")
        self.assertEqual(caught.exception.status_code, 409)

    def test_site_id_is_signed(self):
        headers = self._signed(site="farm-a", nonce="n2")
        headers["X-Site-ID"] = "someone-else"  # swapped after signing
        with self.assertRaises(_HTTPException) as caught:
            self.server._verify_signed_request(_Request(headers, "/v1/ping"), "orin-001", b"")
        self.assertEqual(caught.exception.status_code, 401)

    def test_device_can_only_upload_to_its_registered_site(self):
        with patch.dict(_os.environ, {"CHICKEN_PROVIDER_SITE_ID": "farm-a"}):
            self.server._verify_signed_request(_Request(self._signed(site="farm-a", nonce="s1"), "/v1/ping"),
                                               "orin-001", b"")
            with self.assertRaises(_HTTPException) as caught:  # correctly signed, but another farm's folder
                self.server._verify_signed_request(_Request(self._signed(site="farm-b", nonce="s2"), "/v1/ping"),
                                                   "orin-001", b"")
            self.assertEqual(caught.exception.status_code, 403)

    def test_legacy_signatures_need_explicit_opt_in(self):
        legacy = self._signed(version=None, nonce="n3")
        with self.assertRaises(_HTTPException):
            self.server._verify_signed_request(_Request(legacy, "/v1/ping"), "orin-001", b"")
        with patch.object(self.server, "ACCEPT_LEGACY_SIGNATURES", True):
            self.server._verify_signed_request(_Request(self._signed(version=None, nonce="n4"), "/v1/ping"),
                                               "orin-001", b"")


class ProgressRingTests(unittest.TestCase):
    def test_ring_follows_job_progress(self):
        import dual_camera_ui
        import ui_style
        self.assertEqual(dual_camera_ui.ring_percent({"進度": "-"}), 40)   # queued: photos taken
        self.assertEqual(dual_camera_ui.ring_percent({"進度": "0/2"}), 40)
        self.assertEqual(dual_camera_ui.ring_percent({"進度": "1/2"}), 60)  # comb done
        self.assertEqual(dual_camera_ui.ring_percent({"進度": "2/2"}), 80)  # shank done, saving
        ring = ui_style.progress_ring_html("C<1>", 60, 40)
        self.assertIn('style="--ring-p:40;--ring-pi:40" data-job="" data-continue="" data-floor="60" data-from="40"', ring)
        self.assertIn("C&lt;1&gt;", ring)
        self.assertNotIn("pr-step", ring)  # only the circular ring, no stage list
        self.assertIn("推論腳脛長寬", ring)  # the stage now running is named inside the ring


class ReviewFilterTests(unittest.TestCase):
    def test_clicking_the_selected_segment_keeps_the_choice(self):
        import product_review_ui
        state = {}
        with patch.object(product_review_ui.st, "session_state", state):
            state["role"] = "SHANK"
            product_review_ui._keep_selection("role", "COMB")
            state["role"] = None  # clicked the highlighted button again
            product_review_ui._keep_selection("role", "COMB")
        self.assertEqual(state["role"], "SHANK")


if __name__ == "__main__":
    unittest.main()
