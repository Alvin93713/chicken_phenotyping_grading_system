"""One-click secure upload of a complete review snapshot to the vendor.

When the farm presses「回傳測試影像」, every reviewed image (正確/不正確/無法判斷) is packed into
one fine-tuning bundle (the same layout as the on-site download, with chicken IDs pseudonymised),
then uploaded in signed parts over HTTPS. Packing and uploading run in a background thread; the
upload resumes after network or power loss and keeps retrying (back-off capped at one hour).
Nothing is uploaded unless someone presses the button.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

import model_registry
import product_review
from i18n import is_english, t

MAX_RETRY_DELAY_SECONDS = 3600
STALLED_AFTER_ATTEMPTS = 12  # ~3.4 hours of consecutive failures -> highlighted in the UI
PART_SIZE_BYTES = int(float(os.environ.get("CHICKEN_SYNC_PART_MB", "16")) * 1024 * 1024)
ACTIVE_STATES = ("pending", "uploading", "failed")
_WORKER_LOCK = threading.Lock()
_JOB_LOCK = threading.Lock()
_WAKE = threading.Event()
_WORKER_STARTED = False

CONFIG_FILENAME = "secure_sync_config.json"
SETUP_CODE_PREFIX = "CHKSYNC1:"
REQUIRED_KEYS = ("endpoint", "device_id", "site_id", "api_key", "hmac_secret")
OPTIONAL_KEYS = ("ca_bundle",)
ENV_NAMES = {
    "endpoint": "CHICKEN_SYNC_ENDPOINT",
    "device_id": "CHICKEN_SYNC_DEVICE_ID",
    "site_id": "CHICKEN_SYNC_SITE_ID",
    "api_key": "CHICKEN_SYNC_API_KEY",
    "hmac_secret": "CHICKEN_SYNC_HMAC_SECRET",
    "ca_bundle": "CHICKEN_SYNC_CA_BUNDLE",
}
KEY_LABELS = {
    "endpoint": "上傳網址",
    "device_id": "裝置代號",
    "site_id": "場域代號",
    "api_key": "API 金鑰",
    "hmac_secret": "簽章密鑰",
}


def _now() -> datetime:
    return datetime.now()


def _now_text() -> str:
    return _now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Configuration: saved settings file (written from the UI) + environment overrides
# ---------------------------------------------------------------------------

def config_path() -> Path:
    return product_review.REVIEW_DIR / CONFIG_FILENAME


def load_saved_config() -> dict[str, str]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: str(data.get(key, "") or "").strip() for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS)}


def save_config(values: dict[str, str]) -> None:
    """Persist connection settings locally. The file is git-ignored and readable only by the owner."""
    cleaned = {key: str(values.get(key, "") or "").strip() for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS)}
    error = SyncConfig.from_values(cleaned).validation_error()
    if error:
        raise ValueError(error)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(path)


def clear_saved_config() -> None:
    try:
        config_path().unlink()
    except FileNotFoundError:
        pass


def environment_overrides() -> list[str]:
    """Names of CHICKEN_SYNC_* variables that currently override the saved settings."""
    return [ENV_NAMES[key] for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS) if os.environ.get(ENV_NAMES[key], "").strip()]


def encode_setup_code(values: dict[str, str]) -> str:
    payload = {key: str(values[key]).strip() for key in REQUIRED_KEYS}
    if str(values.get("ca_bundle", "") or "").strip():
        payload["ca_bundle"] = str(values["ca_bundle"]).strip()
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return SETUP_CODE_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _label_list(keys) -> str:
    """Operator-facing names of setting keys, joined for display (called from the UI thread)."""
    return (", " if is_english() else "、").join(t(KEY_LABELS[key]) for key in keys)


def display_validation_error(error: str) -> str:
    """Display text (current UI language) for a validation_error() message; call at render time only."""
    error = str(error or "")
    prefix = "尚未設定："
    if error.startswith(prefix):
        return t("尚未設定：{names}", names=error[len(prefix):])
    return t(error)


def decode_setup_code(code: str) -> dict[str, str]:
    """Accept the provider's one-line setup code (CHKSYNC1:...) or the equivalent JSON.

    Called from the UI thread: error messages are in the current UI language.
    """
    text = "".join(str(code or "").split())
    if not text:
        raise ValueError(t("請貼上廠商提供的連線設定碼"))
    try:
        if text.startswith(SETUP_CODE_PREFIX):
            body = text[len(SETUP_CODE_PREFIX):]
            data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode("utf-8"))
        elif text.startswith("{"):
            data = json.loads(str(code))
        else:
            raise ValueError
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(t("連線設定碼格式不正確，請確認是否完整複製（應以 CHKSYNC1: 開頭）")) from exc
    if not isinstance(data, dict):
        raise ValueError(t("連線設定碼格式不正確"))
    missing = [key for key in REQUIRED_KEYS if not str(data.get(key, "") or "").strip()]
    if missing:
        raise ValueError(t("連線設定碼缺少：{names}", names=_label_list(missing)))
    return {key: str(data.get(key, "") or "").strip() for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS)}


@dataclass(frozen=True)
class SyncConfig:
    endpoint: str
    device_id: str
    site_id: str
    api_key: str
    hmac_secret: str
    ca_bundle: str | bool
    timeout_seconds: float
    interval_seconds: float

    @classmethod
    def from_values(cls, values: dict[str, str]) -> "SyncConfig":
        return cls(
            endpoint=str(values.get("endpoint", "") or "").strip(),
            device_id=str(values.get("device_id", "") or "").strip(),
            site_id=str(values.get("site_id", "") or "").strip(),
            api_key=str(values.get("api_key", "") or "").strip(),
            hmac_secret=str(values.get("hmac_secret", "") or "").strip(),
            ca_bundle=str(values.get("ca_bundle", "") or "").strip() or True,
            timeout_seconds=float(os.environ.get("CHICKEN_SYNC_TIMEOUT_SECONDS", "60")),
            interval_seconds=float(os.environ.get("CHICKEN_SYNC_INTERVAL_SECONDS", "15")),
        )

    @classmethod
    def from_environment(cls) -> "SyncConfig":
        """Effective settings: CHICKEN_SYNC_* environment variables override the saved settings file."""
        saved = load_saved_config()
        values = {
            key: os.environ.get(ENV_NAMES[key], "").strip() or saved.get(key, "")
            for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS)
        }
        return cls.from_values(values)

    def missing_keys(self) -> list[str]:
        return [key for key in REQUIRED_KEYS if not getattr(self, key)]

    def validation_error(self) -> str:
        missing = self.missing_keys()
        if missing:
            return f"尚未設定：{', '.join(ENV_NAMES[key] for key in missing)}"
        parsed = urlsplit(self.endpoint)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            return "CHICKEN_SYNC_ENDPOINT 必須是有效的 HTTPS 網址"
        if len(self.hmac_secret.encode("utf-8")) < 32:
            return "CHICKEN_SYNC_HMAC_SECRET 至少需要 32 bytes"
        return ""

    def friendly_status(self) -> str:
        """Plain-language status for operators (the technical names stay in validation_error).

        Display text in the current UI language: call from the UI thread only.
        """
        missing = self.missing_keys()
        if len(missing) == len(REQUIRED_KEYS):
            return t("尚未連線到廠商：請貼上廠商提供的連線設定碼。")
        if missing:
            return t("連線設定不完整，缺少：{names}。", names=_label_list(missing))
        error = self.validation_error()
        if "HTTPS" in error:
            return t("上傳網址必須以 https:// 開頭，請向廠商確認。")
        if "32 bytes" in error:
            return t("簽章密鑰長度不足，請向廠商重新索取連線設定碼。")
        return ""


# ---------------------------------------------------------------------------
# Request signing (shared by the connection test and the snapshot upload)
# ---------------------------------------------------------------------------

def _subject_token(chicken_id: str, config: SyncConfig) -> str:
    """Irreversible per-site pseudonym so real chicken IDs never leave the farm."""
    key = config.hmac_secret.encode("utf-8")
    message = f"{config.site_id}:{chicken_id}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()[:20]


SIGNATURE_VERSION = "2"  # v2 also signs X-Site-ID, so the farm folder name can't be swapped in transit


def _canonical_signature(secret: str, path: str, timestamp: str, nonce: str, payload_sha256: str,
                         site_id: str | None = None) -> str:
    canonical = f"POST\n{path}\n{timestamp}\n{nonce}\n{payload_sha256}"
    if site_id is not None:
        canonical += f"\n{site_id}"
    return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def _signed_headers(config: SyncConfig, url: str, digest: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    parsed = urlsplit(url)
    request_path = parsed.path or "/"
    if parsed.query:
        request_path += f"?{parsed.query}"
    return {
        "Authorization": f"Bearer {config.api_key}",
        "X-Device-ID": config.device_id,
        "X-Site-ID": config.site_id,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Content-SHA256": digest,
        "X-Signature-Version": SIGNATURE_VERSION,
        "X-Signature": _canonical_signature(config.hmac_secret, request_path, timestamp, nonce, digest,
                                            config.site_id),
    }


def _signed_post(config: SyncConfig, url: str, body: bytes, session, content_type: str, extra=None):
    headers = _signed_headers(config, url, hashlib.sha256(body).hexdigest())
    headers["Content-Type"] = content_type
    headers.update(extra or {})
    response = session.post(url, data=body, headers=headers, timeout=config.timeout_seconds, verify=config.ca_bundle)
    response.raise_for_status()
    return response


def ping_url(endpoint: str) -> str:
    """The provider's connection-test URL next to the configured endpoint (…/v1/… → …/v1/ping)."""
    return urljoin(endpoint, "ping")


def snapshot_url(endpoint: str, upload_id: str, suffix: str) -> str:
    return urljoin(endpoint, f"snapshots/{upload_id}/{suffix}")


def check_connection(config: SyncConfig | None = None, session=None) -> tuple[bool, str]:
    """Send a signed, empty request to the provider to check address, certificate and credentials."""
    config = config or SyncConfig.from_environment()
    error = config.validation_error()
    if error:
        return False, config.friendly_status() or display_validation_error(error)
    url = ping_url(config.endpoint)
    digest = hashlib.sha256(b"").hexdigest()
    headers = _signed_headers(config, url, digest)
    headers["Content-Type"] = "application/octet-stream"
    client = session or requests.Session()
    try:
        response = client.post(url, data=b"", headers=headers, timeout=min(config.timeout_seconds, 15), verify=config.ca_bundle)
    except requests.exceptions.SSLError:
        return False, t("HTTPS 憑證驗證失敗：請確認上傳網址正確，或向廠商索取憑證設定")
    except requests.exceptions.Timeout:
        return False, t("連線逾時：請檢查網路或稍後再試")
    except requests.exceptions.ConnectionError:
        return False, t("無法連線到廠商伺服器：請檢查網路連線與上傳網址")
    except Exception as exc:  # pragma: no cover - defensive
        return False, t("連線測試失敗：{error}", error=exc)
    if response.status_code == 200:
        return True, t("連線成功：網址、憑證與金鑰皆正確，可以按「回傳測試影像」。")
    if response.status_code in (401, 403):
        try:
            detail = str(response.json().get("detail", ""))
        except Exception:
            detail = ""
        if detail == "Expired request":
            return False, t("裝置時間與伺服器相差太多：請校正這台電腦的日期與時間")
        return False, t("認證失敗：API 金鑰或簽章密鑰不正確，請向廠商重新索取連線設定碼")
    if response.status_code == 404:
        return False, t("找不到廠商的連線測試端點（404）：請向廠商確認上傳網址")
    return False, t("廠商伺服器回應異常（HTTP {status}）", status=response.status_code)


def describe_error(text: str | None) -> str:
    """Translate a stored transport error into an operator-facing explanation (UI thread, display time)."""
    text = str(text or "")
    lowered = text.lower()
    if "401" in text or "unauthorized" in lowered or "invalid credentials" in lowered or "signature" in lowered:
        return t("認證失敗：API 金鑰或簽章密鑰不正確，請向廠商重新索取連線設定碼")
    if "certificate" in lowered or "ssl" in lowered:
        return t("HTTPS 憑證驗證失敗：請確認上傳網址正確，或向廠商索取憑證設定")
    if "expired request" in lowered:
        return t("裝置時間與伺服器相差太多：請校正這台電腦的日期與時間")
    if "413" in text or "too large" in lowered:
        return t("資料包超過廠商伺服器的大小上限：請聯絡廠商調整")
    if "no space" in lowered or "errno 28" in lowered:
        return t("這台裝置的硬碟空間不足，無法打包：請清理空間後重試")
    if "timed out" in lowered or "timeout" in lowered:
        return t("連線逾時：網路不穩或伺服器忙碌，系統會自動重試")
    if "connection" in lowered or "name resolution" in lowered or "getaddrinfo" in lowered or "max retries" in lowered:
        return t("無法連線到廠商伺服器：請檢查網路，系統會自動重試")
    if "404" in text:
        return t("上傳網址錯誤（404），或廠商伺服器尚未支援整包回傳：請向廠商確認")
    if any(code in text for code in ("500", "502", "503", "504")):
        return t("廠商伺服器暫時異常，系統會自動重試")
    return text[:200] or t("未知錯誤")


# ---------------------------------------------------------------------------
# Snapshot jobs
# ---------------------------------------------------------------------------

def outgoing_dir() -> Path:
    return product_review.REVIEW_DIR / "outgoing"


def _ensure_table(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshot_uploads (
            upload_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            file_path TEXT,
            size_bytes INTEGER,
            sha256 TEXT,
            part_size INTEGER,
            parts_total INTEGER,
            parts_done INTEGER NOT NULL DEFAULT 0,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT NOT NULL,
            last_error TEXT,
            completed_at TEXT,
            stored_as TEXT,
            summary_json TEXT
        )
        """
    )


def _db():
    connection = product_review.get_connection()
    _ensure_table(connection)
    return connection


def _job(upload_id: str) -> dict | None:
    with closing(_db()) as connection, connection:
        row = connection.execute("SELECT * FROM snapshot_uploads WHERE upload_id = ?", (upload_id,)).fetchone()
    return dict(row) if row else None


def _update(upload_id: str, **fields) -> None:
    fields["updated_at"] = _now_text()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with closing(_db()) as connection, connection:
        connection.execute(
            f"UPDATE snapshot_uploads SET {assignments} WHERE upload_id = ?", (*fields.values(), upload_id)
        )


def _discard_file(path_text: str | None) -> None:
    """Best-effort delete. On Windows a file the worker still has open can't be removed (PermissionError):
    leave it, the worker removes superseded files once it has closed them (_cleanup_superseded)."""
    if path_text:
        for path in (Path(path_text), Path(path_text + ".part")):
            try:
                path.unlink()
            except OSError:
                pass


def _claim(upload_id: str, **fields) -> bool:
    """Update a job only while it is still active. A job replaced by a newer「回傳」(superseded) stays
    superseded, so it can never come back to life and be uploaded as a duplicate."""
    fields["updated_at"] = _now_text()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with closing(_db()) as connection, connection:
        cursor = connection.execute(
            f"UPDATE snapshot_uploads SET {assignments} WHERE upload_id = ? "
            f"AND status IN ({','.join('?' * len(ACTIVE_STATES))})",
            (*fields.values(), upload_id, *ACTIVE_STATES),
        )
    return cursor.rowcount > 0


def _cleanup_superseded() -> None:
    """Delete zips of replaced snapshots that could not be removed while they were open."""
    with closing(_db()) as connection, connection:
        rows = connection.execute(
            "SELECT upload_id, file_path FROM snapshot_uploads WHERE status = 'superseded' AND file_path IS NOT NULL"
        ).fetchall()
    for row in rows:
        _discard_file(row["file_path"])
        if not Path(row["file_path"]).exists() and not Path(row["file_path"] + ".part").exists():
            with closing(_db()) as connection, connection:
                connection.execute("UPDATE snapshot_uploads SET file_path = NULL WHERE upload_id = ?",
                                   (row["upload_id"],))


def request_snapshot() -> str:
    """Queue a complete snapshot of all reviewed images; an unfinished older one is replaced."""
    upload_id = f"{_now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
    now = _now_text()
    with _JOB_LOCK, closing(_db()) as connection, connection:
        for row in connection.execute(
            f"SELECT upload_id, file_path FROM snapshot_uploads WHERE status IN ({','.join('?' * len(ACTIVE_STATES))})",
            ACTIVE_STATES,
        ).fetchall():
            connection.execute(
                "UPDATE snapshot_uploads SET status = 'superseded', updated_at = ? WHERE upload_id = ?",
                (now, row["upload_id"]),
            )
            _discard_file(row["file_path"])
        connection.execute(
            "INSERT INTO snapshot_uploads (upload_id, status, created_at, updated_at, next_attempt_at) "
            "VALUES (?, 'pending', ?, ?, ?)",
            (upload_id, now, now, now),
        )
    _WAKE.set()
    return upload_id


def retry_now() -> bool:
    """Retry a failed snapshot immediately instead of waiting for its back-off."""
    with closing(_db()) as connection, connection:
        cursor = connection.execute(
            "UPDATE snapshot_uploads SET next_attempt_at = ?, updated_at = ? WHERE status = 'failed'",
            (_now_text(), _now_text()),
        )
    _WAKE.set()
    return cursor.rowcount > 0


def snapshot_status() -> dict[str, dict | None]:
    """The job in progress (if any) and the last completed one, for the status panel."""
    with closing(_db()) as connection, connection:
        active = connection.execute(
            f"SELECT * FROM snapshot_uploads WHERE status IN ({','.join('?' * len(ACTIVE_STATES))}) "
            "ORDER BY created_at DESC LIMIT 1",
            ACTIVE_STATES,
        ).fetchone()
        done = connection.execute(
            "SELECT * FROM snapshot_uploads WHERE status = 'done' ORDER BY completed_at DESC LIMIT 1"
        ).fetchone()
    result = {"active": dict(active) if active else None, "last_done": dict(done) if done else None}
    for job in result.values():
        if job:
            job["summary"] = json.loads(job.get("summary_json") or "{}")
            job["stalled"] = job["status"] == "failed" and int(job["attempt_count"]) >= STALLED_AFTER_ATTEMPTS
    return result


_MODEL_HASH_CACHE: dict[tuple, str] = {}


def file_hash12(path: Path) -> str:
    """First 12 hex chars of the file's SHA-256, cached per (path, size, mtime): the comb model is >100 MB,
    so it is hashed once per file version, not on every page run or snapshot. Raises OSError if missing."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _MODEL_HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        _MODEL_HASH_CACHE[key] = digest.hexdigest()[:12]
    return _MODEL_HASH_CACHE[key]


def _model_versions() -> dict[str, dict[str, str]]:
    versions = {}
    for role in ("comb", "shank"):
        path = model_registry.active_path(role)
        try:
            versions[role] = {"file": path.name, "sha256_12": file_hash12(path)}
        except OSError:
            versions[role] = {"file": path.name, "sha256_12": "missing"}
    return versions


def _build(job: dict, config: SyncConfig) -> dict:
    """Pack every reviewed image into outgoing/<upload_id>.zip and record its size and checksum."""
    folder = outgoing_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{job['upload_id']}.zip"
    temporary = Path(str(target) + ".part")
    info = {
        "schema": "chicken-snapshot/1",
        "upload_id": job["upload_id"],
        "site_id": config.site_id,
        "device_id": config.device_id,
        "requested_at": job["created_at"],
        "packed_at": _now_text(),
        "chicken_ids": "pseudonymised per site (HMAC); the farm keeps the mapping",
        "review_summary": product_review.review_summary(),
        "models": _model_versions(),
    }
    summary = product_review.write_review_bundle(
        temporary,
        subject_id=lambda chicken_id: _subject_token(chicken_id, config),
        extra_json={"snapshot_info.json": info},
    )
    temporary.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    size = target.stat().st_size
    claimed = _claim(
        job["upload_id"],
        status="uploading",
        file_path=str(target),
        size_bytes=size,
        sha256=digest.hexdigest(),
        part_size=PART_SIZE_BYTES,
        parts_total=max(1, math.ceil(size / PART_SIZE_BYTES)),
        parts_done=0,
        summary_json=json.dumps(summary, ensure_ascii=False),
    )
    if not claimed:  # 「回傳」was pressed again while packing: this one was replaced
        _discard_file(str(target))
        return None
    return _job(job["upload_id"])


def _still_active(upload_id: str) -> bool:
    job = _job(upload_id)
    return bool(job and job["status"] in ACTIVE_STATES)


def _next_job() -> dict | None:
    with closing(_db()) as connection, connection:
        row = connection.execute(
            f"SELECT * FROM snapshot_uploads WHERE status IN ({','.join('?' * len(ACTIVE_STATES))}) "
            "AND next_attempt_at <= ? ORDER BY created_at DESC LIMIT 1",
            (*ACTIVE_STATES, _now_text()),
        ).fetchone()
    return dict(row) if row else None


def process_next_snapshot(config: SyncConfig | None = None, session=None) -> bool:
    """Build (if needed) and upload the pending snapshot, resuming from the last confirmed part.

    Returns True when a snapshot was completed in this call.
    """
    config = config or SyncConfig.from_environment()
    if config.validation_error():
        return False
    _cleanup_superseded()
    job = _next_job()
    if job is None:
        return False
    upload_id = job["upload_id"]
    client = session or requests.Session()
    try:
        if not job["file_path"] or not Path(job["file_path"]).is_file():
            job = _build(job, config)
            if job is None:
                return False  # replaced by a newer snapshot while packing
        elif not _claim(upload_id, status="uploading"):
            return False  # replaced by a newer snapshot
        part_size = int(job["part_size"])
        replaced = False
        with open(job["file_path"], "rb") as handle:
            for index in range(int(job["parts_done"]), int(job["parts_total"])):
                if not _still_active(upload_id):
                    replaced = True  # replaced by a newer snapshot
                    break
                handle.seek(index * part_size)
                chunk = handle.read(part_size)
                _signed_post(
                    config,
                    snapshot_url(config.endpoint, upload_id, f"parts/{index}"),
                    chunk,
                    client,
                    "application/octet-stream",
                    {"X-Part-Count": str(job["parts_total"])},
                )
                _update(upload_id, parts_done=index + 1)
        if replaced:
            _discard_file(job["file_path"])  # now closed, so this also works on Windows
            return False
        completion = json.dumps(
            {
                "parts": int(job["parts_total"]),
                "size": int(job["size_bytes"]),
                "sha256": job["sha256"],
                "requested_at": job["created_at"],
                "summary": json.loads(job["summary_json"] or "{}"),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            response = _signed_post(
                config, snapshot_url(config.endpoint, upload_id, "complete"), completion, client, "application/json"
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (400, 409):
                # The server is missing parts or the assembled file did not match: send everything again.
                _update(upload_id, parts_done=0)
            raise
        stored_as = ""
        try:
            stored_as = str(response.json().get("stored_as", ""))
        except Exception:
            pass
        _update(upload_id, status="done", completed_at=_now_text(), stored_as=stored_as, last_error=None,
                attempt_count=0)
        _discard_file(job["file_path"])
        return True
    except Exception as exc:
        current = _job(upload_id)
        if current and current["status"] in ACTIVE_STATES:
            attempts = int(current["attempt_count"]) + 1
            delay = min(MAX_RETRY_DELAY_SECONDS, 5 * (2 ** min(attempts - 1, 10)))
            _update(
                upload_id,
                status="failed",
                attempt_count=attempts,
                next_attempt_at=(_now() + timedelta(seconds=delay)).strftime("%Y-%m-%d %H:%M:%S"),
                last_error=str(exc)[:500],
            )
        return False
    finally:
        if session is None:
            client.close()


def worker_running() -> bool:
    return _WORKER_STARTED


def start_sync_worker() -> bool:
    """Start the background uploader once settings are valid (safe to call repeatedly)."""
    global _WORKER_STARTED
    if os.environ.get("CHICKEN_SYNC_DISABLE_WORKER") == "1":  # tests / maintenance
        return False
    config = SyncConfig.from_environment()
    if config.validation_error():
        return False
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return True
        _WORKER_STARTED = True

    def worker() -> None:
        while True:
            current = SyncConfig.from_environment()
            try:
                if not current.validation_error():
                    process_next_snapshot(current)
            except Exception:
                pass
            _WAKE.wait(timeout=max(5.0, current.interval_seconds))
            _WAKE.clear()

    threading.Thread(target=worker, name="review-snapshot-sync", daemon=True).start()
    return True
