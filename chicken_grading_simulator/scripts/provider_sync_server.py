"""Reference provider receiver for securely uploaded review snapshots.

Farm devices send one complete snapshot (all reviewed images, fine-tuning layout) per「回傳」press,
in signed parts:  POST /v1/snapshots/<upload_id>/parts/<n>  then  POST /v1/snapshots/<upload_id>/complete
Stored as  <DATA_DIR>/<site_id>_<YYYYMMDD>/<device_id>_<HHMMSS>.zip  (server local date/time; set TZ).
POST /v1/ping is the connection test. POST /v1/review-packages (per-image packages) is kept for
older devices.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request


# Received files go to <DATA_DIR>/<site_id>_<YYYYMMDD>/ (one folder per farm per day, server local
# date; set TZ, e.g. TZ=Asia/Taipei, to choose the time zone).
DATA_DIR = Path(os.environ.get("CHICKEN_PROVIDER_DATA_DIR", "provider_inbox")).resolve()
# Duplicate/replay bookkeeping. Keep it on a local Linux filesystem: SQLite is unreliable on network
# or Windows-mounted drives (e.g. /mnt/c under WSL), which is where DATA_DIR may point.
DB_PATH = Path(os.environ.get("CHICKEN_PROVIDER_DB_PATH", str(DATA_DIR / "received_uploads.db"))).resolve()
MAX_PAYLOAD_BYTES = int(os.environ.get("CHICKEN_PROVIDER_MAX_PAYLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_CLOCK_SKEW_SECONDS = int(os.environ.get("CHICKEN_PROVIDER_MAX_CLOCK_SKEW_SECONDS", "300"))
# Snapshot parts stay below common proxy limits (Cloudflare: 100 MB per request).
MAX_PART_BYTES = int(os.environ.get("CHICKEN_PROVIDER_MAX_PART_BYTES", str(48 * 1024 * 1024)))
MAX_SNAPSHOT_BYTES = int(os.environ.get("CHICKEN_PROVIDER_MAX_SNAPSHOT_BYTES", str(20 * 1024 ** 3)))
MAX_SNAPSHOT_ENTRIES = int(os.environ.get("CHICKEN_PROVIDER_MAX_SNAPSHOT_ENTRIES", "200000"))
# Parts are staged next to the bookkeeping DB (local filesystem) until the snapshot is complete.
STAGING_DIR = DB_PATH.parent / "staging"
app = FastAPI(title="Chicken Review Secure Receiver", docs_url=None, redoc_url=None)
# Must not start with "." so "." / ".." (or hidden names) can never be used as a path segment.
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}$")
# Farm clients sign X-Site-ID too (signature version 2). Set to "1" only while older farm devices that
# sign without it are being updated.
ACCEPT_LEGACY_SIGNATURES = os.environ.get("CHICKEN_PROVIDER_ACCEPT_LEGACY_SIGNATURES", "0") == "1"


def receive_folder_name(site_id: str, received_at: float) -> str:
    """'<farm>_<YYYYMMDD>' using the server's local date (the day the package arrived)."""
    return f"{site_id}_{time.strftime('%Y%m%d', time.localtime(received_at))}"


def _required_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _safe_identifier(value: str, label: str) -> str:
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return value


def _staging_folder(device_id: str, upload_id: str) -> Path:
    """STAGING_DIR/<device>/<upload>, refusing anything that would resolve outside STAGING_DIR."""
    root = STAGING_DIR.resolve()
    folder = (root / device_id / upload_id).resolve()
    if folder == root or root not in folder.parents:
        raise HTTPException(status_code=400, detail="Invalid upload ID")
    return folder


def _remember_nonce(nonce: str, device_id: str, request_time: int) -> None:
    """Reject a signed request whose nonce was already used (replay within the clock-skew window)."""
    connection = _connection()
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS seen_nonces (nonce TEXT PRIMARY KEY, device_id TEXT NOT NULL, "
            "seen_at INTEGER NOT NULL)"
        )
        connection.execute("DELETE FROM seen_nonces WHERE seen_at < ?",
                           (int(time.time()) - 2 * MAX_CLOCK_SKEW_SECONDS,))
        connection.execute("INSERT INTO seen_nonces (nonce, device_id, seen_at) VALUES (?, ?, ?)",
                           (nonce, device_id, max(request_time, int(time.time()))))
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Replay detected") from exc
    finally:
        connection.close()


def _device_credentials(device_id: str) -> tuple[str, str]:
    api_key, hmac_secret, _ = _device_record(device_id)
    return api_key, hmac_secret


def _device_record(device_id: str) -> tuple[str, str, str]:
    """(api_key, hmac_secret, registered site_id or "") for a device."""
    credentials_path = os.environ.get("CHICKEN_PROVIDER_CREDENTIALS_FILE", "").strip()
    if credentials_path:
        try:
            credentials = json.loads(Path(credentials_path).read_text(encoding="utf-8"))
            device = credentials[device_id]
            api_key = str(device["api_key"])
            hmac_secret = str(device["hmac_secret"])
            site_id = str(device.get("site_id") or "")
        except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=401, detail="Unknown device") from exc
    else:
        api_key = _required_secret("CHICKEN_PROVIDER_API_KEY")
        hmac_secret = _required_secret("CHICKEN_PROVIDER_HMAC_SECRET")
        site_id = os.environ.get("CHICKEN_PROVIDER_SITE_ID", "").strip()  # optional in single-key mode
    if len(hmac_secret.encode("utf-8")) < 32:
        raise RuntimeError("Provider HMAC secret must be at least 32 bytes")
    return api_key, hmac_secret, site_id


def _check_site(request: Request, device_id: str) -> None:
    """A device may only upload into the farm (site) it was issued for (provider_issue_device.py records it)."""
    _, _, registered = _device_record(device_id)
    claimed = request.headers.get("X-Site-ID", "").strip()
    if registered and not hmac.compare_digest(claimed, registered):
        raise HTTPException(status_code=403, detail="Device is not registered for this site")


def _connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS received_uploads (
            item_id TEXT PRIMARY KEY,
            nonce TEXT UNIQUE NOT NULL,
            payload_sha256 TEXT NOT NULL,
            site_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            received_at INTEGER NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def _header(request: Request, name: str) -> str:
    value = request.headers.get(name, "").strip()
    if not value:
        raise HTTPException(status_code=400, detail=f"Missing header: {name}")
    return value


def _safe_archive(payload: bytes) -> dict[str, object]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if len(archive.infolist()) > 10:
                raise HTTPException(status_code=400, detail="Too many archive entries")
            total_uncompressed = 0
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise HTTPException(status_code=400, detail="Unsafe archive path")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_PAYLOAD_BYTES * 2:
                    raise HTTPException(status_code=413, detail="Expanded archive is too large")
            metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid review package") from exc
    if metadata.get("schema_version") != 1 or metadata.get("role") not in ("comb", "shank"):
        raise HTTPException(status_code=400, detail="Unsupported package schema")
    return metadata


def _verify_signed_request(request: Request, device_id: str, body: bytes) -> tuple[str, int]:
    """Check API key, clock skew, content checksum, HMAC signature (covering X-Site-ID) and that the nonce
    is new. Returns (digest, request_time)."""
    api_key, hmac_secret = _device_credentials(device_id)
    authorization = _header(request, "Authorization")
    if not hmac.compare_digest(authorization, f"Bearer {api_key}"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    timestamp = _header(request, "X-Timestamp")
    nonce = _header(request, "X-Nonce")
    expected_digest = _header(request, "X-Content-SHA256").lower()
    signature = _header(request, "X-Signature").lower()
    try:
        request_time = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid timestamp") from exc
    if abs(int(time.time()) - request_time) > MAX_CLOCK_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="Expired request")
    actual_digest = hashlib.sha256(body).hexdigest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise HTTPException(status_code=400, detail="Content checksum mismatch")
    request_path = request.url.path
    if request.url.query:
        request_path += f"?{request.url.query}"
    canonical = f"POST\n{request_path}\n{timestamp}\n{nonce}\n{actual_digest}"
    if request.headers.get("X-Signature-Version", "").strip() == "2":
        canonical += f"\n{_header(request, 'X-Site-ID')}"  # the farm folder name is signed too
    elif not ACCEPT_LEGACY_SIGNATURES:
        raise HTTPException(status_code=401, detail="Signature version 2 required; update the farm device")
    expected_signature = hmac.new(hmac_secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid package signature")
    _check_site(request, device_id)
    _remember_nonce(nonce, device_id, request_time)
    return actual_digest, request_time


@app.post("/v1/ping")
async def ping(request: Request) -> dict[str, object]:
    """Connection test used by the farm UI: verifies address, TLS and credentials, stores nothing."""
    device_id = _safe_identifier(_header(request, "X-Device-ID"), "device ID")
    _safe_identifier(_header(request, "X-Site-ID"), "site ID")
    body = await request.body()
    if len(body) > 1024:
        raise HTTPException(status_code=413, detail="Payload is too large")
    _verify_signed_request(request, device_id, body)
    return {"ok": True, "device_id": device_id, "server_time": int(time.time())}


def _key_matches_item(upload_key: str, item_id: object) -> bool:
    """Upload keys are '<item_id>.<content digest prefix>' (older clients send the bare item_id)."""
    item_id = str(item_id or "")
    return bool(item_id) and (upload_key == item_id or upload_key.startswith(item_id + "."))


@app.post("/v1/review-packages", status_code=201)
async def receive_review_package(request: Request) -> dict[str, object]:
    device_id = _safe_identifier(_header(request, "X-Device-ID"), "device ID")
    site_id = _safe_identifier(_header(request, "X-Site-ID"), "site ID")
    item_id = _safe_identifier(_header(request, "X-Idempotency-Key"), "item ID")
    api_key, hmac_secret = _device_credentials(device_id)
    authorization = _header(request, "Authorization")
    if not hmac.compare_digest(authorization, f"Bearer {api_key}"):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    timestamp = _header(request, "X-Timestamp")
    nonce = _header(request, "X-Nonce")
    expected_digest = _header(request, "X-Content-SHA256").lower()
    signature = _header(request, "X-Signature").lower()
    try:
        request_time = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid timestamp") from exc
    if abs(int(time.time()) - request_time) > MAX_CLOCK_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="Expired request")

    body = bytearray()
    digest = hashlib.sha256()
    async for chunk in request.stream():
        body.extend(chunk)
        digest.update(chunk)
        if len(body) > MAX_PAYLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Payload is too large")
    actual_digest = digest.hexdigest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise HTTPException(status_code=400, detail="Content checksum mismatch")

    request_path = request.url.path
    if request.url.query:
        request_path += f"?{request.url.query}"
    canonical = f"POST\n{request_path}\n{timestamp}\n{nonce}\n{actual_digest}".encode("utf-8")
    expected_signature = hmac.new(
        hmac_secret.encode("utf-8"), canonical, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid package signature")

    metadata = _safe_archive(bytes(body))
    if not _key_matches_item(item_id, metadata.get("item_id")):
        raise HTTPException(status_code=400, detail="Item identity mismatch")
    if metadata.get("site_id") != site_id or metadata.get("device_id") != device_id:
        raise HTTPException(status_code=400, detail="Device identity mismatch")
    _check_site(request, device_id)

    connection = _connection()
    try:
        existing = connection.execute(
            "SELECT payload_sha256 FROM received_uploads WHERE item_id = ?", (item_id,)
        ).fetchone()
        if existing:
            if hmac.compare_digest(existing["payload_sha256"], actual_digest):
                return {"accepted": True, "duplicate": True, "item_id": item_id}
            raise HTTPException(status_code=409, detail="Item ID already has different content")
        if connection.execute(
            "SELECT 1 FROM received_uploads WHERE nonce = ?", (nonce,)
        ).fetchone():
            raise HTTPException(status_code=409, detail="Replay detected")

        destination_dir = DATA_DIR / receive_folder_name(site_id, time.time())
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{device_id}_{item_id}.zip"
        temporary = destination.with_suffix(".zip.part")
        temporary.write_bytes(body)
        temporary.replace(destination)
        connection.execute(
            """
            INSERT INTO received_uploads (
                item_id, nonce, payload_sha256, site_id, device_id, stored_path, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (item_id, nonce, actual_digest, site_id, device_id, str(destination), int(time.time())),
        )
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Duplicate or replayed request") from exc
    finally:
        connection.close()
    return {"accepted": True, "duplicate": False, "item_id": item_id}

# ---------------------------------------------------------------------------
# Snapshot upload: one complete review bundle per「回傳」press, sent in signed parts
# ---------------------------------------------------------------------------

def _snapshot_db() -> sqlite3.Connection:
    connection = _connection()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS received_snapshots (
            device_id TEXT NOT NULL,
            upload_id TEXT NOT NULL,
            site_id TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            stored_path TEXT NOT NULL,
            received_at INTEGER NOT NULL,
            PRIMARY KEY (device_id, upload_id)
        )
        """
    )
    connection.commit()
    return connection


def _stored_name(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return path.name


def _validate_snapshot_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_SNAPSHOT_ENTRIES:
                raise HTTPException(status_code=400, detail="Too many archive entries")
            total = 0
            for info in entries:
                name = PurePosixPath(info.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise HTTPException(status_code=400, detail="Unsafe archive path")
                total += info.file_size
                if total > MAX_SNAPSHOT_BYTES * 2:
                    raise HTTPException(status_code=413, detail="Expanded archive is too large")
            json.loads(archive.read("manifest.json").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid snapshot archive") from exc


@app.post("/v1/snapshots/{upload_id}/parts/{index}")
async def receive_snapshot_part(upload_id: str, index: int, request: Request) -> dict[str, object]:
    device_id = _safe_identifier(_header(request, "X-Device-ID"), "device ID")
    _safe_identifier(_header(request, "X-Site-ID"), "site ID")
    _safe_identifier(upload_id, "upload ID")
    try:
        part_count = int(_header(request, "X-Part-Count"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid part count") from exc
    if not 0 <= index < part_count or part_count * MAX_PART_BYTES > MAX_SNAPSHOT_BYTES + MAX_PART_BYTES:
        raise HTTPException(status_code=400, detail="Invalid part index")
    body = await request.body()
    if len(body) > MAX_PART_BYTES:
        raise HTTPException(status_code=413, detail="Part is too large")
    digest, _ = _verify_signed_request(request, device_id, body)
    folder = _staging_folder(device_id, upload_id)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{index:06d}.part.tmp"
    temporary.write_bytes(body)  # re-sent parts (after a timeout) simply overwrite: idempotent
    temporary.replace(folder / f"{index:06d}.part")
    return {"received": index, "sha256": digest}


@app.post("/v1/snapshots/{upload_id}/complete", status_code=201)
async def complete_snapshot(upload_id: str, request: Request) -> dict[str, object]:
    device_id = _safe_identifier(_header(request, "X-Device-ID"), "device ID")
    site_id = _safe_identifier(_header(request, "X-Site-ID"), "site ID")
    _safe_identifier(upload_id, "upload ID")
    body = await request.body()
    if len(body) > 1024 * 1024:
        raise HTTPException(status_code=413, detail="Completion request is too large")
    _verify_signed_request(request, device_id, body)
    try:
        details = json.loads(body.decode("utf-8"))
        parts, size, expected = int(details["parts"]), int(details["size"]), str(details["sha256"]).lower()
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid completion request") from exc
    if size > MAX_SNAPSHOT_BYTES:
        raise HTTPException(status_code=413, detail="Snapshot is too large")

    connection = _snapshot_db()
    try:
        existing = connection.execute(
            "SELECT stored_path, sha256 FROM received_snapshots WHERE device_id = ? AND upload_id = ?",
            (device_id, upload_id),
        ).fetchone()
        if existing:  # completion re-sent after a timeout
            if not hmac.compare_digest(existing["sha256"], expected):
                raise HTTPException(status_code=409, detail="Upload ID already has different content")
            return {"accepted": True, "duplicate": True, "stored_as": _stored_name(Path(existing["stored_path"]))}

        staging = _staging_folder(device_id, upload_id)
        part_files = [staging / f"{index:06d}.part" for index in range(parts)]
        missing = [index for index, path in enumerate(part_files) if not path.is_file()]
        if missing:
            raise HTTPException(status_code=409, detail=f"Missing parts: {missing[:20]}")

        received_at = time.time()
        destination_dir = DATA_DIR / receive_folder_name(site_id, received_at)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{device_id}_{time.strftime('%H%M%S', time.localtime(received_at))}.zip"
        if destination.exists():
            destination = destination.with_name(f"{destination.stem}_{upload_id}.zip")
        temporary = destination.with_suffix(".zip.part")
        digest = hashlib.sha256()
        total = 0
        with temporary.open("wb") as output:
            for path in part_files:
                with path.open("rb") as part:
                    for chunk in iter(lambda: part.read(1 << 20), b""):
                        digest.update(chunk)
                        total += len(chunk)
                        output.write(chunk)
        if total != size or not hmac.compare_digest(digest.hexdigest(), expected):
            temporary.unlink(missing_ok=True)
            shutil.rmtree(staging, ignore_errors=True)  # corrupted: the device rebuilds and re-sends
            raise HTTPException(status_code=400, detail="Snapshot checksum mismatch")
        try:
            _validate_snapshot_zip(temporary)
        except HTTPException:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(staging, ignore_errors=True)
            raise
        temporary.replace(destination)
        connection.execute(
            "INSERT INTO received_snapshots (device_id, upload_id, site_id, sha256, size_bytes, stored_path, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (device_id, upload_id, site_id, expected, size, str(destination), int(received_at)),
        )
        connection.commit()
        shutil.rmtree(staging, ignore_errors=True)
    finally:
        connection.close()
    return {"accepted": True, "duplicate": False, "stored_as": _stored_name(destination)}
