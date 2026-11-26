"""Vendor tool: issue upload credentials for one farm device and print its one-line setup code.

Usage (on the vendor server):
    python scripts/provider_issue_device.py \
        --endpoint https://provider.example.com/v1/review-packages \
        --site-id farm-a --device-id orin-001 \
        --credentials /etc/chicken-provider-credentials.json

The device's random API key and HMAC secret are added to the credentials file read by
provider_sync_server.py (CHICKEN_PROVIDER_CREDENTIALS_FILE). Give the printed CHKSYNC1:... code to the
farm through a trusted channel (in person, encrypted message); they paste it into
「資料與模型維護 → 回傳影像給廠商 → 連線設定」. Re-running for the same device rotates its keys.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import sys
from pathlib import Path
from urllib.parse import urlsplit

SETUP_CODE_PREFIX = "CHKSYNC1:"  # must match src/secure_sync.py
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}$")  # must match provider_sync_server.py


def encode_setup_code(values: dict[str, str]) -> str:
    raw = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return SETUP_CODE_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def issue(endpoint: str, site_id: str, device_id: str, credentials_path: Path, ca_bundle: str = "") -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError("endpoint must be an https:// URL")
    for label, value in (("site-id", site_id), ("device-id", device_id)):
        if not SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError(f"{label} may only contain letters, digits, '.', '_' and '-' (max 64)")

    api_key = secrets.token_urlsafe(32)
    hmac_secret = secrets.token_urlsafe(48)  # 64 characters, well above the 32-byte minimum

    credentials: dict[str, object] = {}
    if credentials_path.is_file():
        credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    credentials[device_id] = {"api_key": api_key, "hmac_secret": hmac_secret, "site_id": site_id}
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = credentials_path.with_suffix(credentials_path.suffix + ".part")
    temporary.write_text(json.dumps(credentials, indent=2), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(credentials_path)

    values = {
        "endpoint": endpoint,
        "device_id": device_id,
        "site_id": site_id,
        "api_key": api_key,
        "hmac_secret": hmac_secret,
    }
    if ca_bundle:
        values["ca_bundle"] = ca_bundle
    return encode_setup_code(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True, help="public upload URL, e.g. https://host/v1/review-packages")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--credentials", required=True, type=Path, help="credentials JSON used by the receiver")
    parser.add_argument("--ca-bundle", default="", help="optional CA file path on the farm device (private CA only)")
    args = parser.parse_args(argv)
    try:
        code = issue(args.endpoint, args.site_id, args.device_id, args.credentials, args.ca_bundle)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Credentials for {args.device_id} saved to {args.credentials}")
    print("Give this setup code to the farm (treat it like a password):\n")
    print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
