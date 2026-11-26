"""YOLO-seg model versions: best_<role>_<雞舍名稱>_<YYYYMMDD>.pt files in models/.

- Every imported model is kept (models/ is the version history; older versions allow rollback).
- The model in use per role is either pinned (data/model_versions.json) or, by default, the newest
  version by update date (then by file time).
- With no versioned file for a role, the original best_<role>.pt in the project folder is used, so a
  station that has not imported anything behaves exactly as before.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

from inference_utils import BASE_DIR, DATA_DIR

ROLES = ("comb", "shank")
MODELS_DIR = BASE_DIR / "models"
SELECTION_PATH = DATA_DIR / "model_versions.json"
LEGACY_PATHS = {role: BASE_DIR / f"best_{role}.pt" for role in ROLES}
AUTO = ""  # selection value meaning "newest version"
_NAME = re.compile(r"^best_(comb|shank)_(.+)_(\d{8})(?:_(\d+))?\.pt$")
_CACHE_SECONDS = 1.0
_cache: dict[str, tuple[float, Path]] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class ModelVersion:
    role: str
    path: Path
    farm: str
    updated: str  # YYYYMMDD ("" for the original unversioned file)
    legacy: bool = False

    @property
    def updated_text(self) -> str:
        return f"{self.updated[:4]}-{self.updated[4:6]}-{self.updated[6:]}" if self.updated else "-"


def safe_farm_name(name: str) -> str:
    """Farm name usable in a file name: letters/digits (any script) and '-', at most 40 characters."""
    cleaned = re.sub(r"[^\w-]+|_", "-", str(name or "").strip(), flags=re.UNICODE).strip("-")
    return re.sub(r"-{2,}", "-", cleaned)[:40] or "farm"


def version_file_name(role: str, farm: str, updated: date | str) -> str:
    stamp = updated.strftime("%Y%m%d") if isinstance(updated, (date, datetime)) else str(updated)
    if role not in ROLES or not re.fullmatch(r"\d{8}", stamp):
        raise ValueError("invalid model role or date")
    return f"best_{role}_{safe_farm_name(farm)}_{stamp}.pt"


def parse_version(path: Path) -> Optional[ModelVersion]:
    match = _NAME.match(path.name)
    if not match:
        return None
    return ModelVersion(role=match.group(1), path=path, farm=match.group(2), updated=match.group(3))


def list_versions(role: str) -> list[ModelVersion]:
    """Versioned files for `role`, newest first; the original best_<role>.pt is listed last if present."""
    versions = []
    if MODELS_DIR.is_dir():
        for path in MODELS_DIR.glob(f"best_{role}_*.pt"):
            version = parse_version(path)
            if version and version.role == role and path.is_file():
                versions.append(version)
    versions.sort(key=lambda v: (v.updated, v.path.stat().st_mtime_ns), reverse=True)
    if LEGACY_PATHS[role].is_file():
        versions.append(ModelVersion(role=role, path=LEGACY_PATHS[role], farm="-", updated="", legacy=True))
    return versions


def _load_selection() -> dict[str, str]:
    try:
        data = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        return {role: str(data.get(role, "") or "") for role in ROLES}
    except (OSError, ValueError, AttributeError):
        return {role: AUTO for role in ROLES}


def selection(role: str) -> str:
    """The pinned file name for `role`, or AUTO (newest version)."""
    return _load_selection()[role]


def _resolve(role: str) -> Path:
    pinned = _load_selection()[role]
    versions = list_versions(role)
    for version in versions:
        if pinned and version.path.name == pinned:
            return version.path
    return versions[0].path if versions else LEGACY_PATHS[role]


def active_path(role: str) -> Path:
    """The model file in use for `role` (cached for a second: preview inference asks every frame)."""
    now = time.monotonic()
    with _lock:
        cached = _cache.get(role)
        if cached and now - cached[0] < _CACHE_SECONDS:
            return cached[1]
    path = _resolve(role)
    with _lock:
        _cache[role] = (now, path)
    return path


def _forget_cache() -> None:
    with _lock:
        _cache.clear()


def set_active(role: str, file_name: str) -> None:
    """Pin a version (file name from list_versions), or AUTO to always use the newest one."""
    if role not in ROLES:
        raise ValueError(f"unknown model role: {role}")
    if file_name and file_name not in {v.path.name for v in list_versions(role)}:
        raise ValueError(f"model version not found: {file_name}")
    data = _load_selection()
    data[role] = file_name
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SELECTION_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SELECTION_PATH)
    _forget_cache()


def import_model(role: str, source: bytes | Path, farm: str, updated: date | str,
                 validate: Optional[Callable[[Path, str], None]] = None) -> Path:
    """Store a model as models/best_<role>_<farm>_<date>.pt (never overwrites: adds _2, _3 …).
    `validate(path, role)` may raise to reject the file (e.g. not a segmentation model)."""
    name = version_file_name(role, farm, updated)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    incoming = MODELS_DIR / f"_incoming_{uuid.uuid4().hex}.pt"  # not a version name: never listed
    try:
        if isinstance(source, (bytes, bytearray)):
            incoming.write_bytes(source)
        else:
            shutil.copyfile(source, incoming)
        if validate is not None:
            validate(incoming, role)
        target = MODELS_DIR / name
        counter = 2
        while target.exists():
            target = MODELS_DIR / name.replace(".pt", f"_{counter}.pt")
            counter += 1
        incoming.replace(target)
    finally:
        incoming.unlink(missing_ok=True)
    _forget_cache()
    return target


def register_original(role: str, farm: str, validate=None) -> Path:
    """Add the original best_<role>.pt to the version list (copied, dated by its file time)."""
    source = LEGACY_PATHS[role]
    if not source.is_file():
        raise FileNotFoundError(str(source))
    updated = datetime.fromtimestamp(source.stat().st_mtime).date()
    return import_model(role, source, farm, updated, validate)


class ActiveModels(Mapping):
    """Drop-in for the old fixed {"comb": path, "shank": path}: always the model currently in use."""

    def __getitem__(self, role: str) -> Path:
        if role not in ROLES:
            raise KeyError(role)
        return active_path(role)

    def __iter__(self):
        return iter(ROLES)

    def __len__(self) -> int:
        return len(ROLES)
