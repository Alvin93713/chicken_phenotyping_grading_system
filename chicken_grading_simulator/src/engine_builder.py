"""Automatic TensorRT conversion: whenever a model in use has no up-to-date engine (a vendor model was just
imported, another version was chosen, a new .pt arrived by git pull or copy), build <model>.engine next to it
in the background, check that it loads and predicts, and only then put it in place. dual_camera.runtime_model
picks it up by itself (same name, not older than the .pt); until then the .pt is used.

Runs only where TensorRT is installed with a CUDA GPU (the Orin); elsewhere it does nothing.
CHICKEN_AUTO_BUILD_ENGINE=0 or CHICKEN_DISABLE_TENSORRT=1 turns it off.
Export settings follow TENSORRT_FP16_GUIDE.md (FP16; comb 960 px, shank 640 px)."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROLES = ("comb", "shank")
IMGSZ = {"comb": int(os.environ.get("CHICKEN_ENGINE_IMGSZ_COMB", "960")),
         "shank": int(os.environ.get("CHICKEN_ENGINE_IMGSZ_SHANK", "640"))}
CHECK_SECONDS = 60  # a model that appears without a UI action (git pull, copy) is noticed within a minute

# role -> {"state": idle|building|ready|failed|unavailable, "model": name, "message": str, "since": time}
status: dict[str, dict] = {}
_lock = threading.Lock()
_wake = threading.Event()
_started = False
_failed: dict[str, float] = {}  # "<pt path>|<pt mtime>" -> time: a model that failed is not retried until it changes


def enabled() -> bool:
    if os.environ.get("CHICKEN_AUTO_BUILD_ENGINE", "1") == "0" or os.environ.get("CHICKEN_DISABLE_TENSORRT") == "1":
        return False
    if importlib.util.find_spec("tensorrt") is None:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _set(role, state, model="", message=""):
    with _lock:
        status[role] = {"state": state, "model": model, "message": message, "since": time.time()}


def _model_key(pt: Path) -> str:
    return f"{pt}|{pt.stat().st_mtime_ns}"


def needs_engine(pt: Path) -> bool:
    """True when `pt` exists and has no engine, or only one built from an older .pt."""
    if not pt.is_file() or pt.suffix != ".pt":
        return False
    engine = pt.with_suffix(".engine")
    return not engine.is_file() or engine.stat().st_mtime_ns < pt.stat().st_mtime_ns


def export_engine(pt: Path, role: str, workdir: Path) -> Path:
    """Export a copy of `pt` inside `workdir` (so a half-written engine never sits next to the model).
    Returns the engine file."""
    from ultralytics import YOLO

    copy = workdir / pt.name
    shutil.copy2(pt, copy)
    exported = YOLO(str(copy)).export(format="engine", quantize=16, imgsz=IMGSZ[role], device=0, verbose=False)
    engine = Path(exported) if exported else copy.with_suffix(".engine")
    if not engine.is_file():
        raise RuntimeError(f"export produced no engine for {pt.name}")
    return engine


def check_engine(engine: Path, role: str) -> None:
    """Load the new engine and run it once; raises when it is unusable."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(engine), task="segment")
    if model.task != "segment":
        raise RuntimeError(f"engine is not a segmentation model ({model.task})")
    model.predict(np.zeros((720, 1280, 3), dtype=np.uint8), conf=0.5, device=0, verbose=False)


def build(role: str, pt: Path, export=None, check=None) -> Path:
    """Build, check and install <pt>.engine. The engine gets a modification time not older than the .pt."""
    export, check = export or export_engine, check or check_engine  # looked up now (tests replace them)
    _set(role, "building", pt.name, "")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="engine_build_") as folder:
        engine = export(pt, role, Path(folder))
        check(engine, role)
        target = pt.with_suffix(".engine")
        partial = target.with_name(target.name + ".part")
        shutil.copyfile(engine, partial)
        partial.replace(target)  # atomic: the app sees either the old file or the finished engine
    now = time.time()
    os.utime(target, (now, max(now, pt.stat().st_mtime)))
    _set(role, "ready", pt.name, f"{time.monotonic() - started:.0f} s")
    return target


def check_and_build(paths=None) -> list[str]:
    """Build missing / stale engines for the models in use. Returns the roles that were built."""
    if paths is None:
        import model_registry
        paths = {role: model_registry.active_path(role) for role in ROLES}
    built = []
    for role in ROLES:
        pt = Path(paths[role])
        if not needs_engine(pt):
            if pt.with_suffix(".engine").is_file() and status.get(role, {}).get("state") != "ready":
                _set(role, "ready", pt.name)
            continue
        key = _model_key(pt)
        if key in _failed:
            continue
        try:
            build(role, pt)
            built.append(role)
            _warm(role)
        except Exception as exc:
            _failed[key] = time.time()
            _set(role, "failed", pt.name, str(exc))
            print(f"TensorRT build failed for {pt}: {exc}", file=sys.stderr)
    return built


def _warm(role):
    """Load + run the new engine in the app so the next chicken does not wait for it."""
    try:
        import dual_camera
        dual_camera._warm_up(role)
    except Exception as exc:
        print(f"Warm-up after build failed for {role}: {exc}", file=sys.stderr)


def _loop():
    while True:
        try:
            check_and_build()
        except Exception as exc:  # never let the watcher die
            print(f"Engine watcher error: {exc}", file=sys.stderr)
        _wake.wait(CHECK_SECONDS)
        _wake.clear()


def start() -> bool:
    """Start the background watcher once per process (only where TensorRT can be built)."""
    global _started
    if not enabled():
        for role in ROLES:
            status.setdefault(role, {"state": "unavailable", "model": "", "message": "", "since": time.time()})
        return False
    with _lock:
        if _started:
            return False
        _started = True
    threading.Thread(target=_loop, name="engine-builder", daemon=True).start()
    return True


def request_check() -> None:
    """Check now (after a model was imported or another version chosen)."""
    _wake.set()


def retry(role: str) -> None:
    """Forget a failed build of the role's current model and check again."""
    for key in [k for k in _failed if f"best_{role}" in Path(k.split("|")[0]).name]:
        _failed.pop(key, None)
    request_check()
