"""Check the runtime assets and hardware visibility needed by the app."""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = APP_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    failures = []
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")

    modules = (
        "streamlit",
        "streamlit_image_coordinates",
        "numpy",
        "PIL",
        "cv2",
        "serial",
        "torch",
        "ultralytics",
    )
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] import {name}: {version}")
        except Exception as exc:
            failures.append(f"import {name}: {exc}")
            print(f"[FAIL] import {name}: {exc}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception:
        pass

    if not failures:
        from ultralytics import YOLO

        for filename, expected_class in (("best.pt", "comb"), ("best_shank.pt", "shank")):
            path = APP_DIR / filename
            if not path.is_file():
                failures.append(f"missing model: {path}")
                print(f"[FAIL] model missing: {path}")
                continue
            try:
                model = YOLO(str(path))
                names = [str(value).lower() for value in model.names.values()]
                valid = model.task == "segment" and expected_class in names
                status = "OK" if valid else "FAIL"
                print(f"[{status}] {filename}: task={model.task}, classes={model.names}")
                if not valid:
                    failures.append(f"invalid {filename}: task={model.task}, classes={model.names}")
            except Exception as exc:
                failures.append(f"load {filename}: {exc}")
                print(f"[FAIL] load {filename}: {exc}")

    if sys.platform.startswith("linux"):
        cameras = sorted(Path("/dev").glob("video*"))
        serial_devices = sorted(Path("/dev").glob("ttyUSB*")) + sorted(Path("/dev").glob("ttyACM*"))
        print("Camera devices: " + (", ".join(map(str, cameras)) or "none detected"))
        print("Serial devices: " + (", ".join(map(str, serial_devices)) or "none detected"))

    if failures:
        print("\nRuntime check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nRequired Python modules and dual-camera models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
