# Jetson Orin TensorRT FP16 Migration Guide

This guide keeps the committed `.pt` models as the portable source models and
builds hardware-specific TensorRT engines directly on the target Jetson Orin.

## 1. Update the PT baseline from Git

The Git repository root is the directory above `chicken_grading_simulator`.

```bash
cd /home/ubuntu/Documents/chicken_grading_system
git status
git fetch origin
git switch codex/dual-camera-snapshot
git pull --ff-only origin codex/dual-camera-snapshot
cd chicken_grading_simulator
```

Confirm that both portable models are present:

```bash
ls -lh best_comb.pt best_shank.pt
```

## 2. Confirm the Jetson software stack

Use the JetPack-provided CUDA, TensorRT, OpenCV, PyTorch, and torchvision
packages. Do not install the generic PyPI PyTorch wheel on Jetson.

```bash
python3 --version
cat /etc/nv_tegra_release
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

`torch.cuda.is_available()` must print `True` before continuing.

## 3. Prepare the application environment

```bash
cd /home/ubuntu/Documents/chicken_grading_system/chicken_grading_simulator
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
python -m pip install --no-deps ultralytics==8.4.90 ultralytics-thop==2.0.20
python scripts/check_ubuntu_runtime.py
```

## 4. Build FP16 TensorRT engines on the Orin

TensorRT engines are tied to the target GPU and TensorRT version. Always build
them on the Orin that will run the grading system. Do not commit `.engine`
files to Git.

```bash
cd /home/ubuntu/Documents/chicken_grading_system/chicken_grading_simulator
source .venv/bin/activate

python - <<'PY'
from ultralytics import YOLO

YOLO("best_comb.pt").export(
    format="engine",
    quantize=16,
    imgsz=960,
    device=0,
)

YOLO("best_shank.pt").export(
    format="engine",
    quantize=16,
    imgsz=640,
    device=0,
)
PY
```

Confirm the output:

```bash
ls -lh best_comb.engine best_shank.engine
```

## 5. Make the application prefer TensorRT with PT fallback

Update `src/dual_camera.py` so `MODEL_PATHS` prefers a local engine but still
falls back to the committed PT model:

```python
def preferred_model(stem: str) -> Path:
    engine = BASE_DIR / f"{stem}.engine"
    return engine if engine.is_file() else BASE_DIR / f"{stem}.pt"


MODEL_PATHS = {
    "comb": preferred_model("best_comb"),
    "shank": preferred_model("best_shank"),
}
```

Keep the existing `YOLO(path)` and `model.predict(...)` calls. Ultralytics uses
the TensorRT runtime automatically when `path` ends in `.engine`.

Update `scripts/check_ubuntu_runtime.py` to report both the selected runtime
model and the retained PT source model. The PT files remain required because
they are the rebuild and rollback source.

## 6. Validate results before production use

Start at 15 FPS:

```bash
CHICKEN_REALTIME_INFERENCE_FPS=15 streamlit run src/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --browser.gatherUsageStats=false \
  --server.headless=true
```

For at least 20 representative chickens, compare the PT and FP16 results:

- Comb segmentation overlay and area
- Shank segmentation overlay, width, and length
- Preliminary grade and light output
- Preview FPS and final inference latency
- GPU memory, temperature, and throttling from `tegrastats`

```bash
sudo tegrastats
```

Only increase to 30 FPS after both 15 FPS streams remain stable during final
capture, scale collection, database writes, and light control.

## 7. Roll back to PT

Stop Streamlit, rename the engines, and restart. The fallback code will select
the PT models automatically.

```bash
mv best_comb.engine best_comb.engine.disabled
mv best_shank.engine best_shank.engine.disabled
```

Rebuild the engines whenever JetPack, CUDA, TensorRT, the Jetson module, or the
PT model changes.
