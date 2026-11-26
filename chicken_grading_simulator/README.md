# Chicken Phenotyping Grading System

This Streamlit application provides a local workflow for chicken phenotype measurement and grading.

## Main Workflow

1. Identify each chicken by QR code or manual chicken ID input.
2. Freeze a two-second average weight from the RS485/USB scale.
3. Capture one still image from each calibrated comb and shank camera.
4. Run dedicated YOLO segmentation inference for comb and shank measurements.
5. Store the combined result in the local SQLite database.
6. Apply configurable grading thresholds.
7. Output pass, standby, or fail results through the UI and Arduino-controlled lights.

## Secure Review Sync

The product review workflow can queue independently reviewed comb and shank
images for secure background delivery to the provider. Incorrect and uncertain
results, as well as correct results, are all queued for delivery.
Transport requires HTTPS, certificate verification, HMAC-SHA256 request
signatures, timestamps, nonces, and SHA-256 content verification. Chicken IDs
are pseudonymized before transmission. See `SECURE_SYNC.md` for deployment and
secret configuration.

## Hardware Integration

The system is designed to work with:

- USB QR code scanner
- Two USB cameras
- RS485/USB scale using 19200 bps and STX/ASCII/ETX packets
- Arduino Leonardo
- Physical buttons
- Green, yellow, and red indicator lights

## Data Storage

The local SQLite database is stored at:

```text
data/chicken_phenotype.db
```

The deployment branch includes `best_comb.pt` and `best_shank.pt`. Videos, packaged executables, captured photos, and inference output folders are not included.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Ubuntu Runtime Check

For Jetson Orin, first use the NVIDIA PyTorch, torchvision, CUDA, TensorRT, and
OpenCV builds supplied for the installed JetPack release. Then install the app
dependencies without replacing NVIDIA's packages:

```bash
cd /home/ubuntu/Documents/chicken_grading_system/chicken_grading_simulator
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
python -m pip install --no-deps ultralytics==8.4.90 ultralytics-thop==2.0.20
```

After installing dependencies, validate the environment, models, CUDA, and
visible hardware:

```bash
python scripts/check_ubuntu_runtime.py
```

Start the system with:

```bash
CHICKEN_REALTIME_INFERENCE_FPS=15 streamlit run src/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --browser.gatherUsageStats=false \
  --server.headless=true
```

For x86_64 Ubuntu, install `requirements.txt`. Do not use `requirements.txt`
on Jetson because it would replace JetPack-compatible NVIDIA packages with
generic wheels.
