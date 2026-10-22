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

The deployment branch includes `best.pt` and `best_shank.pt`. Videos, packaged executables, captured photos, and inference output folders are not included.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Ubuntu Runtime Check

After installing dependencies, run:

```bash
python scripts/check_ubuntu_runtime.py
```

For x86_64 Ubuntu, install `requirements.txt`. For Jetson Orin, keep the
JetPack-compatible NVIDIA PyTorch/OpenCV installation and follow
`requirements-jetson.txt`; do not replace NVIDIA PyTorch with a generic wheel.
