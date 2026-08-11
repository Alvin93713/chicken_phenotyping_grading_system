# Chicken Phenotyping Grading System

This Streamlit application provides a local workflow for chicken phenotype measurement and grading.

## Main Workflow

1. Identify each chicken by QR code or manual chicken ID input.
2. Capture phenotype data using a camera and calibrated scale.
3. Run YOLO segmentation inference for comb and shank measurements.
4. Store results in the local SQLite database.
5. Apply configurable grading thresholds.
6. Output pass, standby, or fail results through the UI and Arduino-controlled lights.

## Hardware Integration

The system is designed to work with:

- USB QR code scanner
- External camera
- Arduino Leonardo
- Physical buttons
- Green, yellow, and red indicator lights
- Optional buzzer

## Data Storage

The local SQLite database is stored at:

```text
data/chicken_phenotype.db
```

Large model files, videos, packaged executables, and inference output folders are not included in this GitHub repository.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For deployment on another machine, prepare the required model weights and hardware-specific configuration separately.
