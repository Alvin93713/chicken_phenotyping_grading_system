# chicken_phenotyping_grading_system

A local chicken phenotyping and grading system built with Streamlit.

The system supports QR-based chicken identification, phenotype data entry, camera-based image capture, YOLO segmentation inference for comb and shank measurements, scale calibration, SQLite data management, configurable grading thresholds, and Arduino-controlled light/buzzer feedback.

## Key Features

- QR code based chicken ID workflow
- Manual and camera-assisted phenotype data collection
- YOLO segmentation inference for comb and shank traits
- Scale calibration for pixel-to-centimeter conversion
- Queue-based inference workflow
- SQLite database storage
- Three-level grading result: pass, standby, fail
- Arduino Leonardo integration for physical lights, buzzer, and buttons
- Local Streamlit browser interface

## Repository Notes

Large runtime assets are intentionally excluded from this repository, including model weights, videos, packaged executables, and inference outputs.

Required external assets may include:

- YOLO segmentation model weights, such as `best.pt`
- Optional SAM2 weights, such as `sam2_b.pt`
- Local camera hardware
- USB QR code scanner
- Arduino Leonardo based light and button controller

## Basic Usage

Install dependencies:

```bash
pip install -r chicken_grading_simulator/requirements.txt
```

Run the Streamlit app:

```bash
cd chicken_grading_simulator
streamlit run app.py
```

Then open the local browser URL shown by Streamlit.

## Intended Use

This project is intended for prototyping and field testing automated breeder chicken phenotype measurement and grading workflows.
