# chicken-phenotyping-grading-system

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
- Manually entered grading threshold
- Arduino Leonardo + 12V power supply integration for physical lights and buttons
- Local Streamlit browser interface

## Repository Notes

Videos, packaged executables, and inference outputs are intentionally excluded from this repository. The deployment branch includes the two model files required by the dual-camera workflow.

Required runtime assets include:

- `chicken_grading_simulator/best.pt` for comb segmentation
- `chicken_grading_simulator/best_shank.pt` for shank segmentation
- Two local USB cameras
- USB QR code scanner
- Arduino Leonardo
- 12V power supply + physical light
- button controller

## Basic Usage

Install dependencies:

```bash
pip install -r chicken_grading_simulator/requirements.txt
```

Run the Streamlit app:

```bash
cd chicken_grading_simulator
streamlit run src/app.py
```

Then open the local browser URL shown by Streamlit.

## Update the System on Ubuntu / Jetson Orin

The deployed Git repository is expected at:

```text
/home/ubuntu/Documents/chicken_grading_system
```

Stop the running Streamlit process before updating. Confirm that there are no
unexpected local source-code changes, then update the deployment branch:

```bash
cd /home/ubuntu/Documents/chicken_grading_system

git status --short
git fetch origin
git switch codex/dual-camera-snapshot
git pull --ff-only origin codex/dual-camera-snapshot
git log -1 --oneline
```

### One-time Ubuntu / Jetson environment

Grant camera and serial-device access, then sign out and sign in again:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0 libzbar0 v4l-utils usbutils
sudo usermod -aG dialout,video "$USER"
```

For a regular x86_64 Ubuntu PC:

```bash
cd /home/ubuntu/Documents/chicken_grading_system/chicken_grading_simulator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Jetson Orin, use the NVIDIA PyTorch and OpenCV supplied for the installed
JetPack release. Make the virtual environment inherit those system packages:

```bash
cd /home/ubuntu/Documents/chicken_grading_system/chicken_grading_simulator
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
python -m pip install --no-deps ultralytics==8.4.90 ultralytics-thop==2.0.20
python scripts/check_ubuntu_runtime.py
```

Do not install generic PyPI `torch`, `torchvision`, or `opencv-python` over the
JetPack builds. Their CUDA and camera support must match the JetPack release.

If `git status --short` reports modified source files, review or back them up
before pulling. Do not use `git reset --hard` on a field system containing
measurement data.

Start the updated system:

```bash
cd /home/ubuntu/Documents/chicken_grading_system/chicken_grading_simulator
source .venv/bin/activate

streamlit run src/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --browser.gatherUsageStats=false \
  --server.headless=true
```

Before starting the service, verify the Python environment, both segmentation
models, camera devices, and serial devices:

```bash
python scripts/check_ubuntu_runtime.py
```

The checker exits with a non-zero status when a required import or model is
missing. Cameras and serial devices are reported for field verification because
their device numbers depend on the connected Ubuntu machine.

Open `http://localhost:8501` on the Orin, or use
`http://<ORIN_IP_ADDRESS>:8501` from another device on the same network.

### One-Time Repair When `.git` Is Missing

The following error means that the deployed folder was copied or extracted
without Git metadata, or that the command was run outside the repository:

```text
fatal: not a git repository (or any of the parent directories): .git
```

First, check whether a repository already exists elsewhere:

```bash
find /home/ubuntu/Documents -maxdepth 5 -type d -name .git -print
```

If the command prints a path, change to its parent directory and run the normal
update flow above. If it prints nothing, use the following one-time repair. It
keeps the existing virtual environment, untracked model weights, captured
media, and inference outputs in place. The SQLite database is backed up before
tracked files are synchronized and restored afterward.

```bash
ROOT=/home/ubuntu/Documents/chicken_grading_system
DB="$ROOT/chicken_grading_simulator/data/chicken_phenotype.db"
DB_BACKUP="$HOME/chicken_phenotype_$(date +%Y%m%d_%H%M%S).db"

cp -a "$DB" "$DB_BACKUP"

TMPDIR=$(mktemp -d)
git clone \
  --branch codex/dual-camera-snapshot \
  --single-branch \
  https://github.com/Alvin93713/chicken_phenotyping_grading_system.git \
  "$TMPDIR/repository"

cp -a "$TMPDIR/repository/.git" "$ROOT/.git"

cd "$ROOT"
git restore --source=HEAD --staged --worktree -- .

cp -a "$DB_BACKUP" "$DB"
git update-index --skip-worktree \
  chicken_grading_simulator/data/chicken_phenotype.db

git status --short --branch
git log -1 --oneline
```

The dual-camera deployment branch tracks `best.pt` and `best_shank.pt`. Other
large experimental model weights remain excluded.

## Intended Use

This project is intended for prototyping and field testing automated breeder chicken phenotype measurement and grading workflows.
