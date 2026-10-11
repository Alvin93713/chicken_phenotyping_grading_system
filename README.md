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

Large runtime assets are intentionally excluded from this repository, including model weights, videos, packaged executables, and inference outputs.

Required external assets may include:

- YOLO segmentation model weights, such as `best.pt`
- Local camera hardware
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
git switch codex/premium-ui-refresh
git pull --ff-only origin codex/premium-ui-refresh
git log -1 --oneline
```

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
  --branch codex/premium-ui-refresh \
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

The repository excludes large model files. Keep the existing YOLO and SAM
weights under `chicken_grading_simulator/` and
`chicken_grading_simulator/data/`; the repair does not remove untracked files.

## Intended Use

This project is intended for prototyping and field testing automated breeder chicken phenotype measurement and grading workflows.
