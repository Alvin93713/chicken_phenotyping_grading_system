"""Compatibility entry point for launching the packaged app."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

runpy.run_path(str(SRC / "launcher.py"), run_name="__main__")
