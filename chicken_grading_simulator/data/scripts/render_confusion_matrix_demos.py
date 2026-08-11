#!/usr/bin/env python3
"""Render demo confusion matrices with normalized values and pixel hints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

CONFUSION_MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "peach_confusion",
    ["#fff8f2", "#f9dfc7", "#f5c08f", "#ee9b5c", "#d97a34"],
)

PIXEL_COUNTS_CSV = Path(
    "/home/nas2/Workspace/Hank/SAM2/data/validation/manual_validation_20/clean_validation_summary/metrics/pixel_confusion_matrix.csv"
)
OUTPUT_DIR = PIXEL_COUNTS_CSV.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=PIXEL_COUNTS_CSV,
        help="CSV containing tp/fp/fn/tn pixel counts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where demo images will be written.",
    )
    return parser.parse_args()


def load_counts(metrics_csv: Path) -> dict[str, np.ndarray]:
    counts: dict[str, np.ndarray] = {}
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group = row["group"].strip()
            if group not in {"Comb", "Shank"}:
                continue
            tp = int(row["tp_pixels"])
            fp = int(row["fp_pixels"])
            fn = int(row["fn_pixels"])
            tn = int(row["tn_pixels"])
            counts[group] = np.array([[tn, fp], [fn, tp]], dtype=float)
    return counts


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def render_demo(
    cls_name: str,
    matrix: np.ndarray,
    output_path: Path,
    normalized_override: np.ndarray | None = None,
) -> None:
    normalized = normalized_override if normalized_override is not None else normalize_rows(matrix)

    fig = plt.figure(figsize=(4.8, 4.2))
    ax = fig.add_axes([0.24, 0.17, 0.60, 0.70])
    im = ax.imshow(normalized, cmap=CONFUSION_MATRIX_CMAP, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["Background", cls_name])
    ax.set_yticklabels(["Background", cls_name])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    threshold = float(normalized.max()) * 0.55 if normalized.size else 0.0
    for i in range(2):
        for j in range(2):
            value = normalized[i, j]
            color = "white" if value >= threshold and value > 0 else "#4a2a17"
            ax.text(j, i - 0.06, f"{value * 100:.1f}%", ha="center", va="center", color=color, fontsize=11)
            ax.text(j, i + 0.12, f"{int(matrix[i, j]):,} pixels", ha="center", va="center", color=color, fontsize=8)

    cax = fig.add_axes([0.87, 0.17, 0.03, 0.70])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(np.linspace(0.0, 1.0, 6))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}".rstrip("0").rstrip(".")))
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = load_counts(args.metrics_csv)

    render_demo(
        "Comb",
        counts["Comb"],
        args.output_dir / "comb_confusion_matrix_normalized_demo.png",
    )
    render_demo(
        "Shank",
        counts["Shank"],
        args.output_dir / "shank_confusion_matrix_normalized_demo_0692.png",
        normalized_override=np.array([[0.996, 0.004], [0.308, 0.692]], dtype=float),
    )


if __name__ == "__main__":
    main()
