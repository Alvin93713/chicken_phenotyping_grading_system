"""YOLO segmentation labels saved next to each inference result, for vendor fine-tuning.

Convention (also used when labels are produced by hand for older reviews):
    the label file sits next to the result image, same name, ".txt" extension
    e.g. outputs/<job>/samples/comb.png -> outputs/<job>/samples/comb.txt
Format: one object per line, "<class_id> x1 y1 x2 y2 ..." with coordinates normalised to 0..1
(Ultralytics YOLO-seg format).
"""
from __future__ import annotations

import json
from pathlib import Path

CLASS_NAMES_FILENAME = "class_names.json"
DEFAULT_CLASS_NAMES = {"comb": {0: "comb"}, "shank": {0: "shank"}}


def label_path_for(result_path: str | Path) -> Path:
    return Path(result_path).with_suffix(".txt")


def format_yolo_seg(class_ids, polygons) -> str:
    """class_ids: iterable of ints; polygons: iterable of [[x, y], ...] normalised to 0..1."""
    lines = []
    for class_id, polygon in zip(class_ids, polygons):
        points = [(float(x), float(y)) for x, y in polygon]
        if len(points) < 3:
            continue  # a segment needs at least a triangle
        coords = " ".join(f"{min(1.0, max(0.0, v)):.6f}" for point in points for v in point)
        lines.append(f"{int(class_id)} {coords}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_result_labels(result, result_image_path: str | Path) -> Path | None:
    """Save every mask shown on the result image as a YOLO-seg label. Never raises."""
    try:
        masks, boxes = getattr(result, "masks", None), getattr(result, "boxes", None)
        if masks is None or boxes is None:
            return None
        class_ids = [int(value) for value in boxes.cls.detach().cpu().tolist()]
        polygons = [polygon.tolist() if hasattr(polygon, "tolist") else polygon for polygon in masks.xyn]
        path = label_path_for(result_image_path)
        path.write_bytes(format_yolo_seg(class_ids, polygons).encode("ascii"))  # "\n" on every OS
        return path
    except Exception:
        return None


def class_names_path(directory: str | Path) -> Path:
    return Path(directory) / CLASS_NAMES_FILENAME


def save_class_names(directory: str | Path, role: str, names) -> None:
    """Remember the model's class names per role so exported data.yaml matches the label ids."""
    try:
        path = class_names_path(directory)
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        items = names.items() if isinstance(names, dict) else enumerate(names)
        data[role] = {str(int(key)): str(value) for key, value in items}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_class_names(directory: str | Path, role: str) -> dict[int, str]:
    try:
        data = json.loads(class_names_path(directory).read_text(encoding="utf-8"))
        names = {int(key): str(value) for key, value in data.get(role, {}).items()}
        if names:
            return names
    except Exception:
        pass
    return dict(DEFAULT_CLASS_NAMES.get(role, {0: role}))


def data_yaml(role: str, names: dict[int, str]) -> str:
    lines = [
        f"# Ultralytics YOLO segmentation dataset for the {role} model.",
        "# train_ready/ = confirmed correct, use as-is.",
        "# to_relabel/  = fix labels first, then move image + label into train_ready/.",
        "# Merge into your own dataset and point 'val' at your held-out validation split.",
        "path: .",
        "train: train_ready/images",
        "val: train_ready/images",
        "names:",
    ]
    lines += [f"  {class_id}: {name}" for class_id, name in sorted(names.items())]
    return "\n".join(lines) + "\n"
