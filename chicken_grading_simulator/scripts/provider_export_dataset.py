"""Vendor tool: turn received review packages AND snapshots (the「回傳」zips) into fine-tuning datasets
(same layout as the farm's manual download bundle).

Usage:
    python scripts/provider_export_dataset.py --inbox /srv/chicken-review-inbox --out /srv/chicken-dataset

Output:
    <out>/<role>/data.yaml
    <out>/<role>/train_ready/images|labels   reviewed correct with the model's label -> train as-is
    <out>/<role>/to_relabel/images|labels    incorrect / uncertain / correct-but-unlabelled;
                                             labels = model prediction to correct, not draw from scratch
    <out>/<role>/to_relabel/review_notes.csv worst first, with the farm's note
    <out>/manifest.csv                       one row per image

When a farm re-reviews the same image, the latest review wins. Use a new --out folder per export
(e.g. dated). An existing non-empty --out is refused unless --overwrite is given, because
rebuilding it would discard any relabelling done inside it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

DECISION_CODES = {"正確": "correct", "不正確": "incorrect", "無法判斷": "uncertain"}
ROLES = ("comb", "shank")
SAFE = re.compile(r"[^A-Za-z0-9._-]+")
NOTES_FIELDS = ("image", "label", "decision", "issue", "reviewer_note", "confidence", "case_id", "captured_at", "reviewed_at")
MANIFEST_FIELDS = (
    "role", "split", "image", "label", "decision", "issue", "reviewer_note", "confidence",
    "site_id", "device_id", "case_id", "subject_token", "model", "measurements", "captured_at", "reviewed_at",
)


def _read_package(path: Path):
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
            files = {}
            for info in archive.infolist():
                name = PurePosixPath(info.filename)
                if len(name.parts) == 2 and name.parts[0] in ("images", "labels") and ".." not in name.parts:
                    files[f"{name.parts[0]}/{name.stem}"] = (name.suffix.lower(), archive.read(info))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return metadata, files


def _data_yaml(role: str, names: dict) -> str:
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
    lines += [f"  {int(key)}: {value}" for key, value in sorted(names.items(), key=lambda kv: int(kv[0]))]
    return "\n".join(lines) + "\n"


def _priority(row: dict) -> tuple:
    order = {"incorrect": 0, "uncertain": 1, "correct": 2}
    confidence = row["confidence"]
    return (order.get(row["decision"], 3), float(confidence) if confidence not in (None, "") else -1.0)


def _read_snapshot(path: Path):
    """A snapshot zip (one「回傳」press: manifest.json + <role>/<split>/images|labels + snapshot_info.json).
    Returns (info, manifest rows, class names per role) or None if the zip is not a snapshot."""
    try:
        with zipfile.ZipFile(path) as archive:
            rows = json.loads(archive.read("manifest.json").decode("utf-8"))
            try:
                info = json.loads(archive.read("snapshot_info.json").decode("utf-8"))
            except KeyError:
                info = {}
            names = {}
            for role in ROLES:
                try:
                    names[role] = _parse_yaml_names(archive.read(f"{role}/data.yaml").decode("utf-8"))
                except KeyError:
                    pass
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(rows, list):
        return None
    return info, rows, names


def _parse_yaml_names(text: str) -> dict:
    names, inside = {}, False
    for line in text.splitlines():
        if line.strip() == "names:":
            inside = True
        elif inside and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            if key.isdigit():
                names[key] = value.strip()
        elif inside and line.strip():
            break
    return names


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _collect(inbox: Path):
    """Every reviewed image the inbox holds, latest review per (site, device, case, role) winning.

    Two package formats arrive: per-image review packages (metadata.json, older devices) and snapshots
    (manifest.json; each one is the device's COMPLETE set of reviewed images at that moment, so only the
    newest snapshot per site/device is used). Each item carries a `load()` returning (image, label) bytes.
    """
    items: dict[tuple, dict] = {}
    class_names: dict[str, dict] = {}

    def offer(key, stamp, item):
        if key not in items or stamp >= items[key]["stamp"]:
            items[key] = {**item, "stamp": stamp}

    newest_snapshot: dict[tuple[str, str], tuple[str, Path, dict, list, dict]] = {}
    for package in sorted(inbox.rglob("*.zip")):
        parsed = _read_package(package)
        if parsed is not None:
            metadata, files = parsed
            role = str(metadata.get("role"))
            if role not in ROLES or "images/source" not in files:
                continue
            if metadata.get("class_names"):
                class_names[role] = metadata["class_names"]
            key = (str(metadata.get("site_id")), str(metadata.get("device_id")), str(metadata.get("case_id")), role)

            def load(files=files):
                image = files["images/source"]
                label = files.get("labels/prediction")
                return image[0], image[1], (label[1] if label else None)

            offer(key, str(metadata.get("reviewed_at") or ""), {
                "role": role, "site_id": key[0], "device_id": key[1], "case_id": metadata.get("case_id"),
                "decision": DECISION_CODES.get(str(metadata.get("decision")), "unknown"),
                "reviewer_note": metadata.get("reviewer_note") or "", "confidence": metadata.get("confidence"),
                "subject_token": metadata.get("subject_token"), "model": metadata.get("model"),
                "measurements": json.dumps(metadata.get("measurements") or {}, ensure_ascii=False),
                "captured_at": metadata.get("captured_at"), "reviewed_at": metadata.get("reviewed_at"),
                "load": load,
            })
            continue
        snapshot = _read_snapshot(package)
        if snapshot is None:
            continue
        info, rows, names = snapshot
        device = (str(info.get("site_id") or ""), str(info.get("device_id") or ""))
        stamp = str(info.get("packed_at") or info.get("requested_at") or "")
        if device not in newest_snapshot or stamp >= newest_snapshot[device][0]:
            newest_snapshot[device] = (stamp, package, info, rows, names)

    for (site_id, device_id), (_, package, _info, rows, names) in newest_snapshot.items():
        class_names.update(names)
        for row in rows:
            role = str(row.get("role"))
            image_member = str(row.get("image") or "")
            if role not in ROLES or not _safe_member(image_member):
                continue  # e.g. issue=missing_image: no photo in the snapshot
            label_member = str(row.get("label") or "")

            def load(package=package, image_member=image_member, label_member=label_member):
                with zipfile.ZipFile(package) as archive:
                    image = archive.read(image_member)
                    label = archive.read(label_member) if _safe_member(label_member) else None
                return PurePosixPath(image_member).suffix.lower(), image, label

            key = (site_id, device_id, str(row.get("case_id")), role)
            offer(key, str(row.get("reviewed_at") or ""), {
                "role": role, "site_id": site_id, "device_id": device_id, "case_id": row.get("case_id"),
                "decision": str(row.get("decision") or "unknown"),
                "reviewer_note": row.get("reviewer_note") or "", "confidence": row.get("confidence"),
                "subject_token": row.get("chicken_id"), "model": row.get("model"),
                "measurements": row.get("measurements") or "{}",
                "captured_at": row.get("captured_at"), "reviewed_at": row.get("reviewed_at"),
                "load": load,
            })
    return items, class_names


def export(inbox: Path, out: Path, overwrite: bool = False) -> int:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{out} is not empty. Use a new output folder, or pass --overwrite to rebuild it "
            "(this discards any relabelling done inside it)."
        )
    items, found_names = _collect(inbox)

    if overwrite:
        for role in ROLES:
            shutil.rmtree(out / role, ignore_errors=True)
    class_names: dict[str, dict] = {role: {"0": role} for role in ROLES}
    class_names.update(found_names)
    rows = []
    for key, item in sorted(items.items()):
        role, site_id, device_id = item["role"], item["site_id"], item["device_id"]
        image_suffix, image_bytes, label_bytes = item["load"]()
        has_label = label_bytes is not None
        decision = item["decision"]
        split = "train_ready" if decision == "correct" and has_label else "to_relabel"
        stem = SAFE.sub("_", f"{site_id}_{device_id}_{item['case_id']}_{role}")
        image_rel = f"{role}/{split}/images/{stem}{image_suffix}"
        (out / image_rel).parent.mkdir(parents=True, exist_ok=True)
        (out / image_rel).write_bytes(image_bytes)
        label_rel = ""
        if has_label:
            label_rel = f"{role}/{split}/labels/{stem}.txt"
            (out / label_rel).parent.mkdir(parents=True, exist_ok=True)
            (out / label_rel).write_bytes(label_bytes)
        rows.append(
            {
                "role": role,
                "split": split,
                "image": image_rel,
                "label": label_rel,
                "decision": decision,
                "issue": "" if has_label else "missing_label",
                "reviewer_note": item["reviewer_note"],
                "confidence": item["confidence"],
                "site_id": site_id,
                "device_id": device_id,
                "case_id": item["case_id"],
                "subject_token": item["subject_token"],
                "model": item["model"],
                "measurements": item["measurements"],
                "captured_at": item["captured_at"],
                "reviewed_at": item["reviewed_at"],
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        for split in ("train_ready", "to_relabel"):
            for kind in ("images", "labels"):
                (out / role / split / kind).mkdir(parents=True, exist_ok=True)
        (out / role / "data.yaml").write_text(_data_yaml(role, class_names[role]), encoding="utf-8")
        notes = sorted((r for r in rows if r["role"] == role and r["split"] == "to_relabel"), key=_priority)
        with (out / role / "to_relabel" / "review_notes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=NOTES_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in notes:
                writer.writerow({**row, "image": Path(row["image"]).name, "label": Path(row["label"]).name if row["label"] else ""})
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inbox", required=True, type=Path, help="CHICKEN_PROVIDER_DATA_DIR of the receiver")
    parser.add_argument("--out", required=True, type=Path, help="new (empty) output folder, e.g. /srv/chicken-dataset/2026-11-25")
    parser.add_argument("--overwrite", action="store_true", help="rebuild a non-empty --out folder")
    args = parser.parse_args(argv)
    try:
        count = export(args.inbox, args.out, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"error: {exc}")
        return 2
    print(f"Exported {count} reviewed images to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
