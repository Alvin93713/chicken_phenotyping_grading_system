"""Product review queue for vendor-managed active learning."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import zipfile
from contextlib import closing
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
REVIEW_DIR = BASE_DIR / "data" / "product_review"
REVIEW_DB_PATH = REVIEW_DIR / "product_review.db"
DEMO_ASSET_DIR = REVIEW_DIR / "demo_assets"
DEFAULT_REVIEW_CONFIDENCE = float(os.environ.get("CHICKEN_REVIEW_CONFIDENCE", "0.70"))
DEFAULT_QA_SAMPLE_RATE = float(os.environ.get("CHICKEN_REVIEW_SAMPLE_RATE", "0.05"))
DECISIONS = ("待審核", "正確", "不正確", "無法判斷")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_SCHEMA_READY: set[str] = set()


def get_connection() -> sqlite3.Connection:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    key = str(REVIEW_DB_PATH)
    ready = key in _SCHEMA_READY and Path(REVIEW_DB_PATH).exists()
    connection = sqlite3.connect(REVIEW_DB_PATH)
    connection.row_factory = sqlite3.Row
    if not ready:  # schema / migration only once per database file per process
        _ensure_schema(connection)
        _SCHEMA_READY.add(key)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS review_cases (
            case_id TEXT PRIMARY KEY,
            job_id TEXT UNIQUE NOT NULL,
            chicken_id TEXT NOT NULL,
            pipeline_status TEXT NOT NULL,
            review_state TEXT NOT NULL DEFAULT '待審核',
            priority TEXT NOT NULL,
            reason TEXT NOT NULL,
            confidence_threshold REAL,
            minimum_confidence REAL,
            role_confidences_json TEXT NOT NULL,
            measurements_json TEXT NOT NULL,
            source_paths_json TEXT NOT NULL,
            result_paths_json TEXT NOT NULL,
            models_json TEXT NOT NULL,
            error TEXT,
            reviewer_note TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT
        )
        """
    )
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(review_cases)").fetchall()
    }
    for column in ("comb_review_state", "shank_review_state"):
        if column not in columns:
            connection.execute(
                f"ALTER TABLE review_cases ADD COLUMN {column} TEXT NOT NULL DEFAULT '待審核'"
            )
            connection.execute(
                f"UPDATE review_cases SET {column} = review_state"
            )
    # One note per image: a COMB note must not be overwritten by the SHANK review (older rows: copy the
    # single case note into both).
    for column in ("comb_reviewer_note", "shank_reviewer_note"):
        if column not in columns:
            connection.execute(f"ALTER TABLE review_cases ADD COLUMN {column} TEXT")
            connection.execute(f"UPDATE review_cases SET {column} = reviewer_note")
    connection.commit()


def role_note(case: dict[str, object], role: str) -> str:
    """The reviewer's note for one image (comb or shank) of a case."""
    return str(case.get(f"{role}_reviewer_note") or "")


def _qa_sample_selected(job_id: str, rate: float) -> bool:
    rate = min(1.0, max(0.0, float(rate)))
    value = int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < rate


def record_case(
    *,
    job_id: str,
    chicken_id: str,
    pipeline_status: str,
    confidence_threshold: float,
    role_confidences: dict[str, float | None],
    measurements: dict[str, object] | None,
    source_paths: dict[str, str],
    result_paths: dict[str, str],
    models: dict[str, str],
    error: str = "",
    force: bool = False,
    review_confidence: float = DEFAULT_REVIEW_CONFIDENCE,
    sample_rate: float = DEFAULT_QA_SAMPLE_RATE,
    flag_reason: str = "",
) -> bool:
    """flag_reason: a measurement check that failed (e.g. the two legs disagree) — always sent to review."""
    valid_confidences = [float(value) for value in role_confidences.values() if value is not None]
    minimum_confidence = min(valid_confidences) if valid_confidences else None
    failed = pipeline_status == "失敗" or bool(error)
    low_confidence = minimum_confidence is None or minimum_confidence < float(review_confidence)
    sampled = _qa_sample_selected(job_id, sample_rate)
    if not (force or failed or low_confidence or flag_reason or sampled):
        return False

    if failed:
        priority, reason = "高", "推論失敗"
    elif flag_reason:
        priority, reason = "高", flag_reason
    elif low_confidence:
        priority, reason = "高", f"最低信心 {minimum_confidence:.2f}" if minimum_confidence is not None else "缺少信心分數"
    else:
        priority, reason = "一般", "例行品質抽樣"

    timestamp = now_text()
    with closing(get_connection()) as connection, connection:
        connection.execute(
            """
            INSERT INTO review_cases (
                case_id, job_id, chicken_id, pipeline_status, review_state,
                priority, reason, confidence_threshold, minimum_confidence,
                role_confidences_json, measurements_json, source_paths_json,
                result_paths_json, models_json, error, created_at
            ) VALUES (?, ?, ?, ?, '待審核', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                pipeline_status = excluded.pipeline_status,
                priority = excluded.priority,
                reason = excluded.reason,
                confidence_threshold = excluded.confidence_threshold,
                minimum_confidence = excluded.minimum_confidence,
                role_confidences_json = excluded.role_confidences_json,
                measurements_json = excluded.measurements_json,
                source_paths_json = excluded.source_paths_json,
                result_paths_json = excluded.result_paths_json,
                models_json = excluded.models_json,
                error = excluded.error
            """,
            (
                job_id,
                job_id,
                chicken_id,
                pipeline_status,
                priority,
                reason,
                float(confidence_threshold),
                minimum_confidence,
                json.dumps(role_confidences, ensure_ascii=False),
                json.dumps(measurements or {}, ensure_ascii=False),
                json.dumps(source_paths, ensure_ascii=False),
                json.dumps(result_paths, ensure_ascii=False),
                json.dumps(models, ensure_ascii=False),
                error,
                timestamp,
            ),
        )
    return True


def record_snapshot_review(job, output_folder: Path, measurements=None, error: str = "") -> bool:
    metadata_path = output_folder / "metadata.json"
    metadata = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    cameras = metadata.get("cameras", {})
    role_confidences = {
        role: cameras.get(role, {}).get("detection_confidence")
        for role in ("comb", "shank")
    }
    source_paths = dict(job.snapshot_paths or {})
    result_paths = {
        role: str(output_folder / "samples" / f"{role}.png")
        for role in ("comb", "shank")
        if (output_folder / "samples" / f"{role}.png").is_file()
    }
    models = {
        role: str(cameras.get(role, {}).get("model", ""))
        for role in ("comb", "shank")
    }
    leg_check = cameras.get("shank", {}).get("leg_check") or {}
    flag_reason = ""
    if leg_check.get("disagree"):
        difference = max((leg_check.get("difference") or {}).values(), default=0.0)
        flag_reason = f"兩腳差異 {difference * 100:.0f}%"
    return record_case(
        flag_reason=flag_reason,
        job_id=job.job_id,
        chicken_id=job.chicken_id,
        pipeline_status="失敗" if error else "已完成",
        confidence_threshold=job.confidence_threshold,
        role_confidences=role_confidences,
        measurements=measurements,
        source_paths=source_paths,
        result_paths=result_paths,
        models=models,
        error=error,
    )


def _decode_row(row: sqlite3.Row) -> dict[str, object]:
    item = dict(row)
    for column in (
        "role_confidences_json",
        "measurements_json",
        "source_paths_json",
        "result_paths_json",
        "models_json",
    ):
        item[column.removesuffix("_json")] = json.loads(item.pop(column) or "{}")
    return item


def list_cases(review_state: str | None = None) -> list[dict[str, object]]:
    query = "SELECT * FROM review_cases"
    parameters: tuple[object, ...] = ()
    if review_state and review_state != "全部":
        query += " WHERE review_state = ?"
        parameters = (review_state,)
    query += " ORDER BY CASE priority WHEN '高' THEN 0 ELSE 1 END, created_at DESC"
    with closing(get_connection()) as connection, connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_decode_row(row) for row in rows]


def get_case(case_id: str) -> dict[str, object] | None:
    with closing(get_connection()) as connection, connection:
        row = connection.execute("SELECT * FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
    return _decode_row(row) if row else None


def _combined_review_state(comb_state: str, shank_state: str) -> str:
    states = {comb_state, shank_state}
    if "不正確" in states:
        return "不正確"
    if "無法判斷" in states:
        return "無法判斷"
    if states == {"正確"}:
        return "正確"
    return "待審核"


def review_case_role(case_id: str, role: str, decision: str, note: str = "") -> None:
    if role not in ("comb", "shank"):
        raise ValueError(f"不支援的影像類型：{role}")
    if decision not in DECISIONS:
        raise ValueError(f"不支援的審核結果：{decision}")
    with closing(get_connection()) as connection, connection:
        row = connection.execute(
            "SELECT comb_review_state, shank_review_state FROM review_cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"找不到審核案例：{case_id}")
        states = {
            "comb": row["comb_review_state"],
            "shank": row["shank_review_state"],
        }
        states[role] = decision
        combined_state = _combined_review_state(states["comb"], states["shank"])
        cursor = connection.execute(
            f"""
            UPDATE review_cases
            SET {role}_review_state = ?, review_state = ?,
                {role}_reviewer_note = ?, reviewer_note = ?, reviewed_at = ?
            WHERE case_id = ?
            """,
            (decision, combined_state, note.strip(), note.strip(), now_text(), case_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"找不到審核案例：{case_id}")
    # Nothing is uploaded per review: the farm sends a complete snapshot of all reviewed images
    # when they press「回傳」(see secure_sync.request_snapshot).


def review_case(case_id: str, decision: str, note: str = "") -> None:
    """Apply a legacy whole-case decision to both camera results."""
    review_case_role(case_id, "comb", decision, note)
    review_case_role(case_id, "shank", decision, note)


def review_summary() -> dict[str, int]:
    with closing(get_connection()) as connection, connection:
        rows = connection.execute(
            "SELECT review_state, COUNT(*) AS count FROM review_cases GROUP BY review_state"
        ).fetchall()
    counts = {state: 0 for state in DECISIONS}
    counts.update({row["review_state"]: int(row["count"]) for row in rows})
    counts["總案例"] = sum(counts.values())
    return counts


EXPORT_STATES = ("正確", "無法判斷", "不正確")


def _exported_roles(case: dict[str, object], states) -> list[str]:
    return [role for role in ("comb", "shank") if case[f"{role}_review_state"] in states]


def export_counts(states=EXPORT_STATES, cases=None) -> dict[str, int]:
    """Count reviewed images (per comb/shank role) that the vendor bundle will contain.
    Pass `cases` (from list_cases()) to reuse an already-loaded list."""
    counts = {state: 0 for state in states}
    for case in list_cases() if cases is None else cases:
        for role in _exported_roles(case, states):
            counts[case[f"{role}_review_state"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Vendor fine-tuning bundle
#   <role>/data.yaml
#   <role>/train_ready/images|labels   reviewed correct AND has a label -> train as-is
#   <role>/to_relabel/images|labels    incorrect / uncertain / correct-but-unlabelled -> fix first
#   <role>/to_relabel/review_notes.csv what to look at, worst first
#   manifest.csv, manifest.json, README.txt
# ---------------------------------------------------------------------------
DECISION_CODES = {"正確": "correct", "無法判斷": "uncertain", "不正確": "incorrect"}
SPLIT_TRAIN_READY = "train_ready"
SPLIT_TO_RELABEL = "to_relabel"
BUNDLE_MANIFEST_FIELDS = (
    "role",
    "split",
    "image",
    "label",
    "decision",
    "issue",
    "reviewer_note",
    "confidence",
    "case_id",
    "chicken_id",
    "reason",
    "model",
    "measurements",
    "captured_at",
    "reviewed_at",
)
REVIEW_NOTES_FIELDS = (
    "image",
    "label",
    "decision",
    "issue",
    "reviewer_note",
    "confidence",
    "case_id",
    "captured_at",
    "reviewed_at",
)
BUNDLE_README = """chicken_review_bundle - vendor fine-tuning package
====================================================
One folder per model (comb/, shank/). Each is an Ultralytics YOLO segmentation dataset.

<role>/train_ready/   Reviewed CORRECT by the farm. images/ + labels/ (the model's masks,
                      confirmed by a person). Add to your training set as-is.
<role>/to_relabel/    Reviewed INCORRECT or UNCERTAIN (plus correct images that have no saved
                      label). labels/ holds the model's prediction as a pre-annotation.
                      1. Import images/ + labels/ into CVAT / Label Studio / Roboflow (YOLO format).
                      2. Fix or delete polygons; discard unusable images.
                         An image with nothing to detect can keep an empty label (negative sample).
                      3. Export YOLO and move image + label into train_ready/.
                      review_notes.csv lists every image, worst first, with the farm's note.
<role>/data.yaml      Class names matching the label ids.

File names: <case_id>_<role>.<ext>; image and label share the same stem.
manifest.csv / manifest.json: one row per image (decision, split, note, confidence, times).

Decisions: correct = 正確, incorrect = 不正確, uncertain = 無法判斷.
issue: missing_label = no saved label (create one before training);
       missing_image = source photo no longer on the device (not included).
"""


def _bundle_entries(states=EXPORT_STATES, cases=None) -> list[dict[str, object]]:
    """One entry per reviewed comb/shank image, with its destination in the bundle."""
    from training_labels import label_path_for

    entries = []
    for case in list_cases() if cases is None else cases:
        for role in _exported_roles(case, states):
            decision = case[f"{role}_review_state"]
            source_text = case["source_paths"].get(role, "")
            result_text = case["result_paths"].get(role, "")
            source = Path(source_text) if source_text else None
            label = label_path_for(result_text) if result_text else None
            has_image = bool(source and source.is_file())
            has_label = bool(label and label.is_file())
            if not has_image:
                issue = "missing_image"
            elif not has_label:
                issue = "missing_label"
            else:
                issue = ""
            split = SPLIT_TRAIN_READY if decision == "正確" and has_label and has_image else SPLIT_TO_RELABEL
            stem = f"{case['case_id']}_{role}"
            entries.append(
                {
                    "case": case,
                    "role": role,
                    "decision": DECISION_CODES.get(decision, decision),
                    "split": split if has_image else "",
                    "issue": issue,
                    "source": source if has_image else None,
                    "label": label if has_label else None,
                    "image_name": f"{role}/{split}/images/{stem}{source.suffix.lower()}" if has_image else "",
                    "label_name": f"{role}/{split}/labels/{stem}.txt" if has_image and has_label else "",
                }
            )
    return entries


def _relabel_priority(entry: dict[str, object]) -> tuple:
    order = {"incorrect": 0, "uncertain": 1, "correct": 2}
    confidence = entry["case"]["role_confidences"].get(entry["role"])
    return (order.get(str(entry["decision"]), 3), float(confidence) if confidence is not None else -1.0)


def _csv_bytes(fields, rows) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def export_review_bundle(states=EXPORT_STATES) -> bytes:
    """In-memory bundle for the on-site「下載審核資料包」button (keeps real chicken IDs)."""
    buffer = io.BytesIO()
    write_review_bundle(buffer, states)
    return buffer.getvalue()


def write_review_bundle(target, states=EXPORT_STATES, subject_id=None, extra_json=None) -> dict[str, int]:
    """Write the fine-tuning bundle of every reviewed comb/shank image to a path or file object.

    Correct images with a saved label go to <role>/train_ready/; incorrect, uncertain and
    unlabelled images go to <role>/to_relabel/ with the model's prediction as a pre-annotation.
    Images still pending review are left out.

    subject_id: optional callable mapping a chicken ID to a pseudonym (used for network uploads so
    real chicken IDs never leave the farm). extra_json: {archive name: JSON-serialisable object}.
    Returns the bundle_summary-style counts of what was written.
    """
    from training_labels import data_yaml, load_class_names

    entries = _bundle_entries(states)
    manifest_rows = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for role in ("comb", "shank"):
            for split in (SPLIT_TRAIN_READY, SPLIT_TO_RELABEL):
                for kind in ("images", "labels"):
                    archive.writestr(f"{role}/{split}/{kind}/", b"")
            archive.writestr(f"{role}/data.yaml", data_yaml(role, load_class_names(REVIEW_DIR, role)))

        for entry in entries:
            case = entry["case"]
            if entry["source"] is not None:
                # Photos are already compressed: store them as-is to save CPU on the device.
                archive.write(entry["source"], entry["image_name"], compress_type=zipfile.ZIP_STORED)
            if entry["label"] is not None:
                archive.write(entry["label"], entry["label_name"])
            chicken = str(case["chicken_id"])
            manifest_rows.append(
                {
                    "role": entry["role"],
                    "split": entry["split"],
                    "image": entry["image_name"],
                    "label": entry["label_name"],
                    "decision": entry["decision"],
                    "issue": entry["issue"],
                    "reviewer_note": role_note(case, entry["role"]),
                    "confidence": case["role_confidences"].get(entry["role"]),
                    "case_id": case["case_id"],
                    "chicken_id": subject_id(chicken) if subject_id else chicken,
                    "reason": case["reason"],
                    "model": Path(str(case["models"].get(entry["role"], ""))).name,
                    "measurements": json.dumps(case["measurements"] or {}, ensure_ascii=False),
                    "captured_at": case["created_at"],
                    "reviewed_at": case["reviewed_at"],
                }
            )

        for role in ("comb", "shank"):
            relabel = sorted(
                (e for e in entries if e["role"] == role and e["split"] == SPLIT_TO_RELABEL),
                key=_relabel_priority,
            )
            notes = [
                {
                    "image": Path(str(e["image_name"])).name,
                    "label": Path(str(e["label_name"])).name if e["label_name"] else "",
                    "decision": e["decision"],
                    "issue": e["issue"],
                    "reviewer_note": role_note(e["case"], role),
                    "confidence": e["case"]["role_confidences"].get(role),
                    "case_id": e["case"]["case_id"],
                    "captured_at": e["case"]["created_at"],
                    "reviewed_at": e["case"]["reviewed_at"],
                }
                for e in relabel
            ]
            archive.writestr(f"{role}/{SPLIT_TO_RELABEL}/review_notes.csv", _csv_bytes(REVIEW_NOTES_FIELDS, notes))

        archive.writestr("manifest.csv", _csv_bytes(BUNDLE_MANIFEST_FIELDS, manifest_rows))
        archive.writestr("manifest.json", json.dumps(manifest_rows, ensure_ascii=False, indent=2))
        archive.writestr("README.txt", BUNDLE_README.encode("utf-8"))
        for name, payload in (extra_json or {}).items():
            archive.writestr(name, json.dumps(payload, ensure_ascii=False, indent=2))
    return {"images": len(manifest_rows), **_entry_counts(entries)}


def _entry_counts(entries) -> dict[str, int]:
    return {
        "train_ready": sum(1 for e in entries if e["split"] == SPLIT_TRAIN_READY),
        "to_relabel": sum(1 for e in entries if e["split"] == SPLIT_TO_RELABEL),
        "missing_label": sum(1 for e in entries if e["issue"] == "missing_label"),
        "missing_image": sum(1 for e in entries if e["issue"] == "missing_image"),
    }


def bundle_summary(states=EXPORT_STATES, cases=None) -> dict[str, int]:
    """Counts shown in the UI: how many images are train-ready vs need relabelling."""
    return _entry_counts(_bundle_entries(states, cases))


def seed_demo_cases() -> int:
    assets = DEMO_ASSET_DIR
    demo = (
        ("DEMO-REAL-01", "C-DEMO-001", {"comb": None, "shank": 0.8962}),
        ("DEMO-REAL-02", "C-DEMO-002", {"comb": 0.9693, "shank": 0.8775}),
        ("DEMO-REAL-03", "C-DEMO-003", {"comb": 0.9767, "shank": 0.8666}),
        ("DEMO-REAL-04", "C-DEMO-004", {"comb": 0.9347, "shank": 0.8294}),
        ("DEMO-REAL-05", "C-DEMO-005", {"comb": 0.9345, "shank": 0.8577}),
    )
    with closing(get_connection()) as connection, connection:
        connection.execute("DELETE FROM review_cases WHERE job_id LIKE 'DEMO-%'")

    created = 0
    for index, (job_id, chicken_id, confidences) in enumerate(demo):
        sources = {
            role: str(assets / f"{job_id}_{role}_source.jpg")
            for role in ("comb", "shank")
        }
        results = {
            role: str(assets / f"{job_id}_{role}_result.jpg")
            for role in ("comb", "shank")
        }
        missing = [path for path in (*sources.values(), *results.values()) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"缺少真實示範影像：{', '.join(missing)}")
        inserted = record_case(
            job_id=job_id,
            chicken_id=chicken_id,
            pipeline_status="已完成",
            confidence_threshold=0.25,
            role_confidences=confidences,
            measurements={"weight_g": 2500 + index * 120, "comb_area_cm2": 3.1, "shank_width_cm": 1.4, "shank_length_cm": 4.2},
            source_paths=sources,
            result_paths=results,
            models={"comb": "best_comb.pt", "shank": "best_shank.pt"},
            error="comb 未偵測到有效遮罩" if confidences["comb"] is None else "",
            force=True,
        )
        created += int(inserted)
    return created
