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
DEFAULT_REVIEW_CONFIDENCE = float(os.environ.get("CHICKEN_REVIEW_CONFIDENCE", "0.70"))
DEFAULT_QA_SAMPLE_RATE = float(os.environ.get("CHICKEN_REVIEW_SAMPLE_RATE", "0.05"))
DECISIONS = ("待審核", "正確", "不正確", "無法判斷")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_connection() -> sqlite3.Connection:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(REVIEW_DB_PATH)
    connection.row_factory = sqlite3.Row
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
    return connection


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
) -> bool:
    valid_confidences = [float(value) for value in role_confidences.values() if value is not None]
    minimum_confidence = min(valid_confidences) if valid_confidences else None
    failed = pipeline_status == "失敗" or bool(error)
    low_confidence = minimum_confidence is None or minimum_confidence < float(review_confidence)
    sampled = _qa_sample_selected(job_id, sample_rate)
    if not (force or failed or low_confidence or sampled):
        return False

    if failed:
        priority, reason = "高", "推論失敗"
    elif low_confidence:
        priority, reason = "高", f"最低信心 {minimum_confidence:.3f}" if minimum_confidence is not None else "缺少信心分數"
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
    return record_case(
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


def review_case(case_id: str, decision: str, note: str = "") -> None:
    if decision not in DECISIONS:
        raise ValueError(f"不支援的審核結果：{decision}")
    with closing(get_connection()) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE review_cases
            SET review_state = ?, reviewer_note = ?, reviewed_at = ?
            WHERE case_id = ?
            """,
            (decision, note.strip(), now_text(), case_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"找不到審核案例：{case_id}")


def review_summary() -> dict[str, int]:
    with closing(get_connection()) as connection, connection:
        rows = connection.execute(
            "SELECT review_state, COUNT(*) AS count FROM review_cases GROUP BY review_state"
        ).fetchall()
    counts = {state: 0 for state in DECISIONS}
    counts.update({row["review_state"]: int(row["count"]) for row in rows})
    counts["總案例"] = sum(counts.values())
    return counts


def export_review_bundle(states=("不正確", "無法判斷")) -> bytes:
    cases = [case for case in list_cases() if case["review_state"] in states]
    buffer = io.BytesIO()
    manifest_rows = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for case in cases:
            row = {
                "case_id": case["case_id"],
                "chicken_id": case["chicken_id"],
                "review_state": case["review_state"],
                "priority": case["priority"],
                "reason": case["reason"],
                "minimum_confidence": case["minimum_confidence"],
                "reviewer_note": case["reviewer_note"] or "",
                "created_at": case["created_at"],
            }
            for category in ("source_paths", "result_paths"):
                for role, path_text in case[category].items():
                    path = Path(path_text)
                    if path.is_file():
                        archive_name = f"cases/{case['case_id']}/{category.removesuffix('_paths')}_{role}{path.suffix.lower()}"
                        archive.write(path, archive_name)
                        row[f"{category}_{role}"] = archive_name
            manifest_rows.append(row)

        fields = sorted({key for row in manifest_rows for key in row})
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
        archive.writestr("manifest.csv", csv_buffer.getvalue().encode("utf-8-sig"))
        archive.writestr("manifest.json", json.dumps(manifest_rows, ensure_ascii=False, indent=2))
    return buffer.getvalue()


def seed_demo_cases() -> int:
    from PIL import Image, ImageDraw

    assets = REVIEW_DIR / "demo_assets"
    assets.mkdir(parents=True, exist_ok=True)
    demo = (
        ("DEMO-HIGH", "C-DEMO-001", {"comb": 0.93, "shank": 0.89}, "例行品質抽樣"),
        ("DEMO-LOW", "C-DEMO-002", {"comb": 0.46, "shank": 0.81}, "低信心"),
        ("DEMO-FAIL", "C-DEMO-003", {"comb": None, "shank": None}, "推論失敗"),
    )
    created = 0
    for index, (job_id, chicken_id, confidences, kind) in enumerate(demo):
        sources, results = {}, {}
        for role in ("comb", "shank"):
            source = assets / f"{job_id}_{role}_source.png"
            result = assets / f"{job_id}_{role}_result.png"
            if not source.exists():
                image = Image.new("RGB", (640, 360), (32, 38, 45))
                draw = ImageDraw.Draw(image)
                draw.rectangle((28, 28, 612, 332), outline=(94, 107, 120), width=3)
                if role == "comb":
                    draw.ellipse((210, 95, 430, 280), fill=(167, 71, 72))
                else:
                    draw.rounded_rectangle((280, 55, 360, 305), radius=30, fill=(171, 147, 116))
                draw.text((45, 45), f"{chicken_id} / {role.upper()} / SOURCE", fill=(235, 239, 242))
                image.save(source)
                overlay = image.copy()
                overlay_draw = ImageDraw.Draw(overlay)
                if role == "comb":
                    overlay_draw.ellipse((205, 90, 435, 285), outline=(35, 214, 118), width=8)
                else:
                    overlay_draw.rounded_rectangle((275, 50, 365, 310), radius=34, outline=(35, 214, 118), width=8)
                overlay_draw.text((45, 300), f"YOLO CONF: {confidences[role] if confidences[role] is not None else 'N/A'}", fill=(255, 205, 52))
                overlay.save(result)
            sources[role], results[role] = str(source), str(result)
        inserted = record_case(
            job_id=job_id,
            chicken_id=chicken_id,
            pipeline_status="失敗" if kind == "推論失敗" else "已完成",
            confidence_threshold=0.25,
            role_confidences=confidences,
            measurements={"weight_g": 2500 + index * 120, "comb_area_cm2": 3.1, "shank_width_cm": 1.4, "shank_length_cm": 4.2},
            source_paths=sources,
            result_paths=results,
            models={"comb": "best_comb.pt", "shank": "best_shank.pt"},
            error="comb 未偵測到有效遮罩" if kind == "推論失敗" else "",
            force=True,
        )
        created += int(inserted)
    return created
