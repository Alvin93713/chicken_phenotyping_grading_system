"""SQLite 資料庫操作模組。"""

from __future__ import annotations

import sqlite3
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else BASE_DIR
BUNDLED_DATA_DIR = BASE_DIR / "data"
DATA_DIR = APP_DIR / "data" if getattr(sys, "frozen", False) else BUNDLED_DATA_DIR
DB_PATH = DATA_DIR / "chicken_phenotype.db"
BUNDLED_DB_PATH = BUNDLED_DATA_DIR / "chicken_phenotype.db"

DISPLAY_COLUMNS = [
    "chicken_id",
    "weight_g",
    "comb_area_cm2",
    "shank_width_cm",
    "shank_length_cm",
    "grade_result",
    "light_result",
    "selected_features",
    "created_at",
    "updated_at",
    "graded_at",
    "final_scanned_at",
]


def now_text():
    """回傳資料庫使用的時間字串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_connection():
    """建立 SQLite 連線。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(sys, "frozen", False) and not DB_PATH.exists() and BUNDLED_DB_PATH.exists():
        shutil.copy2(BUNDLED_DB_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table_name):
    return {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def normalize_optional_measurement(value, name):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    if number == 0:
        return None
    return number


def normalize_chicken_values(weight_g, comb_area_cm2, shank_width_cm, shank_length_cm=None):
    return (
        normalize_optional_measurement(weight_g, "weight_g"),
        normalize_optional_measurement(comb_area_cm2, "comb_area_cm2"),
        normalize_optional_measurement(shank_width_cm, "shank_width_cm"),
        normalize_optional_measurement(shank_length_cm, "shank_length_cm"),
    )


def migrate_nullable_measurements(conn):
    table_info = conn.execute("PRAGMA table_info(chickens)").fetchall()
    columns = {row["name"]: row for row in table_info}
    needs_rebuild = (
        "comb_area_cm2" in columns
        and "shank_width_cm" in columns
        and (columns["weight_g"]["notnull"] or columns["comb_area_cm2"]["notnull"] or columns["shank_width_cm"]["notnull"])
    )
    if needs_rebuild:
        conn.execute("ALTER TABLE chickens RENAME TO chickens_old")
        conn.execute(
            """
            CREATE TABLE chickens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chicken_id TEXT UNIQUE NOT NULL,
                weight_g REAL,
                comb_area_cm2 REAL,
                shank_width_cm REAL,
                shank_length_cm REAL,
                selected_features TEXT,
                weight_threshold REAL,
                comb_area_threshold REAL,
                shank_width_threshold REAL,
                shank_length_threshold REAL,
                weight_standby_threshold REAL,
                comb_area_standby_threshold REAL,
                shank_width_standby_threshold REAL,
                shank_length_standby_threshold REAL,
                grade_result TEXT,
                light_result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                graded_at TEXT,
                final_scanned_at TEXT
            );
            """
        )
        old_columns = table_columns(conn, "chickens_old")
        shank_source = "shank_width_cm" if "shank_width_cm" in old_columns else "shank_" + "area_cm2"
        shank_threshold_source = (
            "shank_width_threshold"
            if "shank_width_threshold" in old_columns
            else "shank_" + "area_threshold"
        )
        shank_threshold_expr = shank_threshold_source if shank_threshold_source in old_columns else "NULL"
        conn.execute(
            f"""
            INSERT INTO chickens (
                id, chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm,
                selected_features, weight_threshold, comb_area_threshold,
                shank_width_threshold, shank_length_threshold,
                weight_standby_threshold, comb_area_standby_threshold,
                shank_width_standby_threshold, shank_length_standby_threshold,
                grade_result, light_result,
                created_at, updated_at, graded_at, final_scanned_at
            )
            SELECT
                id,
                chicken_id,
                CASE
                    WHEN weight_g IS NULL OR weight_g = '' OR CAST(weight_g AS REAL) = 0 THEN NULL
                    ELSE CAST(weight_g AS REAL)
                END,
                CASE
                    WHEN comb_area_cm2 IS NULL OR comb_area_cm2 = '' OR CAST(comb_area_cm2 AS REAL) = 0 THEN NULL
                    ELSE CAST(comb_area_cm2 AS REAL)
                END,
                CASE
                    WHEN {shank_source} IS NULL OR {shank_source} = '' OR CAST({shank_source} AS REAL) = 0 THEN NULL
                    ELSE CAST({shank_source} AS REAL)
                END,
                NULL,
                selected_features,
                weight_threshold,
                comb_area_threshold,
                {shank_threshold_expr},
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                grade_result,
                light_result,
                created_at,
                updated_at,
                graded_at,
                NULL
            FROM chickens_old
            """
        )
        conn.execute("DROP TABLE chickens_old")
    conn.execute("UPDATE chickens SET comb_area_cm2 = NULL WHERE comb_area_cm2 = '' OR CAST(comb_area_cm2 AS REAL) = 0")
    conn.execute("UPDATE chickens SET shank_width_cm = NULL WHERE shank_width_cm = '' OR CAST(shank_width_cm AS REAL) = 0")
    if "shank_length_cm" in columns:
        conn.execute("UPDATE chickens SET shank_length_cm = NULL WHERE shank_length_cm = '' OR CAST(shank_length_cm AS REAL) = 0")
    conn.execute("UPDATE chickens SET weight_g = NULL WHERE weight_g = '' OR CAST(weight_g AS REAL) = 0")


def init_db():
    """建立 chickens 資料表。"""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chickens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chicken_id TEXT UNIQUE NOT NULL,
                weight_g REAL,
                comb_area_cm2 REAL,
                shank_width_cm REAL,
                shank_length_cm REAL,
                selected_features TEXT,
                weight_threshold REAL,
                comb_area_threshold REAL,
                shank_width_threshold REAL,
                shank_length_threshold REAL,
                weight_standby_threshold REAL,
                comb_area_standby_threshold REAL,
                shank_width_standby_threshold REAL,
                shank_length_standby_threshold REAL,
                grade_result TEXT,
                light_result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                graded_at TEXT,
                final_scanned_at TEXT
            );
            """
        )
        columns = table_columns(conn, "chickens")
        if "shank_width_cm" not in columns:
            conn.execute("ALTER TABLE chickens ADD COLUMN shank_width_cm REAL")
            legacy_width_source = "shank_" + "area_cm2"
            if legacy_width_source in columns:
                conn.execute(f"UPDATE chickens SET shank_width_cm = {legacy_width_source}")
        if "shank_length_cm" not in columns:
            conn.execute("ALTER TABLE chickens ADD COLUMN shank_length_cm REAL")
        if "shank_width_threshold" not in columns:
            conn.execute("ALTER TABLE chickens ADD COLUMN shank_width_threshold REAL")
            legacy_threshold_source = "shank_" + "area_threshold"
            if legacy_threshold_source in columns:
                conn.execute(f"UPDATE chickens SET shank_width_threshold = {legacy_threshold_source}")
        if "shank_length_threshold" not in columns:
            conn.execute("ALTER TABLE chickens ADD COLUMN shank_length_threshold REAL")
        for column_name in (
            "weight_standby_threshold",
            "comb_area_standby_threshold",
            "shank_width_standby_threshold",
            "shank_length_standby_threshold",
        ):
            if column_name not in columns:
                conn.execute(f"ALTER TABLE chickens ADD COLUMN {column_name} REAL")
        if "final_scanned_at" not in columns:
            conn.execute("ALTER TABLE chickens ADD COLUMN final_scanned_at TEXT")
        migrate_nullable_measurements(conn)


def validate_chicken_values(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm=None):
    """Validate chicken phenotype values before database writes."""
    if not str(chicken_id).strip():
        raise ValueError("chicken_id is required")
    normalize_chicken_values(weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)

def add_chicken(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm=None):
    """新增雞隻資料。"""
    validate_chicken_values(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)
    weight_g, comb_area_cm2, shank_width_cm, shank_length_cm = normalize_chicken_values(
        weight_g,
        comb_area_cm2,
        shank_width_cm,
        shank_length_cm,
    )
    timestamp = now_text()

    try:
        with get_connection() as conn:
            columns = table_columns(conn, "chickens")
            insert_columns = [
                "chicken_id",
                "weight_g",
                "comb_area_cm2",
                "shank_width_cm",
                "shank_length_cm",
                "created_at",
                "updated_at",
            ]
            insert_values = [
                str(chicken_id).strip(),
                weight_g,
                comb_area_cm2,
                shank_width_cm,
                shank_length_cm,
                timestamp,
                timestamp,
            ]
            legacy_width_column = "shank_" + "area_cm2"
            if legacy_width_column in columns:
                insert_columns.insert(3, legacy_width_column)
                insert_values.insert(3, shank_width_cm)
            placeholders = ", ".join("?" for _ in insert_columns)
            conn.execute(
                f"""
                INSERT INTO chickens ({", ".join(insert_columns)})
                VALUES ({placeholders})
                """,
                tuple(insert_values),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"chicken_id 已存在：{chicken_id}") from exc


def update_chicken(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm=None):
    """更新雞隻表型資料，並保留既有分級結果。"""
    validate_chicken_values(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)
    weight_g, comb_area_cm2, shank_width_cm, shank_length_cm = normalize_chicken_values(
        weight_g,
        comb_area_cm2,
        shank_width_cm,
        shank_length_cm,
    )

    with get_connection() as conn:
        columns = table_columns(conn, "chickens")
        assignments = [
            "weight_g = ?",
            "comb_area_cm2 = ?",
            "shank_width_cm = ?",
            "shank_length_cm = ?",
            "updated_at = ?",
        ]
        values = [
            weight_g,
            comb_area_cm2,
            shank_width_cm,
            shank_length_cm,
            now_text(),
        ]
        legacy_width_column = "shank_" + "area_cm2"
        if legacy_width_column in columns:
            assignments.insert(2, f"{legacy_width_column} = ?")
            values.insert(2, shank_width_cm)
        values.append(str(chicken_id).strip())
        cursor = conn.execute(
            f"""
            UPDATE chickens
            SET {", ".join(assignments)}
            WHERE chicken_id = ?
            """,
            tuple(values),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"查無此雞隻：{chicken_id}")


def delete_chicken(chicken_id):
    """刪除雞隻資料。"""
    with get_connection() as conn:
        conn.execute("DELETE FROM chickens WHERE chicken_id = ?", (chicken_id,))


def get_chicken(chicken_id):
    """依 chicken_id 查詢單筆資料。"""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM chickens WHERE chicken_id = ?",
            (str(chicken_id).strip(),),
        ).fetchone()
    return dict(row) if row else None


def get_all_chickens():
    """查詢所有雞隻資料。"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM chickens
            ORDER BY chicken_id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_chickens_df():
    """以 DataFrame 回傳所有雞隻資料。"""
    rows = get_all_chickens()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    return df[DISPLAY_COLUMNS]


def update_grading_result(
    chicken_id,
    selected_features,
    pass_thresholds,
    standby_thresholds,
    grade_result,
    light_result,
):
    """將單隻雞的分級結果寫回資料庫。"""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE chickens
            SET selected_features = ?,
                weight_threshold = ?,
                comb_area_threshold = ?,
                shank_width_threshold = ?,
                shank_length_threshold = ?,
                weight_standby_threshold = ?,
                comb_area_standby_threshold = ?,
                shank_width_standby_threshold = ?,
                shank_length_standby_threshold = ?,
                grade_result = ?,
                light_result = ?,
                updated_at = ?,
                graded_at = ?
            WHERE chicken_id = ?
            """,
            (
                ",".join(selected_features),
                pass_thresholds.get("weight_g"),
                pass_thresholds.get("comb_area_cm2"),
                pass_thresholds.get("shank_width_cm"),
                pass_thresholds.get("shank_length_cm"),
                standby_thresholds.get("weight_g"),
                standby_thresholds.get("comb_area_cm2"),
                standby_thresholds.get("shank_width_cm"),
                standby_thresholds.get("shank_length_cm"),
                grade_result,
                light_result,
                now_text(),
                now_text(),
                chicken_id,
            ),
        )


def mark_final_scanned(chicken_id):
    """記錄該雞隻已完成第二階段掃描與亮燈判定。"""
    timestamp = now_text()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE chickens
            SET final_scanned_at = ?,
                updated_at = ?
            WHERE chicken_id = ?
            """,
            (timestamp, timestamp, str(chicken_id).strip()),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"查無此雞隻：{chicken_id}")
    return timestamp


def get_latest_grading_config():
    """Return the latest saved grading thresholds from the database."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                selected_features,
                weight_threshold,
                comb_area_threshold,
                shank_width_threshold,
                shank_length_threshold,
                weight_standby_threshold,
                comb_area_standby_threshold,
                shank_width_standby_threshold,
                shank_length_standby_threshold
            FROM chickens
            WHERE graded_at IS NOT NULL
            ORDER BY graded_at DESC, updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        return None

    selected_features = [
        feature.strip()
        for feature in str(row["selected_features"] or "").split(",")
        if feature.strip()
    ]
    pass_thresholds = {
        "weight_g": row["weight_threshold"],
        "comb_area_cm2": row["comb_area_threshold"],
        "shank_width_cm": row["shank_width_threshold"],
        "shank_length_cm": row["shank_length_threshold"],
    }
    standby_thresholds = {
        "weight_g": row["weight_standby_threshold"],
        "comb_area_cm2": row["comb_area_standby_threshold"],
        "shank_width_cm": row["shank_width_standby_threshold"],
        "shank_length_cm": row["shank_length_standby_threshold"],
    }
    return selected_features, pass_thresholds, standby_thresholds


def import_csv(file):
    """匯入 CSV，已存在 chicken_id 則更新，不存在則新增。"""
    df = pd.read_csv(file)
    required = {"chicken_id", "weight_g", "comb_area_cm2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要欄位：{', '.join(sorted(missing))}")
    if "shank_width_cm" not in df.columns:
        raise ValueError("CSV 缺少必要欄位：shank_width_cm")
    if "shank_length_cm" not in df.columns:
        df["shank_length_cm"] = None

    inserted = 0
    updated = 0
    for _, row in df.iterrows():
        chicken_id = str(row["chicken_id"]).strip()
        validate_chicken_values(
            chicken_id,
            row["weight_g"],
            row["comb_area_cm2"],
            row["shank_width_cm"],
            row["shank_length_cm"],
        )
        if get_chicken(chicken_id):
            update_chicken(
                chicken_id,
                row["weight_g"],
                row["comb_area_cm2"],
                row["shank_width_cm"],
                row["shank_length_cm"],
            )
            updated += 1
        else:
            add_chicken(
                chicken_id,
                row["weight_g"],
                row["comb_area_cm2"],
                row["shank_width_cm"],
                row["shank_length_cm"],
            )
            inserted += 1
    return inserted, updated


def export_csv_bytes():
    """匯出 CSV 位元組資料，供 Streamlit 下載。"""
    return get_all_chickens_df().to_csv(index=False).encode("utf-8-sig")

