"""SQLite 資料庫操作模組。"""

from __future__ import annotations

import sqlite3
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from i18n import t
except Exception:  # pragma: no cover - i18n/streamlit unavailable
    def t(text, **values):
        return text.format(**values) if values else text


BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else BASE_DIR
BUNDLED_DATA_DIR = BASE_DIR / "data"
DATA_DIR = APP_DIR / "data" if getattr(sys, "frozen", False) else BUNDLED_DATA_DIR
DB_PATH = DATA_DIR / "chicken_phenotype.db"
BUNDLED_DB_PATH = BUNDLED_DATA_DIR / "chicken_phenotype.db"

DEFAULT_BREED = "紅羽土雞"  # existing rows (from before breeds existed) belong to this breed

DISPLAY_COLUMNS = [
    "chicken_id",
    "breed",
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


@contextmanager
def get_connection():
    """SQLite 連線：區塊成功結束時 commit、發生例外時 rollback，最後一定關閉連線。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(sys, "frozen", False) and not DB_PATH.exists() and BUNDLED_DB_PATH.exists():
        shutil.copy2(BUNDLED_DB_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


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


_INITIALIZED_DB_PATHS: set[str] = set()


def init_db():
    """建立 chickens 資料表（每個資料庫路徑只在本程序第一次呼叫時執行 schema 與遷移）。"""
    key = str(DB_PATH)
    if key in _INITIALIZED_DB_PATHS and DB_PATH.exists():
        return
    _init_db_schema()
    _INITIALIZED_DB_PATHS.add(key)


def _init_db_schema():
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
        _init_breed_schema(conn)


THRESHOLD_COLUMNS = (
    "weight_threshold", "comb_area_threshold", "shank_width_threshold", "shank_length_threshold",
    "weight_standby_threshold", "comb_area_standby_threshold", "shank_width_standby_threshold",
    "shank_length_standby_threshold",
)


def _init_breed_schema(conn):
    """Breeds, the threshold history of every breed and the current-breed setting. Additive only: existing
    chickens get DEFAULT_BREED, and the thresholds last used before breeds existed become its first record."""
    if "breed" not in table_columns(conn, "chickens"):
        conn.execute("ALTER TABLE chickens ADD COLUMN breed TEXT")
    conn.execute("UPDATE chickens SET breed = ? WHERE breed IS NULL OR breed = ''", (DEFAULT_BREED,))
    conn.execute("CREATE TABLE IF NOT EXISTS breeds (name TEXT PRIMARY KEY, created_at TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS breed_thresholds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            breed TEXT NOT NULL,
            selected_features TEXT,
            {", ".join(f"{column} REAL" for column in THRESHOLD_COLUMNS)},
            graded_count INTEGER,
            saved_at TEXT NOT NULL
        )
        """
    )
    if conn.execute("SELECT COUNT(*) FROM breeds").fetchone()[0] == 0:  # first run (a deleted default stays deleted)
        conn.execute("INSERT INTO breeds (name, created_at) VALUES (?, ?)", (DEFAULT_BREED, now_text()))
    conn.execute("INSERT OR IGNORE INTO breeds (name, created_at) SELECT DISTINCT breed, ? FROM chickens", (now_text(),))
    if conn.execute("SELECT COUNT(*) FROM breed_thresholds").fetchone()[0] == 0:
        legacy = conn.execute(
            f"""SELECT selected_features, {", ".join(THRESHOLD_COLUMNS)}, graded_at FROM chickens
                WHERE graded_at IS NOT NULL ORDER BY graded_at DESC, updated_at DESC LIMIT 1"""
        ).fetchone()
        if legacy:
            conn.execute(
                f"""INSERT INTO breed_thresholds (breed, selected_features, {", ".join(THRESHOLD_COLUMNS)}, saved_at)
                    VALUES (?, ?, {", ".join("?" for _ in THRESHOLD_COLUMNS)}, ?)""",
                (DEFAULT_BREED, legacy["selected_features"], *[legacy[c] for c in THRESHOLD_COLUMNS], legacy["graded_at"]),
            )


# ---------------------------------------------------------------- breeds
def list_breeds():
    init_db()
    with get_connection() as conn:
        return [row["name"] for row in conn.execute("SELECT name FROM breeds ORDER BY rowid")]


def add_breed(name):
    name = str(name or "").strip()
    if not name:
        raise ValueError(t("請輸入品系名稱"))
    init_db()
    with get_connection() as conn:
        try:
            conn.execute("INSERT INTO breeds (name, created_at) VALUES (?, ?)", (name, now_text()))
        except sqlite3.IntegrityError as exc:
            raise ValueError(t("品系已存在：{breed}", breed=name)) from exc
    return name


def get_current_breed():
    """The breed new chickens are recorded under and graded with (shared by every page and the worker)."""
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key = 'current_breed'").fetchone()
        if row and conn.execute("SELECT 1 FROM breeds WHERE name = ?", (row["value"],)).fetchone():
            return row["value"]
        first = conn.execute("SELECT name FROM breeds ORDER BY rowid LIMIT 1").fetchone()
    return first["name"] if first else DEFAULT_BREED


def breed_mismatch_message(chicken_id, breed=None):
    """A message when the chicken is recorded under another breed than `breed` (default: the current breed),
    else "" (also for a new chicken ID). Pages that work on the current breed refuse such chickens."""
    breed = breed or get_current_breed()
    chicken = get_chicken(chicken_id)
    if chicken and chicken.get("breed") and chicken["breed"] != breed:
        return t("chicken ID {chicken_id} 屬於品系「{other}」，不是目前品系「{breed}」；請先切換目前品系",
                 chicken_id=chicken["chicken_id"], other=chicken["breed"], breed=breed)
    return ""


def breed_chicken_count(name):
    init_db()
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM chickens WHERE breed = ?", (name,)).fetchone()[0]


def delete_breed(name):
    """Delete a breed together with all of its chickens (the UI asks for confirmation first); returns the
    number of chickens deleted. Its threshold history is kept (re-adding the same name brings it back).
    Refuses the last breed."""
    breeds = list_breeds()
    if name not in breeds:
        raise ValueError(t("查無此品系：{breed}", breed=name))
    if len(breeds) <= 1:
        raise ValueError(t("至少需保留一個品系"))
    was_current = get_current_breed() == name
    with get_connection() as conn:
        deleted = conn.execute("DELETE FROM chickens WHERE breed = ?", (name,)).rowcount
        conn.execute("DELETE FROM breeds WHERE name = ?", (name,))
        if was_current:
            conn.execute("DELETE FROM app_settings WHERE key = 'current_breed'")
    return deleted


def set_current_breed(name):
    if name not in list_breeds():
        raise ValueError(t("查無此品系：{breed}", breed=name))
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES ('current_breed', ?)", (name,))


def save_breed_thresholds(breed, selected_features, pass_thresholds, standby_thresholds, graded_count=None):
    """Append one record to the breed's threshold history (never overwritten)."""
    init_db()
    values = (
        pass_thresholds.get("weight_g"), pass_thresholds.get("comb_area_cm2"),
        pass_thresholds.get("shank_width_cm"), pass_thresholds.get("shank_length_cm"),
        standby_thresholds.get("weight_g"), standby_thresholds.get("comb_area_cm2"),
        standby_thresholds.get("shank_width_cm"), standby_thresholds.get("shank_length_cm"),
    )
    with get_connection() as conn:
        conn.execute(
            f"""INSERT INTO breed_thresholds (breed, selected_features, {", ".join(THRESHOLD_COLUMNS)}, graded_count, saved_at)
                VALUES (?, ?, {", ".join("?" for _ in THRESHOLD_COLUMNS)}, ?, ?)""",
            (breed, ",".join(selected_features), *values, graded_count, now_text()),
        )


def breed_threshold_history(breed, limit=50):
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT saved_at, selected_features, {", ".join(THRESHOLD_COLUMNS)}, graded_count
                FROM breed_thresholds WHERE breed = ? ORDER BY id DESC LIMIT ?""",
            (breed, limit),
        ).fetchall()
    return [dict(row) for row in rows]


MEASURE_COLUMNS = ("weight_g", "comb_area_cm2", "shank_width_cm", "shank_length_cm")


def breed_stats(breed=None):
    """Measured statistics of one breed (None = every breed): count, graded counts and, per feature,
    the number of measured values, mean, standard deviation, min and max."""
    rows = get_all_chickens(breed)
    stats = {"count": len(rows), "grades": {}, "features": {}}
    for row in rows:
        if row.get("grade_result"):
            stats["grades"][row["grade_result"]] = stats["grades"].get(row["grade_result"], 0) + 1
    for column in MEASURE_COLUMNS:
        values = [float(row[column]) for row in rows if row.get(column) is not None]
        if values:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
            stats["features"][column] = {"n": len(values), "mean": mean, "std": std,
                                         "min": min(values), "max": max(values)}
        else:
            stats["features"][column] = {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return stats


LEGACY_WIDTH_COLUMN = "shank_" + "area_cm2"


def validate_chicken_values(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm=None):
    """Validate and normalize phenotype values once; returns (chicken_id, weight, comb, width, length)."""
    chicken_id = str(chicken_id).strip()
    if not chicken_id:
        raise ValueError("chicken_id is required")
    return (chicken_id, *normalize_chicken_values(weight_g, comb_area_cm2, shank_width_cm, shank_length_cm))


def _insert_chicken(conn, has_legacy_width, values, breed):
    chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm = values
    timestamp = now_text()
    columns = ["chicken_id", "weight_g", "comb_area_cm2", "shank_width_cm", "shank_length_cm", "created_at", "updated_at",
               "breed"]
    row = [chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm, timestamp, timestamp, breed]
    if has_legacy_width:
        columns.insert(3, LEGACY_WIDTH_COLUMN)
        row.insert(3, shank_width_cm)
    try:
        conn.execute(
            f"INSERT INTO chickens ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            tuple(row),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(t("chicken_id 已存在：{chicken_id}", chicken_id=chicken_id)) from exc


def _update_chicken(conn, has_legacy_width, values, breed=None):
    """breed None keeps the chicken's breed."""
    chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm = values
    assignments = ["weight_g = ?", "comb_area_cm2 = ?", "shank_width_cm = ?", "shank_length_cm = ?", "updated_at = ?"]
    row = [weight_g, comb_area_cm2, shank_width_cm, shank_length_cm, now_text()]
    if breed:
        assignments.append("breed = ?")
        row.append(breed)
    if has_legacy_width:
        assignments.insert(2, f"{LEGACY_WIDTH_COLUMN} = ?")
        row.insert(2, shank_width_cm)
    cursor = conn.execute(f"UPDATE chickens SET {', '.join(assignments)} WHERE chicken_id = ?", (*row, chicken_id))
    if cursor.rowcount == 0:
        raise ValueError(t("查無此雞隻：{chicken_id}", chicken_id=chicken_id))


def add_chicken(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm=None, breed=None):
    """新增雞隻資料（未指定品系時記為目前品系）。"""
    values = validate_chicken_values(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)
    breed = str(breed or "").strip() or get_current_breed()
    with get_connection() as conn:
        _insert_chicken(conn, LEGACY_WIDTH_COLUMN in table_columns(conn, "chickens"), values, breed)


def update_chicken(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm=None, breed=None):
    """更新雞隻表型資料，並保留既有分級結果（breed 為 None 時保留原品系）。"""
    values = validate_chicken_values(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)
    with get_connection() as conn:
        _update_chicken(conn, LEGACY_WIDTH_COLUMN in table_columns(conn, "chickens"), values,
                        str(breed or "").strip() or None)


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


def get_all_chickens(breed=None):
    """查詢所有雞隻資料（指定 breed 時只回傳該品系）。"""
    init_db()
    with get_connection() as conn:
        if breed:
            rows = conn.execute("SELECT * FROM chickens WHERE breed = ? ORDER BY chicken_id ASC", (breed,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chickens ORDER BY chicken_id ASC").fetchall()
    return [dict(row) for row in rows]


def get_all_chickens_df(rows=None):
    """以 DataFrame 回傳所有雞隻資料（可傳入已讀取的 rows，避免再查一次）。"""
    rows = get_all_chickens() if rows is None else rows
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
    update_grading_results([(chicken_id, grade_result, light_result)],
                           selected_features, pass_thresholds, standby_thresholds)


def update_grading_results(results, selected_features, pass_thresholds, standby_thresholds):
    """Write many (chicken_id, grade_result, light_result) rows with the same thresholds in one transaction."""
    timestamp = now_text()
    thresholds = (
        ",".join(selected_features),
        pass_thresholds.get("weight_g"),
        pass_thresholds.get("comb_area_cm2"),
        pass_thresholds.get("shank_width_cm"),
        pass_thresholds.get("shank_length_cm"),
        standby_thresholds.get("weight_g"),
        standby_thresholds.get("comb_area_cm2"),
        standby_thresholds.get("shank_width_cm"),
        standby_thresholds.get("shank_length_cm"),
    )
    with get_connection() as conn:
        conn.executemany(
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
            [(*thresholds, grade, light, timestamp, timestamp, chicken_id) for chicken_id, grade, light in results],
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
            raise ValueError(t("查無此雞隻：{chicken_id}", chicken_id=chicken_id))
    return timestamp


def get_latest_grading_config(breed=None):
    """Return the breed's latest saved grading thresholds (breed None = the current breed), or None."""
    init_db()
    breed = breed or get_current_breed()
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT selected_features, {', '.join(THRESHOLD_COLUMNS)} FROM breed_thresholds "
            "WHERE breed = ? ORDER BY id DESC LIMIT 1",
            (breed,),
        ).fetchone()
        if not row:  # no history record yet: the thresholds stored on the breed's last graded chicken
            row = conn.execute(
                f"""SELECT selected_features, {', '.join(THRESHOLD_COLUMNS)} FROM chickens
                    WHERE graded_at IS NOT NULL AND breed = ? ORDER BY graded_at DESC, updated_at DESC LIMIT 1""",
                (breed,),
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
        raise ValueError(t("CSV 缺少必要欄位：{columns}", columns=", ".join(sorted(missing))))
    if "shank_width_cm" not in df.columns:
        raise ValueError(t("CSV 缺少必要欄位：{columns}", columns="shank_width_cm"))
    if "shank_length_cm" not in df.columns:
        df["shank_length_cm"] = None

    # Validate every row first (nothing is written if one row is bad), then write all rows in one transaction.
    rows = [
        validate_chicken_values(row.chicken_id, row.weight_g, row.comb_area_cm2, row.shank_width_cm, row.shank_length_cm)
        for row in df[["chicken_id", "weight_g", "comb_area_cm2", "shank_width_cm", "shank_length_cm"]].itertuples(index=False)
    ]
    # Optional breed column: a filled cell sets the breed (new breeds are added); blank = current breed for new
    # chickens, unchanged for existing ones.
    breeds = [str(value).strip() if "breed" in df.columns and not pd.isna(value) else ""
              for value in (df["breed"] if "breed" in df.columns else [None] * len(df))]
    current_breed = get_current_breed()
    inserted = 0
    updated = 0
    with get_connection() as conn:
        has_legacy_width = LEGACY_WIDTH_COLUMN in table_columns(conn, "chickens")
        existing = {row["chicken_id"] for row in conn.execute("SELECT chicken_id FROM chickens")}
        for breed in {b for b in breeds if b}:
            conn.execute("INSERT OR IGNORE INTO breeds (name, created_at) VALUES (?, ?)", (breed, now_text()))
        for values, breed in zip(rows, breeds):
            if values[0] in existing:
                _update_chicken(conn, has_legacy_width, values, breed or None)
                updated += 1
            else:
                _insert_chicken(conn, has_legacy_width, values, breed or current_breed)
                existing.add(values[0])
                inserted += 1
    return inserted, updated


def export_csv_bytes():
    """匯出 CSV 位元組資料，供 Streamlit 下載。"""
    return get_all_chickens_df().to_csv(index=False).encode("utf-8-sig")

