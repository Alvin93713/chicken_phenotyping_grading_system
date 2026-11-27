"""Two-camera configuration, calibration, and snapshot workflow."""
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from dual_camera import (
    MODEL_PATHS, ROLES, capture_one, capture_pair, ensure_preview_server, save_one, save_pair,
    validate_calibration,
)
from inference_queue import enqueue_snapshot_job, list_jobs, STATUS_COMPLETED, STATUS_FAILED
from inference_utils import DATA_DIR, OUTPUTS_DIR
from i18n import t, t_records
import ui_style

LABELS = {"comb": "Comb 雞冠", "shank": "Shank 腳脛"}
JOB_TABLE_HIDDEN = ("Arduino", "影片", "建立時間", "初步燈號")  # the light says the same as 初步分級
# Translated for display only (the job dicts keep their Chinese values).
JOB_TABLE_VALUE_COLUMNS = ("狀態", "訊息", "初步分級")


def _label(role):
    return t(LABELS[role])


def _job_text(text):
    """Translate a job value written in Chinese by the background worker (display only)."""
    text = "" if text is None else str(text)
    prefix, separator, role = text.removesuffix(" 單張照片").partition("正在推論 ")
    if text.endswith(" 單張照片") and separator and not prefix:
        return t("正在推論 {role} 單張照片", role=role)
    return t(text)


def _job_table(jobs):
    rows = []
    for job in jobs:
        row = {k: v for k, v in job.items() if k not in JOB_TABLE_HIDDEN}
        for column in JOB_TABLE_VALUE_COLUMNS:
            if row.get(column):
                row[column] = _job_text(row[column])
        rows.append(row)
    return t_records(rows)


def begin_final_weight_collection():
    st.session_state.scale_collection_phase = "finalizing"
    # One-shot: the next run measures. It hides the form at once and shows the ring where the result goes,
    # so nothing jumps and the ring appears without waiting for the rest of the page.
    st.session_state["_measure_requested"] = True


DEFAULT_CONFIDENCE = 0.5     # YOLO confidence threshold the phenotype page starts with
MIN_WEIGHT_G = 300.0          # below this the scale is almost certainly empty
WEIGHT_WINDOW_SECONDS = 2.0   # same window as the live scale panel
STABLE_MIN_SAMPLES = 3
STABLE_MAX_SPREAD_G = 20.0    # max − min allowed in the window: 20 g or 2 % of the weight, whichever is larger
STABLE_MAX_SPREAD_RATIO = 0.02


def resolve_weight(scale_port, use_manual, manual_weight, recent_samples, now=None):
    """Return (weight_g, sample_count). Uses the live panel's last 2 s of readings (no second read) when they
    are stable; otherwise raises so the operator retries. Manual weight only when explicitly chosen (or no
    scale is configured). Any weight below MIN_WEIGHT_G is rejected."""
    if use_manual or not scale_port:
        if manual_weight in (None, ""):
            raise ValueError(t("請輸入手動重量 (g)"))  # the field starts blank for every chicken
        weight, count = float(manual_weight), 0
    else:
        now = time.monotonic() if now is None else now
        # +0.6 s: the live panel refreshes every 0.5 s, so the newest batch can be up to that old.
        values = [value for stamp, value in recent_samples or [] if stamp >= now - WEIGHT_WINDOW_SECONDS - 0.6]
        if not values:
            raise ValueError(t("秤重設備未取得有效讀值，請確認後再按一次，或勾選「使用手動重量」"))
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        if len(values) < STABLE_MIN_SAMPLES or spread > max(STABLE_MAX_SPREAD_G, mean * STABLE_MAX_SPREAD_RATIO):
            raise ValueError(t("重量尚未穩定，請等雞隻靜止後再按一次"))
        weight, count = mean, len(values)
    if weight < MIN_WEIGHT_G:
        raise ValueError(t("重量 {weight} g 低於 {minimum} g，請確認雞隻已在秤上",
                           weight=f"{weight:.2f}", minimum=f"{MIN_WEIGHT_G:.0f}"))
    return weight, count


def acquire_measurement_inputs(indices, calibrations, supplied_frames=None, supplied_scales=None):
    """Capture (or take the uploaded) comb/shank frames; the weight is resolved beforehand."""
    if supplied_frames is None:
        frames = capture_pair(indices)
        scales = {
            role: validate_calibration(calibrations.get(role), indices[role], frames[role].shape)
            for role in ROLES
        }
    else:
        if set(supplied_frames) != set(ROLES) or set(supplied_scales or {}) != set(ROLES):
            raise ValueError(t("請上傳雞冠與腳脛兩張照片，並設定兩組 cm/pixel"))
        frames = supplied_frames
        scales = {role: float(supplied_scales[role]) for role in ROLES}
        if any(value <= 0 for value in scales.values()):
            raise ValueError(t("照片模式的 cm/pixel 必須大於 0"))
    return save_pair(frames), scales


def decode_uploaded_frame(uploaded_file):
    import cv2
    import numpy as np

    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(t("無法讀取上傳的照片，請使用 JPG、PNG 或 BMP"))
    return frame


def _camera_settings_path() -> Path:
    return Path(os.environ.get("CHICKEN_CAMERA_SETTINGS_PATH") or (DATA_DIR / "camera_settings.json"))


def _load_camera_settings() -> dict:
    try:
        data = json.loads(_camera_settings_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_camera_settings(**changes) -> None:
    """Merge `cameras` / `calibrations` into data/camera_settings.json (atomic write)."""
    data = _load_camera_settings()
    data.update(changes)
    path = _camera_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


# --- calibration history: one JSON line per confirmed calibration, never overwritten ---------------------
CALIBRATION_CHANGE_WARN = 0.05  # warn when cm/pixel moves more than 5 % from the previous calibration


def _calibration_history_path() -> Path:
    return _camera_settings_path().with_name("calibration_history.jsonl")


def load_calibration_history() -> list:
    """All recorded calibrations, oldest first (a damaged line is skipped, the rest is kept)."""
    try:
        lines = _calibration_history_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def record_calibration(calibrations, points, lengths, photos, now=None) -> dict:
    """Append this calibration to data/calibration_history.jsonl. Per role: camera, the two clicked points,
    their pixel distance, the ruler length, cm/pixel and its change from the previous calibration of that
    role (None the first time). Returns the entry."""
    previous = {}
    for old in load_calibration_history():
        for role, value in (old.get("roles") or {}).items():
            previous[role] = value
    roles = {}
    for role, value in calibrations.items():
        (x1, y1), (x2, y2) = points[role]
        before = (previous.get(role) or {}).get("cm_per_pixel")
        cm_per_pixel = float(value["cm_per_pixel"])
        roles[role] = {
            "camera": value["camera"],
            "image_size": list(value["shape"]),
            "points": [[x1, y1], [x2, y2]],
            "pixel_distance": round(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 3),
            "ruler_cm": float(lengths[role]),
            "cm_per_pixel": cm_per_pixel,
            "previous_cm_per_pixel": before,
            "change_ratio": None if not before else round(cm_per_pixel / float(before) - 1, 6),
            "photo": str(photos[role]),
        }
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "roles": roles}
    path = _calibration_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def calibration_history_table(limit=20) -> None:
    """Newest first: one row per camera per calibration."""
    rows = []
    for entry in reversed(load_calibration_history()[-limit:]):
        for role in ROLES:
            value = (entry.get("roles") or {}).get(role)
            if not value:
                continue
            change = value.get("change_ratio")
            rows.append({
                "校正時間": entry.get("time", ""),
                "相機": _label(role),
                "camera ID": value.get("camera"),
                "量尺長度 (cm)": f'{float(value.get("ruler_cm") or 0):.2f}',
                "像素距離": f'{float(value.get("pixel_distance") or 0):.2f}',
                "cm/pixel": f'{float(value.get("cm_per_pixel") or 0):.6f}',
                "與上次相比": "-" if change is None else f"{change * 100:+.2f}%",
            })
    st.subheader(t("校正歷史"))
    if rows:
        st.dataframe(pd.DataFrame(t_records(rows)), hide_index=True, width="stretch")
    else:
        st.caption(t("尚無校正紀錄"))


def restore_camera_settings() -> None:
    """Once per browser session: take the saved camera IDs and calibrations, so a refresh, a new tab or a
    restart keeps what was set on 相機調整 / 比例尺校正 (no need to choose the IDs again)."""
    if st.session_state.get("_camera_settings_restored"):
        return
    data = _load_camera_settings()
    for role in ROLES:
        saved = (data.get("cameras") or {}).get(role)
        if saved is not None and f"dual_{role}_camera" not in st.session_state:
            st.session_state[f"dual_{role}_camera"] = int(saved)
    if data.get("calibrations") and "dual_calibrations" not in st.session_state:
        st.session_state.dual_calibrations = data["calibrations"]
    st.session_state["_camera_settings_restored"] = True


def camera_settings(page, columns=None):
    """Camera ID inputs. Every page shows the same two IDs: one shared value, saved to disk on change."""
    restore_camera_settings()
    indices = {}
    for i, (role, col) in enumerate(zip(ROLES, columns or st.columns(2))):
        # Permanent state is separate from widgets, which Streamlit removes on page changes.
        saved_key = f"dual_{role}_camera"
        widget_key = f"{page}_{role}_camera"
        value = col.number_input(t("{label} camera ID", label=_label(role)), min_value=0, max_value=30,
                                 value=int(st.session_state.get(saved_key, i)), key=widget_key)
        st.session_state[saved_key] = int(value)
        indices[role] = int(value)
    if indices != (_load_camera_settings().get("cameras") or {}) and (
            _camera_settings_path().exists() or indices != {"comb": 0, "shank": 1}):
        _save_camera_settings(cameras=indices)
    if indices["comb"] == indices["shank"]:
        st.error(t("兩個表型必須使用不同相機 ID"))
    return indices


def _preview_host():
    try:
        host_header = st.context.headers.get("Host", "localhost")
    except Exception:
        host_header = "localhost"
    return urlsplit(f"//{host_header}").hostname or "localhost"


def previews(indices, inference=False, confidence=DEFAULT_CONFIDENCE):
    try:
        port = ensure_preview_server()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    for role, col in zip(ROLES, st.columns(2)):
        with col:
            st.markdown(f"**{_label(role)}**")
            _preview_image(role, indices[role], port, inference, confidence)


def _preview_image(role, index, port, inference=False, confidence=DEFAULT_CONFIDENCE):
    """One camera's live MJPEG preview (plain or with YOLO inference drawn on it)."""
    host = _preview_host()
    if inference:
        source = f"http://{host}:{port}/inference/{role}/{index}.mjpg?confidence={float(confidence):.3f}"
    else:
        source = f"http://{host}:{port}/camera/{index}.mjpg"
    with st.container(key=f"dual_preview_{role}"):
        st.markdown(
            f'<img src="{source}" alt="{_label(role)} camera preview" '
            'style="display:block;width:100%;height:auto;" />',
            unsafe_allow_html=True,
        )


def adjustment_page(heading):
    heading(t("雙相機調整"), t("雞冠與腳脛分別使用獨立相機。"))
    indices = camera_settings("adjustment")
    with st.container(key="adjust_previews"):  # 1.5× larger previews on this page (CSS)
        previews(indices)


@st.fragment
def point_picker(role, photo):
    state_key = f"dual_{role}_points"
    st.session_state.setdefault(state_key, [])
    points = st.session_state[state_key]
    image = Image.open(photo).convert("RGB")
    original_size = image.size
    draw = ImageDraw.Draw(image)
    for n, (x, y) in enumerate(points):
        draw.ellipse((x-6, y-6, x+6, y+6), fill="red")
        draw.text((x+8, y), str(n+1), fill="red")
    display_width = 520
    display_height = max(1, round(image.height * display_width / image.width))
    image = image.resize((display_width, display_height))
    version = st.session_state.get(f"dual_{role}_point_version", 0)
    click = streamlit_image_coordinates(image, width=display_width, height=display_height,
        key=f"dual_points_{role}_{Path(photo).parent.name}_{version}", cursor="crosshair")
    if click and click.get("unix_time") != st.session_state.get(f"dual_{role}_last_click"):
        st.session_state[f"dual_{role}_last_click"] = click.get("unix_time")
        if len(points) < 2:
            x = min(original_size[0]-1, max(0, round(click["x"] * original_size[0] / (click.get("width") or display_width))))
            y = min(original_size[1]-1, max(0, round(click["y"] * original_size[1] / (click.get("height") or display_height))))
            points.append((x, y))
            st.rerun()
    st.caption(t("已選取 {count}/2 個校正點", count=len(points)))
    if st.button(t("清除校正點"), key=f"clear_dual_{role}"):
        st.session_state[state_key] = []
        st.session_state[f"dual_{role}_point_version"] = version + 1
        st.rerun()


def _calibration_ready(role):
    return bool(st.session_state.get(f"dual_calibration_photo_{role}")) and         len(st.session_state.get(f"dual_{role}_points", [])) == 2


def _capture_label(role):
    return t("拍攝 {label} 校正照片", label=_label(role))


def _confirm_label(role):
    return t("確定 {label} 校正", label=_label(role))


def _confirm_calibration(role, index, length):
    """Calibrate ONE camera: its two points + ruler length -> cm/pixel. The other camera's calibration is kept."""
    photo = st.session_state[f"dual_calibration_photo_{role}"]
    if index != st.session_state.get(f"dual_calibration_camera_{role}"):
        raise ValueError(t("拍攝後已更改相機，請重新拍攝校正照片"))
    points = st.session_state[f"dual_{role}_points"]
    (x1, y1), (x2, y2) = points
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if distance <= 0:
        raise ValueError(t("{role} 兩個校正點不可重疊", role=role))
    with Image.open(photo) as image:
        shape = [image.height, image.width]
    calibration = dict(camera=index, shape=shape, cm_per_pixel=length / distance)
    restore_camera_settings()
    calibrations = dict(st.session_state.get("dual_calibrations") or {})
    calibrations[role] = calibration
    st.session_state.dual_calibrations = calibrations
    _save_camera_settings(calibrations=calibrations)
    return record_calibration({role: calibration}, {role: points}, {role: length}, {role: photo})


def calibration_page(heading, hotkey):
    heading(t("雙相機比例尺校正"), t("雞冠與腳脛相機可以分別校正，各自拍照、點兩點、確定即可。"))
    # One card per camera; its camera ID sits inside it.
    cards = [col.container(border=True, key=f"card_calibration_{role}")
             for role, col in zip(ROLES, st.columns(2, gap="medium"))]
    for role, card in zip(ROLES, cards):
        card.subheader(_label(role))
    indices = camera_settings("calibration", columns=cards)
    # N hotkey: confirm a camera whose two points are picked, else take the next camera's photo.
    ready = [role for role in ROLES if _calibration_ready(role)]
    waiting = [role for role in ROLES if not st.session_state.get(f"dual_calibration_photo_{role}")]
    hotkey(_confirm_label(ready[0]) if ready else _capture_label((waiting or list(ROLES))[0]))
    try:
        port = ensure_preview_server()
    except RuntimeError as exc:
        port = None
        st.error(str(exc))
    for role, card in zip(ROLES, cards):
        with card:
            length = st.number_input(t("{label} 比例尺長度 (cm)", label=_label(role)),
                                     min_value=0.001, value=20.0, key=f"dual_{role}_real_length")
            capture_col, confirm_col = st.columns(2)
            if capture_col.button(_capture_label(role), width="stretch", key=f"calibration_capture_{role}"):
                try:
                    frame = capture_one(indices[role])
                    st.session_state[f"dual_calibration_photo_{role}"] = save_one(frame, role)
                    st.session_state[f"dual_calibration_camera_{role}"] = indices[role]
                    st.session_state[f"dual_{role}_points"] = []
                    st.session_state.pop(f"dual_{role}_last_click", None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if confirm_col.button(_confirm_label(role), disabled=not _calibration_ready(role), type="primary",
                                  width="stretch", key=f"calibration_confirm_{role}"):
                try:
                    entry = _confirm_calibration(role, indices[role], length)
                    st.success(t("{label} 校正已套用", label=_label(role)))
                    change = entry["roles"][role]["change_ratio"]
                    if change is not None and abs(change) > CALIBRATION_CHANGE_WARN:
                        st.warning(t("{label} 的 cm/pixel 與上次校正相差 {change}，請確認相機是否被移動或量尺點選是否正確",
                                     label=_label(role), change=f"{change * 100:+.2f}%"))
                except Exception as exc:
                    st.error(str(exc))
            photo = st.session_state.get(f"dual_calibration_photo_{role}")
            if photo:
                point_picker(role, photo)
            elif port:
                _preview_image(role, indices[role], port)
            _calibration_metric(role, indices[role])
    calibration_history_table()


def _calibration_metric(role, index):
    restore_camera_settings()
    value = (st.session_state.get("dual_calibrations") or {}).get(role)
    valid = value and value["camera"] == index
    st.metric(t("{label} cm/pixel", label=_label(role)), f'{value["cm_per_pixel"]:.6f}' if valid else t("未校正"))


def calibration_status(indices, columns=None):
    restore_camera_settings()
    calibrations = st.session_state.get("dual_calibrations", {})
    for role, col in zip(ROLES, columns or st.columns(2)):
        value = calibrations.get(role)
        valid = value and value["camera"] == indices[role]
        col.metric(t("{label} cm/pixel", label=_label(role)),
                   f'{value["cm_per_pixel"]:.6f}' if valid else t("未校正"))
    return calibrations


def _result_images(completed):
    folder = OUTPUTS_DIR / Path(completed["影片"]).stem / "samples"
    with st.container(key="snapshot_result_images"):
        for role, col in zip(ROLES, st.columns(2)):
            path = folder / f"{role}.png"
            if path.exists():
                col.image(str(path), caption=_label(role), width="stretch")


@st.fragment(run_every=1)
def snapshot_results():
    """Latest result (grade, numbers, inference images); sits right of the live previews."""
    jobs = list_jobs()
    latest = jobs[0] if jobs else None
    completed = latest if latest and latest["狀態"] == STATUS_COMPLETED else None
    frozen_weight = st.session_state.get("scale_final_weight_g")
    if ui_style.is_motion():
        _snapshot_results_motion(latest, completed, frozen_weight)
        return
    if latest and not completed and not _progress_ring(latest):
        st.markdown(_current_task_text(latest))
        if frozen_weight is not None:
            st.metric(t("已鎖定體重 (g)"), f"{float(frozen_weight):.2f}")
    if completed and not _finish_hold(completed):
        with st.container(key="snapshot_status"):
            st.markdown(t("**最近完成：{chicken_id} · {grade}**",
                          chicken_id=completed["chicken ID"], grade=t(completed["初步分級"])))
        metric_values = (
            (t("體重 (g)"), completed["體重(g)"], 2),
            (t("雞冠面積 (cm²)"), completed["雞冠面積(cm²)"], 2),
            (t("腳脛寬度 (cm)"), completed["腳脛寬度(cm)"], 2),
            (t("腳脛長度 (cm)"), completed["腳脛長度(cm)"], 2),
        )
        # Numbers 2×2, then the two result images, all in the right-hand column.
        with st.container(key="snapshot_metrics"):
            metric_columns = [*st.columns(2), *st.columns(2)]
            for (label, value, digits), col in zip(metric_values, metric_columns):
                display = "-" if value in (None, "") else f"{float(value):.{digits}f}"
                col.metric(label, display)
        _result_images(completed)


@st.fragment(run_every=1)
def snapshot_job_table():
    jobs = list_jobs()
    if jobs:
        st.dataframe(ui_style.style_grade_columns(pd.DataFrame(_job_table(jobs)), [t("初步分級")]),
                     width="stretch", hide_index=True, height=180)


def ring_percent(job) -> int:
    """Progress of a queued/inferring job as the last finished stage (see ui_style.RING_STAGES):
    photos taken (queued) 40 · comb inferred 60 · shank inferred, writing and grading 80."""
    try:
        done = int(str(job.get("進度", "0/2")).split("/")[0])
    except ValueError:
        done = 0
    return 40 + 20 * max(0, min(done, 2))


def _progress_ring(latest) -> bool:
    """Show the measurement progress ring for an unfinished job. Returns False for a failed job (its message
    is shown instead). The ring slides from the percentage shown last for the same job."""
    if latest["狀態"] == STATUS_FAILED:
        return False
    if st.session_state.get("_capture_ring_on"):
        return True  # the capture ring is still on screen: draw the job's ring only once it is gone
    _show_ring(latest, ring_percent(latest))
    return True


RING_FINISH_HOLD_SECONDS = 1.2  # 100 % stays up at least this long before the result cards replace it


def _show_ring(job, pct) -> None:
    """Draw the ring for `job` at stage `pct`. Its start value is fixed per job (20 after the capture ring,
    else 0) so the HTML stays identical between refreshes; the browser does all the animating."""
    job_id = str(job.get("job_id"))
    ring = st.session_state.get("_ring_last") or {}
    if ring.get("job") != job_id:
        captured = ring.get("job") == "capture"
        ring = {"job": job_id, "first": 20 if captured else 0, "continue": ring.get("key", "") if captured else ""}
        st.session_state["_ring_last"] = ring
    with st.container(key="snapshot_progress"):
        st.markdown(ui_style.progress_ring_html(job["chicken ID"], pct, ring["first"], job_id, ring.get("continue", "")),
                    unsafe_allow_html=True)


def _finish_hold(completed) -> bool:
    """Right after the job the ring was showing completes, show it at 100 % for a moment (the browser glides
    up to 100) before the result cards take the slot. Returns True while holding."""
    ring = st.session_state.get("_ring_last") or {}
    if ring.get("job") != str(completed.get("job_id")) or ring.get("finished"):
        return False
    now = time.monotonic()
    ring.setdefault("done_at", now)
    if now - ring["done_at"] >= RING_FINISH_HOLD_SECONDS:
        ring["finished"] = True
        return False
    _show_ring(completed, 100)
    return True


def _current_task_text(latest):
    return t("**目前任務：{chicken_id} · {status} · {message}**", chicken_id=latest["chicken ID"],
             status=t(latest["狀態"]), message=_job_text(latest["訊息"]))


def _snapshot_results_motion(latest, completed, frozen_weight):
    """動態 style: same content as snapshot_results, with large rolling numbers."""
    if latest and not completed and not _progress_ring(latest):
        st.markdown(_current_task_text(latest))
        if frozen_weight is not None:
            st.markdown(ui_style.metric_cards_html([(t("已鎖定體重 (g)"), f"{float(frozen_weight):.2f}", "weight")],
                                                   "mo-single"), unsafe_allow_html=True)
    if completed and not _finish_hold(completed):
        metric_values = (
            (t("體重 (g)"), completed["體重(g)"], 2, "weight"),
            (t("雞冠面積 (cm²)"), completed["雞冠面積(cm²)"], 2, "comb"),
            (t("腳脛寬度 (cm)"), completed["腳脛寬度(cm)"], 2, "shank"),
            (t("腳脛長度 (cm)"), completed["腳脛長度(cm)"], 2, "shank"),
        )
        cards = [(label, "-" if value in (None, "") else f"{float(value):.{digits}f}", tone)
                 for label, value, digits, tone in metric_values]
        with st.container(key="snapshot_status"):
            st.markdown(ui_style.result_band_html(completed["chicken ID"], completed["初步分級"], cards),
                        unsafe_allow_html=True)
        _result_images(completed)


def phenotype_page(helpers):
    submit_label = t("計算雞冠及腳脛數據")
    helpers["page_heading"](t("雙相機表型量測"), t("Comb 與 Shank 各拍一張照片，合併量測與分級結果。"))
    helpers["install_r_submit_hotkey"](submit_label)
    supplied_frames = {}
    supplied_scales = {}
    # Left: photo-mode toggle, settings row (camera IDs · YOLO confidence), camera previews (or uploaded photos),
    # calibration status. Right, aligned with the toggle: the operator's scan → weigh → measure flow and the latest result.
    stage_column, operator_column = st.columns([1.3, 1], gap="large")  # right block (ID + results) a bit wider
    submitted = bool(st.session_state.pop("_measure_requested", False))
    with operator_column, st.container(key="pheno_operator"):
        for role, path in MODEL_PATHS.items():
            if not path.is_file():
                st.error(t("找不到 {role} 模型：{path}", role=role, path=path))
        query_id = helpers["resolve_qrcode_input"]("phenotype_entry", auto_focus=True,
                                                  scanner_mode=True, show_confirmation=False)
        if query_id:
            if st.session_state.get("scale_active_chicken_id") != query_id:
                st.session_state.scale_active_chicken_id = query_id
                st.session_state.scale_recent_samples = []
                st.session_state.scale_collection_phase = "collecting"
                st.session_state.scale_final_weight_g = None
                st.session_state.scale_final_sample_count = 0
            scale_port = helpers["default_scale_port"]()
            manual_key, use_manual_key = f"dual_manual_weight_{query_id}", f"dual_use_manual_{query_id}"
            if submitted:
                # Measuring now: the panel and form are left out of this run (the layout already matches the
                # next one); their submitted values are read from session state.
                manual_weight = st.session_state.get(manual_key)  # None = left blank
                use_manual = bool(st.session_state.get(use_manual_key, not scale_port)) or not scale_port
            else:
              # One keyed block (scale panel + form + error) so the browser can swap it for a ring the instant the
              # button is pressed (RING_DRIVER_JS), before the server answers.
              with st.container(key="measure_form_area"):
                helpers["live_scale_panel"](scale_port, window_seconds=2.0)
                with st.form("dual_snapshot_form"):
                    st.markdown(t("**Chicken ID：{chicken_id}**", chicken_id=query_id))
                    weight_col, manual_col = st.columns([1.4, 1], vertical_alignment="bottom")
                    # Blank for every chicken (never the previous / stored weight), so nothing is carried over.
                    weight_col.number_input(t("手動重量 (g)"), min_value=0.0, value=None, step=10.0,
                                            placeholder=t("請輸入體重"), key=manual_key)
                    # Manual weight is used only when chosen explicitly (or when no scale is configured).
                    manual_col.checkbox(t("使用手動重量"), value=not scale_port, disabled=not scale_port,
                                        key=use_manual_key)
                    st.form_submit_button(
                        submit_label,
                        type="primary",
                        width="stretch",
                        on_click=begin_final_weight_collection,
                    )
                error = st.session_state.pop("_measure_error", None)
                if error:
                    st.error(error)  # next to the button (not at the page bottom)
        st.html(ui_style.ring_driver_html(), unsafe_allow_javascript=True)  # browser-side ring animation (installed once)
        st.html(ui_style.IMAGE_ZOOM_HTML, unsafe_allow_javascript=True)  # click an inference photo -> full screen
        # The capture ring lives here, a slot that exists in EVERY run (the form above disappears once the ID
        # is cleared), so the next run clears it before the job's ring below is drawn: never two rings.
        capture_slot = st.empty()
        if submitted and query_id:
            # Ring at 20 % straight away, in the same place the job's ring will take over after the re-run.
            capture_key = f"capture-{time.time_ns()}"
            st.session_state["_capture_ring_on"] = True
            capture_slot.markdown(ui_style.progress_ring_html(query_id, 20, 0, capture_key, "instant"),
                                  unsafe_allow_html=True)
            st.session_state["_ring_last"] = {"job": "capture", "key": capture_key}
        else:
            st.session_state["_capture_ring_on"] = False  # this run just cleared the slot above
        snapshot_results()  # latest numbers beside the live previews, visible without scrolling
    # The left column (camera previews / uploaded photos) is filled AFTER the operator column: the right side
    # (scan box, progress ring) never needs it, so a click shows the ring without waiting for the previews.
    with stage_column:
        photo_mode = st.toggle(t("使用照片測試模式（不開啟雙相機）"), key="dual_photo_mode")
        settings_columns = st.columns(3, vertical_alignment="bottom")
        restore_camera_settings()
        if photo_mode:
            indices = {
                role: int(st.session_state.get(f"dual_{role}_camera", i))
                for i, role in enumerate(ROLES)
            }
            calibrations = st.session_state.get("dual_calibrations", {})
            confidence_column = settings_columns[0]
        else:
            indices = camera_settings("phenotype", settings_columns[0:2])
            confidence_column = settings_columns[2]
        confidence = confidence_column.number_input(t("YOLO confidence 門檻"), min_value=0.01, max_value=1.0,
                                                    value=DEFAULT_CONFIDENCE, step=0.05, key="dual_confidence")
        if photo_mode:
            upload_area = st.container(key="photo_mode_uploads")
            for role, col in zip(ROLES, upload_area.columns(2)):
                uploaded = col.file_uploader(
                    t("上傳 {label} 照片", label=_label(role)),
                    type=["jpg", "jpeg", "png", "bmp"],
                    key=f"dual_{role}_photo",
                )
                if uploaded is not None:
                    try:
                        supplied_frames[role] = decode_uploaded_frame(uploaded)
                        col.image(supplied_frames[role], channels="BGR", width="stretch")
                    except ValueError as exc:
                        col.error(str(exc))
            for role, col in zip(ROLES, st.columns(2)):
                default_scale = float(calibrations.get(role, {}).get("cm_per_pixel") or 0.01)
                supplied_scales[role] = col.number_input(
                    t("{label} cm/pixel", label=_label(role)),
                    min_value=0.000001,
                    value=default_scale,
                    step=0.000001,
                    format="%.6f",
                    key=f"dual_{role}_photo_scale",
                )
        else:
            previews(indices, inference=True, confidence=confidence)
            calibrations = calibration_status(indices)  # each camera's cm/pixel under its preview
    if query_id:
        if submitted:
            try:
                if photo_mode and set(supplied_frames) != set(ROLES):
                    raise ValueError(t("請先上傳雞冠與腳脛兩張照片"))
                if not photo_mode and not all(calibrations.get(role, {}).get("camera") == indices[role] for role in ROLES):
                    raise ValueError(t("請先完成兩台相機的比例尺校正"))
                for path in MODEL_PATHS.values():
                    if not path.is_file():
                        raise FileNotFoundError(str(path))
                # Weight first (from the live panel's last 2 s — no second read), then the photos.
                recent = (helpers["scale_monitor"](scale_port).snapshot(WEIGHT_WINDOW_SECONDS + 0.6).samples
                          if scale_port and not use_manual else [])
                weight, sample_count = resolve_weight(scale_port, use_manual, manual_weight, recent)
                with st.spinner(t("拍攝中…")):
                    paths, scales = acquire_measurement_inputs(
                        indices,
                        calibrations,
                        supplied_frames=supplied_frames if photo_mode else None,
                        supplied_scales=supplied_scales if photo_mode else None,
                    )
                st.session_state.scale_final_weight_g = weight
                st.session_state.scale_final_sample_count = sample_count
                st.session_state.scale_collection_phase = "stopped"
                # Next scan — even of the same ID (a retake) — starts a fresh live weight collection.
                st.session_state.scale_active_chicken_id = None
                job_id = enqueue_snapshot_job(query_id, weight, paths, scales, confidence,
                    bool(st.session_state.get("arduino_light_enabled", True)), helpers["default_arduino_port"]())
                st.session_state.dual_last_job = job_id
                st.session_state.phenotype_entry_confirmed_manual_id = ""
                st.session_state.phenotype_entry_input_version = int(st.session_state.get("phenotype_entry_input_version", 0)) + 1
                # Remove the capture ring now; otherwise it lingers (stale) next to the job's ring while the
                # page re-runs. The job's ring continues from the capture ring's value.
                st.rerun()
            except Exception as exc:
                if st.session_state.get("scale_collection_phase") == "finalizing":
                    st.session_state.scale_collection_phase = "collecting"
                st.session_state["_capture_ring_on"] = False
                st.session_state["_ring_last"] = {}
                st.session_state["_measure_error"] = str(exc)  # shown under the button once the form is back
                st.rerun()
    if st.session_state.get("dual_last_job"):
        st.caption(t("最近建立任務：{job_id}", job_id=st.session_state.dual_last_job))
    snapshot_job_table()
