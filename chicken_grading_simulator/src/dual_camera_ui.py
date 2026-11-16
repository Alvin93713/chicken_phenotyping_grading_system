"""Two-camera configuration, calibration, and snapshot workflow."""
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from dual_camera import (
    MODEL_PATHS, ROLES, capture_pair, ensure_preview_server, save_pair,
    validate_calibration,
)
from inference_queue import enqueue_snapshot_job, list_jobs, STATUS_COMPLETED
from inference_utils import OUTPUTS_DIR

LABELS = {"comb": "Comb 雞冠", "shank": "Shank 腳脛"}


def begin_final_weight_collection():
    st.session_state.scale_collection_phase = "finalizing"


def acquire_measurement_inputs(
    indices,
    calibrations,
    scale_port,
    manual_weight,
    read_average,
    supplied_frames=None,
    supplied_scales=None,
):
    """Freeze one weight result before acquiring the two camera frames."""
    weight = float(manual_weight)
    scale_result = None
    if scale_port:
        scale_result = read_average(scale_port, duration_seconds=2.0)
        if scale_result.average_g is not None:
            weight = float(scale_result.average_g)

    if supplied_frames is None:
        frames = capture_pair(indices)
        scales = {
            role: validate_calibration(calibrations.get(role), indices[role], frames[role].shape)
            for role in ROLES
        }
    else:
        if set(supplied_frames) != set(ROLES) or set(supplied_scales or {}) != set(ROLES):
            raise ValueError("請上傳雞冠與腳脛兩張照片，並設定兩組 cm/pixel")
        frames = supplied_frames
        scales = {role: float(supplied_scales[role]) for role in ROLES}
        if any(value <= 0 for value in scales.values()):
            raise ValueError("照片模式的 cm/pixel 必須大於 0")
    return weight, scale_result, save_pair(frames), scales


def decode_uploaded_frame(uploaded_file):
    import cv2
    import numpy as np

    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("無法讀取上傳的照片，請使用 JPG、PNG 或 BMP")
    return frame


def camera_settings(page):
    indices = {}
    for i, (role, col) in enumerate(zip(ROLES, st.columns(2))):
        # Permanent state is separate from widgets, which Streamlit removes on page changes.
        saved_key = f"dual_{role}_camera"
        widget_key = f"{page}_{role}_camera"
        value = col.number_input(f"{LABELS[role]} camera ID", min_value=0, max_value=30,
                                 value=int(st.session_state.get(saved_key, i)), key=widget_key)
        st.session_state[saved_key] = int(value)
        indices[role] = int(value)
    if indices["comb"] == indices["shank"]:
        st.error("兩個表型必須使用不同相機 ID")
    return indices


def _preview_host():
    try:
        host_header = st.context.headers.get("Host", "localhost")
    except Exception:
        host_header = "localhost"
    return urlsplit(f"//{host_header}").hostname or "localhost"


def previews(indices, inference=False, confidence=0.25):
    try:
        port = ensure_preview_server()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    host = _preview_host()
    for role, col in zip(ROLES, st.columns(2)):
        with col:
            st.markdown(f"**{LABELS[role]}**")
            if inference:
                source = (
                    f"http://{host}:{port}/inference/{role}/{indices[role]}.mjpg"
                    f"?confidence={float(confidence):.3f}"
                )
            else:
                source = f"http://{host}:{port}/camera/{indices[role]}.mjpg"
            with st.container(key=f"dual_preview_{role}"):
                st.markdown(
                    f'<img src="{source}" alt="{LABELS[role]} camera preview" '
                    'style="display:block;width:100%;height:auto;" />',
                    unsafe_allow_html=True,
                )


def adjustment_page(heading):
    heading("雙相機調整", "雞冠與腳脛分別使用獨立相機。")
    indices = camera_settings("adjustment")
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
    st.caption(f"已選取 {len(points)}/2 個校正點")
    if st.button("清除校正點", key=f"clear_dual_{role}"):
        st.session_state[state_key] = []
        st.session_state[f"dual_{role}_point_version"] = version + 1
        st.rerun()


def calibration_page(heading, hotkey):
    heading("雙相機比例尺校正", "分別設定雞冠與腳脛相機的比例尺。")
    indices = camera_settings("calibration")
    lengths = {}
    for role, col in zip(ROLES, st.columns(2)):
        lengths[role] = col.number_input(f"{LABELS[role]} 比例尺長度 (cm)",
            min_value=0.001, value=20.0, key=f"dual_{role}_real_length")
    photos = st.session_state.get("dual_calibration_photos")
    ready = photos and all(len(st.session_state.get(f"dual_{r}_points", [])) == 2 for r in ROLES)
    hotkey("確定雙相機校正" if ready else "拍攝雙相機校正照片")
    a, b = st.columns(2)
    if a.button("拍攝雙相機校正照片", use_container_width=True):
        try:
            frames = capture_pair(indices)
            st.session_state.dual_calibration_photos = save_pair(frames, "calibration_captures")
            st.session_state.dual_calibration_indices = dict(indices)
            for role in ROLES:
                st.session_state[f"dual_{role}_points"] = []
                st.session_state.pop(f"dual_{role}_last_click", None)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if b.button("確定雙相機校正", disabled=not ready, type="primary", use_container_width=True):
        try:
            if indices != st.session_state.dual_calibration_indices:
                raise ValueError("拍攝後已更改相機，請重新拍攝校正照片")
            calibrations = {}
            for role in ROLES:
                (x1,y1),(x2,y2) = st.session_state[f"dual_{role}_points"]
                distance = ((x2-x1)**2+(y2-y1)**2)**0.5
                if distance <= 0:
                    raise ValueError(f"{role} 兩個校正點不可重疊")
                with Image.open(photos[role]) as image:
                    shape = [image.height, image.width]
                calibrations[role] = dict(camera=indices[role], shape=shape,
                                           cm_per_pixel=lengths[role]/distance)
            st.session_state.dual_calibrations = calibrations
            st.success("雙相機校正已套用")
        except Exception as exc:
            st.error(str(exc))
    if photos:
        for role, col in zip(ROLES, st.columns(2)):
            with col:
                st.markdown(f"**{LABELS[role]}**")
                point_picker(role, photos[role])
    else:
        previews(indices)
    calibration_status(indices)


def calibration_status(indices):
    calibrations = st.session_state.get("dual_calibrations", {})
    for role, col in zip(ROLES, st.columns(2)):
        value = calibrations.get(role)
        valid = value and value["camera"] == indices[role]
        col.metric(f"{LABELS[role]} cm/pixel", f'{value["cm_per_pixel"]:.6f}' if valid else "未校正")
    return calibrations


@st.fragment(run_every=1)
def snapshot_results():
    jobs = list_jobs()
    latest = jobs[0] if jobs else None
    completed = latest if latest and latest["狀態"] == STATUS_COMPLETED else None
    frozen_weight = st.session_state.get("scale_final_weight_g")
    if latest and not completed:
        st.markdown(f'**目前任務：{latest["chicken ID"]} · {latest["狀態"]} · {latest["訊息"]}**')
        if frozen_weight is not None:
            st.metric("已鎖定重量 (g)", f"{float(frozen_weight):.1f}")
    if completed:
        st.markdown(f'**最近完成：{completed["chicken ID"]} · {completed["初步分級"]}**')
        metric_values = (
            ("重量 (g)", completed["體重(g)"], 1),
            ("雞冠面積 (cm²)", completed["雞冠面積(cm²)"], 4),
            ("腳脛寬度 (cm)", completed["腳脛寬度(cm)"], 4),
            ("腳脛長度 (cm)", completed["腳脛長度(cm)"], 4),
        )
        for (label, value, digits), col in zip(metric_values, st.columns(4)):
            display = "-" if value in (None, "") else f"{float(value):.{digits}f}"
            col.metric(label, display)
        folder = OUTPUTS_DIR / Path(completed["影片"]).stem / "samples"
        for role, col in zip(ROLES, st.columns(2)):
            path = folder / f"{role}.png"
            if path.exists():
                col.image(str(path), caption=LABELS[role], use_container_width=True)
    if jobs:
        st.dataframe([{k:v for k,v in j.items() if k not in ("Arduino", "影片", "建立時間")} for j in jobs],
                     use_container_width=True, hide_index=True, height=180)


def phenotype_page(helpers):
    helpers["page_heading"]("雙相機表型量測", "Comb 與 Shank 各拍一張照片，合併量測與分級結果。")
    helpers["install_r_submit_hotkey"]("計算雞冠及腳脛數據")
    photo_mode = st.toggle("使用照片測試模式（不開啟雙相機）", key="dual_photo_mode")
    if photo_mode:
        indices = {
            role: int(st.session_state.get(f"dual_{role}_camera", i))
            for i, role in enumerate(ROLES)
        }
        calibrations = st.session_state.get("dual_calibrations", {})
    else:
        indices = camera_settings("phenotype")
        calibrations = calibration_status(indices)
    confidence = st.number_input("YOLO confidence 門檻", min_value=0.01, max_value=1.0,
                                 value=0.25, step=0.05, key="dual_confidence")
    supplied_frames = {}
    supplied_scales = {}
    if photo_mode:
        st.info("照片模式不會開啟 Camera 0/1。cm/pixel 會直接影響面積、寬度與長度結果。")
        for role, col in zip(ROLES, st.columns(2)):
            uploaded = col.file_uploader(
                f"上傳 {LABELS[role]} 照片",
                type=["jpg", "jpeg", "png", "bmp"],
                key=f"dual_{role}_photo",
            )
            if uploaded is not None:
                try:
                    supplied_frames[role] = decode_uploaded_frame(uploaded)
                    col.image(supplied_frames[role], channels="BGR", use_container_width=True)
                except ValueError as exc:
                    col.error(str(exc))
        for role, col in zip(ROLES, st.columns(2)):
            default_scale = float(calibrations.get(role, {}).get("cm_per_pixel") or 0.01)
            supplied_scales[role] = col.number_input(
                f"{LABELS[role]} cm/pixel",
                min_value=0.000001,
                value=default_scale,
                step=0.000001,
                format="%.6f",
                key=f"dual_{role}_photo_scale",
            )
    else:
        st.caption("即時 YOLO segmentation 預覽（背景推論目標 15 FPS；正式量測仍使用原始單張照片）")
        previews(indices, inference=True, confidence=confidence)
    for role, path in MODEL_PATHS.items():
        if not path.is_file():
            st.error(f"找不到 {role} 模型：{path}")
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
        helpers["live_scale_panel"](scale_port, window_seconds=2.0)
        selected = helpers["get_chicken"](query_id)
        with st.form("dual_snapshot_form"):
            st.markdown(f"**Chicken ID：{query_id}**")
            manual_weight = st.number_input("手動重量備援 (g)", min_value=0.0,
                value=float((selected or {}).get("weight_g") or 0), step=10.0)
            submitted = st.form_submit_button(
                "計算雞冠及腳脛數據",
                type="primary",
                use_container_width=True,
                on_click=begin_final_weight_collection,
            )
        if submitted:
            try:
                if photo_mode and set(supplied_frames) != set(ROLES):
                    raise ValueError("請先上傳雞冠與腳脛兩張照片")
                if not photo_mode and not all(calibrations.get(role, {}).get("camera") == indices[role] for role in ROLES):
                    raise ValueError("請先完成兩台相機的比例尺校正")
                for path in MODEL_PATHS.values():
                    if not path.is_file():
                        raise FileNotFoundError(str(path))
                with st.spinner("正在收集 2 秒平均重量，完成後將停止秤重並拍攝雙相機照片..."):
                    weight, result, paths, scales = acquire_measurement_inputs(
                        indices,
                        calibrations,
                        scale_port,
                        manual_weight,
                        helpers["read_scale_average"],
                        supplied_frames=supplied_frames if photo_mode else None,
                        supplied_scales=supplied_scales if photo_mode else None,
                    )
                sample_count = len(result.samples) if result is not None else 0
                if result is not None and result.average_g is None:
                    st.warning("未取得有效秤重值，使用手動重量備援")
                st.session_state.scale_final_weight_g = weight
                st.session_state.scale_final_sample_count = sample_count
                st.session_state.scale_collection_phase = "stopped"
                job_id = enqueue_snapshot_job(query_id, weight, paths, scales, confidence,
                    bool(st.session_state.get("arduino_light_enabled", True)), helpers["default_arduino_port"]())
                st.session_state.dual_last_job = job_id
                st.session_state.phenotype_entry_confirmed_manual_id = ""
                st.session_state.phenotype_entry_input_version = int(st.session_state.get("phenotype_entry_input_version", 0)) + 1
                st.rerun()
            except Exception as exc:
                if st.session_state.get("scale_collection_phase") == "finalizing":
                    st.session_state.scale_collection_phase = "collecting"
                st.error(str(exc))
    if st.session_state.get("dual_last_job"):
        st.caption(f"最近建立任務：{st.session_state.dual_last_job}")
    snapshot_results()
