"""雞隻表型資料分級與亮燈篩選系統。"""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from database import (
    add_chicken,
    delete_chicken,
    export_csv_bytes,
    get_all_chickens,
    get_all_chickens_df,
    get_chicken,
    import_csv,
    init_db,
    get_latest_grading_config,
    mark_final_scanned,
    update_chicken,
    update_grading_result,
)
from arduino_control import commands_for_grade, list_serial_ports, reconnect_arduino, send_arduino_commands
from scale_control import (
    SCALE_BAUD_RATE,
    append_recent_scale_samples,
    average_recent_samples,
    read_scale_average,
    read_scale_samples,
)
from grading import (
    FEATURE_COLUMNS,
    GRADE_FAIL,
    GRADE_PASS,
    GRADE_STANDBY,
    active_features,
    grade_chicken,
    summarize_results,
    validate_grading_config,
)
from inference_utils import (
    OUTPUTS_DIR,
    capture_calibration_image,
    validate_yolo_seg_assets,
)
from inference_queue import (
    STATUS_COMPLETED,
    enqueue_inference_job,
    has_active_jobs,
    latest_completed_job_for_chicken,
    list_jobs,
)
from sample_data import create_sample_data


CAMERA_PREVIEW_WIDTH_PX = 672
CAMERA_PREVIEW_HEIGHT_PX = 420
CAMERA_PREVIEW_ASPECT_RATIO = CAMERA_PREVIEW_WIDTH_PX / CAMERA_PREVIEW_HEIGHT_PX
FINAL_LIGHT_SECONDS = 3.0


st.set_page_config(
    page_title="雞隻表型資料分級與亮燈篩選系統",
    page_icon="🐔",
    layout="wide",
)

init_db()


def number_input_value(value, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return float(value)


def install_r_submit_hotkey(button_text: str) -> None:
    escaped_text = json.dumps(button_text, ensure_ascii=False)
    components.html(
        f"""
        <script>
        (() => {{
            const buttonText = {escaped_text};
            try {{
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                if (parentWindow.__chickenrHotkeyHandler) {{
                    parentDocument.removeEventListener(
                        "keydown",
                        parentWindow.__chickenrHotkeyHandler,
                        true
                    );
                }}
                const handler = (event) => {{
                    if (event.key.toLowerCase() !== "r" && event.code !== "KeyR") {{
                        return;
                    }}
                    const buttons = Array.from(parentDocument.querySelectorAll("button"));
                    const target = buttons.find((button) =>
                        (button.innerText || button.textContent || "").trim().includes(buttonText)
                    );
                    if (!target || target.disabled) {{
                        return;
                    }}
                    event.preventDefault();
                    event.stopPropagation();
                    target.click();
                }};
                parentWindow.__chickenrHotkeyHandler = handler;
                parentDocument.addEventListener("keydown", handler, true);
            }} catch (error) {{
                console.warn("r hotkey could not be installed.", error);
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def install_n_scanner_focus_hotkey() -> None:
    components.html(
        """
        <script>
        (() => {
            try {
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                if (parentWindow.__chickennScannerFocusHotkeyHandler) {
                    parentDocument.removeEventListener(
                        "keydown",
                        parentWindow.__chickennScannerFocusHotkeyHandler,
                        true
                    );
                }
                const focusScannerInput = () => {
                    const inputs = Array.from(parentDocument.querySelectorAll('input:not([type="file"])'));
                    const target = inputs.find((input) =>
                        !input.disabled &&
                        input.offsetParent !== null &&
                        (input.getAttribute("placeholder") || "").trim().toUpperCase() === "C001"
                    );
                    if (!target) {
                        return false;
                    }
                    target.focus();
                    if (!target.value) {
                        target.select();
                    }
                    return true;
                };
                const handler = (event) => {
                    if (event.key.toLowerCase() !== "n" && event.code !== "KeyN") {
                        return;
                    }
                    event.preventDefault();
                    event.stopPropagation();
                    if (!focusScannerInput()) {
                        window.setTimeout(focusScannerInput, 40);
                    }
                };
                parentWindow.__chickennScannerFocusHotkeyHandler = handler;
                parentDocument.addEventListener("keydown", handler, true);
            } catch (error) {
                console.warn("n scanner-focus hotkey could not be installed.", error);
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def install_scanner_input_autofocus() -> None:
    components.html(
        """
        <script>
        (() => {
            const autofocusWindowMs = 600000;
            const userPauseMs = 3500;
            const startedAt = Date.now();
            const parentDocument = window.parent.document;
            const parentWindow = window.parent;
            if (parentWindow.__chickenScannerAutofocusPointerHandler) {
                parentDocument.removeEventListener(
                    "pointerdown",
                    parentWindow.__chickenScannerAutofocusPointerHandler,
                    true
                );
            }
            if (parentWindow.__chickenScannerAutofocusKeyHandler) {
                parentDocument.removeEventListener(
                    "keydown",
                    parentWindow.__chickenScannerAutofocusKeyHandler,
                    true
                );
            }
            parentWindow.__chickenScannerAutofocusPausedUntil =
                parentWindow.__chickenScannerAutofocusPausedUntil || 0;
            const isScannerInput = (element) =>
                element &&
                element.matches &&
                element.matches('input:not([type="file"])') &&
                (element.getAttribute("placeholder") || "").trim().toUpperCase() === "C001";
            const pauseAutofocus = () => {
                parentWindow.__chickenScannerAutofocusPausedUntil = Date.now() + userPauseMs;
            };
            const pointerHandler = (event) => {
                if (!isScannerInput(event.target)) {
                    pauseAutofocus();
                }
            };
            const keyHandler = (event) => {
                const active = parentDocument.activeElement;
                if (isScannerInput(active)) {
                    return;
                }
                if (event.key === "Tab" || event.key === "Enter" || event.key === "Escape") {
                    pauseAutofocus();
                }
            };
            parentWindow.__chickenScannerAutofocusPointerHandler = pointerHandler;
            parentWindow.__chickenScannerAutofocusKeyHandler = keyHandler;
            parentDocument.addEventListener("pointerdown", pointerHandler, true);
            parentDocument.addEventListener("keydown", keyHandler, true);
            const focusScannerInput = () => {
                try {
                    if (Date.now() < parentWindow.__chickenScannerAutofocusPausedUntil) {
                        return false;
                    }
                    const active = parentDocument.activeElement;
                    const expandedCombobox = parentDocument.querySelector('[role="combobox"][aria-expanded="true"]');
                    if (
                        expandedCombobox ||
                        (active && !isScannerInput(active) && active !== parentDocument.body)
                    ) {
                        return false;
                    }
                    const inputs = Array.from(parentDocument.querySelectorAll('input:not([type="file"])'));
                    const target = inputs.find((input) =>
                        !input.disabled &&
                        input.offsetParent !== null &&
                        (input.getAttribute("placeholder") || "").trim().toUpperCase() === "C001"
                    );
                    if (!target) {
                        return false;
                    }
                    target.focus();
                    if (!target.value) {
                        target.select();
                    }
                    return true;
                } catch (error) {
                    console.warn("scanner input autofocus failed.", error);
                    return false;
                }
            };
            [0, 50, 150, 300, 700, 1200].forEach((delay) => {
                window.setTimeout(focusScannerInput, delay);
            });
            const intervalId = window.setInterval(() => {
                focusScannerInput();
                if (Date.now() - startedAt > autofocusWindowMs) {
                    window.clearInterval(intervalId);
                }
            }, 150);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def install_n_click_button_hotkey(button_text: str) -> None:
    escaped_button_text = json.dumps(button_text, ensure_ascii=False)
    components.html(
        f"""
        <script>
        (() => {{
            try {{
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                if (parentWindow.__chickennScannerFocusHotkeyHandler) {{
                    parentDocument.removeEventListener(
                        "keydown",
                        parentWindow.__chickennScannerFocusHotkeyHandler,
                        true
                    );
                }}
                const targetText = {escaped_button_text};
                const clickTargetButton = () => {{
                    const buttons = Array.from(parentDocument.querySelectorAll("button"));
                    const target = buttons.find((button) =>
                        !button.disabled &&
                        button.offsetParent !== null &&
                        (button.innerText || button.textContent || "").trim() === targetText
                    );
                    if (!target) {{
                        return false;
                    }}
                    target.click();
                    return true;
                }};
                const handler = (event) => {{
                    if (event.key.toLowerCase() !== "n" && event.code !== "KeyN") {{
                        return;
                    }}
                    event.preventDefault();
                    event.stopPropagation();
                    if (!clickTargetButton()) {{
                        window.setTimeout(clickTargetButton, 40);
                    }}
                }};
                parentWindow.__chickennScannerFocusHotkeyHandler = handler;
                parentDocument.addEventListener("keydown", handler, true);
            }} catch (error) {{
                console.warn("n button-click hotkey could not be installed.", error);
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def install_p_page_cycle_hotkey(page_names: list[str], current_page: str) -> None:
    if not page_names:
        return
    escaped_page_names = json.dumps(page_names, ensure_ascii=False)
    escaped_current_page = json.dumps(current_page if current_page in page_names else page_names[0], ensure_ascii=False)
    components.html(
        f"""
        <script>
        (() => {{
            const pageNames = {escaped_page_names};
            const fallbackCurrentPage = {escaped_current_page};
            try {{
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                if (parentWindow.__chickenPageCycleHotkeyHandler) {{
                    parentDocument.removeEventListener(
                        "keydown",
                        parentWindow.__chickenPageCycleHotkeyHandler,
                        true
                    );
                }}
                const cyclePage = () => {{
                    const allButtons = Array.from(parentDocument.querySelectorAll("button"));
                    const navButtons = pageNames
                        .map((pageName) => allButtons.find((button) =>
                            (button.innerText || button.textContent || "").trim() === pageName
                        ))
                        .filter(Boolean);
                    if (navButtons.length !== pageNames.length) {{
                        return false;
                    }}
                    const selectedIndex = navButtons.findIndex((button) =>
                        button.getAttribute("aria-pressed") === "true" ||
                        button.getAttribute("aria-selected") === "true" ||
                        button.getAttribute("data-selected") === "true"
                    );
                    const fallbackIndex = Math.max(0, pageNames.indexOf(fallbackCurrentPage));
                    const currentIndex = selectedIndex >= 0 ? selectedIndex : fallbackIndex;
                    const target = navButtons[(currentIndex + 1) % navButtons.length];
                    if (!target || target.disabled) {{
                        return false;
                    }}
                    target.click();
                    return true;
                }};
                const handler = (event) => {{
                    if (event.key.toLowerCase() !== "p" && event.code !== "KeyP") {{
                        return;
                    }}
                    event.preventDefault();
                    event.stopPropagation();
                    if (!cyclePage()) {{
                        window.setTimeout(cyclePage, 40);
                    }}
                }};
                parentWindow.__chickenPageCycleHotkeyHandler = handler;
                parentDocument.addEventListener("keydown", handler, true);
            }} catch (error) {{
                console.warn("p page-cycle hotkey could not be installed.", error);
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


LIGHT_CSS = """
<style>
:root {
    --content-width: 1500px;
    --panel-border: rgba(128, 139, 154, 0.32);
}
header[data-testid="stHeader"] {
    height: 2rem;
    background: transparent;
}
[data-testid="stMainBlockContainer"] {
    width: min(96vw, var(--content-width));
    max-width: var(--content-width);
    padding: 0.65rem 1.25rem 1rem;
}
[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.75rem;
}
.app-masthead {
    display: flex;
    align-items: baseline;
    gap: 0.9rem;
    margin: 0 0 0.5rem;
}
.app-title {
    font-size: 1.75rem;
    font-weight: 760;
    line-height: 1.15;
}
.app-subtitle {
    color: #9ca3af;
    font-size: 0.88rem;
}
.page-heading {
    display: flex;
    align-items: baseline;
    gap: 0.8rem;
    margin: 0.15rem 0 0.45rem;
}
.page-heading h2 {
    font-size: 1.45rem;
    line-height: 1.2;
    margin: 0;
}
.page-heading p {
    color: #9ca3af;
    font-size: 0.82rem;
    margin: 0;
}
.light-panel {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 0;
}
.light-circle {
    width: 92px;
    height: 92px;
    border-radius: 50%;
    border: 6px solid rgba(0, 0, 0, 0.12);
    box-shadow: inset 0 7px 18px rgba(255,255,255,0.5), 0 7px 18px rgba(0,0,0,0.14);
}
.light-green { background: #16a34a; }
.light-yellow { background: #facc15; }
.light-red { background: #dc2626; }
.light-gray { background: #9ca3af; }
.light-text {
    font-size: 24px;
    font-weight: 700;
    line-height: 1.35;
}
.status-note {
    color: #4b5563;
    font-size: 16px;
    margin-top: 8px;
}
[data-testid="stSegmentedControl"] button {
    font-size: 0.92rem;
    min-height: 2.25rem;
    padding: 0.25rem 0.7rem;
    white-space: nowrap;
}
[data-testid="stSegmentedControl"] [role="radiogroup"] {
    flex-wrap: nowrap;
    overflow-x: auto;
}
[data-testid="stDivider"] {
    margin: 0.2rem 0 0.35rem;
}
[data-testid="stAlert"] {
    padding: 0.5rem 0.75rem;
}
[data-testid="stMetric"] {
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 0.45rem 0.65rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem;
}
[data-testid="stExpander"] details summary {
    min-height: 2.3rem;
}
[data-testid="stForm"] {
    border-color: var(--panel-border);
    padding: 0.75rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.75rem;
}
.threshold-summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.65rem;
    margin-bottom: 0.45rem;
}
.threshold-summary-card {
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 0.55rem 0.7rem;
    text-align: center;
}
.threshold-summary-label {
    font-size: 0.78rem;
    color: #9ca3af;
}
.threshold-summary-value {
    font-size: 1rem;
    font-weight: 700;
    margin-top: 0.15rem;
}
.threshold-summary-disabled {
    opacity: 0.55;
}
[data-testid="stCameraInput"] {
    width: 100% !important;
    max-width: 960px !important;
    max-height: calc(100vh - 205px);
    height: auto !important;
    margin: 0 auto;
    overflow: hidden;
}
[data-testid="stCameraInput"] > div,
[data-testid="stCameraInput"] section,
[data-testid="stCameraInput"] div,
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] canvas {
    width: 100% !important;
    max-width: 960px !important;
}
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img,
[data-testid="stCameraInput"] canvas {
    width: 100% !important;
    height: auto !important;
    max-height: calc(100vh - 205px);
    object-fit: contain !important;
    background: #000;
}
[data-testid="stCameraInput"] button {
    display: none;
}
iframe {
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: min(100%, 960px) !important;
}
.st-key-pass_threshold_block {
    border: 4px solid #16a34a;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 6px;
}
.st-key-standby_threshold_block {
    border: 4px solid #facc15;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 6px;
}
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        width: 100%;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }
    .app-masthead,
    .page-heading {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.2rem;
    }
    .threshold-summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def page_heading(title: str, description: str = "") -> None:
    description_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        f'<div class="page-heading"><h2>{title}</h2>{description_html}</div>',
        unsafe_allow_html=True,
    )


def show_data_table(hide_result_columns: bool = False, height: int | None = None) -> None:
    df = get_all_chickens_df().rename(columns={"chicken_id": "chicken ID"})
    if hide_result_columns:
        df = df.drop(columns=["grade_result", "light_result"], errors="ignore")
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def image_data_uri(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def show_inference_sample_frames(video_ids: list[str] | None = None) -> None:
    metadata_paths = sorted(OUTPUTS_DIR.glob("*/metadata.json"))
    if video_ids is not None:
        wanted_ids = {str(video_id) for video_id in video_ids}
        metadata_paths = [path for path in metadata_paths if path.parent.name in wanted_ids]

    image_paths: list[Path] = []
    for metadata_path in metadata_paths:
        metadata_image_paths: list[Path] = []
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        for frame_path in metadata.get("sample_frames", []):
            path = Path(frame_path)
            if not path.exists():
                path = metadata_path.parent / "samples" / Path(frame_path).name
            if path.exists():
                metadata_image_paths.append(path)
        if metadata_image_paths:
            image_paths.extend(metadata_image_paths)
        else:
            sample_dir = metadata_path.parent / "samples"
            if sample_dir.exists():
                image_paths.extend(sorted(sample_dir.glob("*.jpg")))
                image_paths.extend(sorted(sample_dir.glob("*.png")))
    image_paths = list({path.resolve(): path for path in image_paths}.values())

    if not image_paths:
        st.info("目前沒有可顯示的推論影格。")
        return

    st.subheader("YOLO推論影格")
    columns = st.columns(3)
    for index, image_path in enumerate(image_paths):
        with columns[index % 3]:
            st.caption(image_path.parent.parent.name)
            st.markdown(
                f"""
                <img
                    src="{image_data_uri(image_path)}"
                    style="width: 100%; height: auto; pointer-events: none; user-select: none;"
                    draggable="false"
                />
                """,
                unsafe_allow_html=True,
            )


@st.fragment(run_every=2)
def render_inference_queue_panel() -> None:
    jobs = list_jobs()
    st.subheader("背景推論任務佇列")
    if not jobs:
        st.info("目前沒有背景推論任務。")
        return

    display_df = pd.DataFrame(jobs).drop(columns=["Arduino", "影片", "錯誤", "建立時間"], errors="ignore")
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=190)

    completed_jobs = [job for job in jobs if job["狀態"] == STATUS_COMPLETED and job.get("影片")]
    if completed_jobs:
        with st.expander("查看已完成推論影格", expanded=False):
            selected_label = st.selectbox(
                "選擇推論任務",
                [f'{job["chicken ID"]} - {job["影片"]}' for job in completed_jobs],
                key="completed_inference_preview_select",
                label_visibility="collapsed",
            )
            selected_index = [f'{job["chicken ID"]} - {job["影片"]}' for job in completed_jobs].index(selected_label)
            selected_video = Path(str(completed_jobs[selected_index]["影片"])).stem
            show_inference_sample_frames(video_ids=[selected_video])

    if has_active_jobs():
        st.info("背景仍有任務執行中，可繼續掃描下一隻 chicken ID。")


def default_arduino_port() -> str:
    existing_port = str(st.session_state.get("arduino_serial_port", "") or "").strip()
    if existing_port:
        return existing_port
    ports = list_serial_ports()
    for port in ports:
        if "leonardo" in port.label.lower():
            return port.device
    return ports[0].device if ports else ""


def default_arduino_port_index(port_labels: list[str], default_port: str) -> int:
    for index, label in enumerate(port_labels):
        if label.startswith(f"{default_port} ") or label == default_port:
            return index
    for index, label in enumerate(port_labels):
        if "leonardo" in label.lower():
            return index
    return 0


def default_scale_port() -> str:
    existing_port = str(st.session_state.get("scale_serial_port", "") or "").strip()
    if existing_port:
        return existing_port
    ports = list_serial_ports()
    preferred_keywords = ("ft232", "rs485", "usb serial", "uart", "serial")
    for port in ports:
        label = port.label.lower()
        if any(keyword in label for keyword in preferred_keywords) and "leonardo" not in label:
            return port.device
    return ""


def default_scale_port_index(port_labels: list[str], default_port: str) -> int:
    for index, label in enumerate(port_labels):
        if label.startswith(f"{default_port} ") or label == default_port:
            return index
    preferred_keywords = ("ft232", "rs485", "usb serial", "uart", "serial")
    for index, label in enumerate(port_labels):
        lowered = label.lower()
        if any(keyword in lowered for keyword in preferred_keywords) and "leonardo" not in lowered:
            return index
    return 0


def grade_with_latest_database_thresholds(chicken: dict) -> tuple[str | None, str | None, str]:
    config = get_latest_grading_config()
    if not config:
        return None, None, "目前資料庫沒有已儲存的分級門檻，請先在「分級門檻設定」執行一次分級。"

    selected_features, pass_thresholds, standby_thresholds = config
    validate_grading_config(selected_features, pass_thresholds, standby_thresholds)
    grade_result, light_result = grade_chicken(chicken, selected_features, pass_thresholds, standby_thresholds)
    update_grading_result(
        chicken["chicken_id"],
        selected_features,
        pass_thresholds,
        standby_thresholds,
        grade_result,
        light_result,
    )
    return grade_result, light_result, ""


def render_current_thresholds() -> None:
    config = get_latest_grading_config()
    if not config:
        st.info("目前尚未儲存分級門檻。請設定門檻後按「確認設定門檻」。")
        return

    selected_features, pass_thresholds, standby_thresholds = config
    cards = []
    for feature, label in FEATURE_COLUMNS.items():
        enabled = feature in selected_features
        disabled_class = "" if enabled else " threshold-summary-disabled"
        enabled_label = "啟用" if enabled else "停用"
        cards.append(
            f'<div class="threshold-summary-card{disabled_class}">'
            f'<div class="threshold-summary-label">{label} · {enabled_label}</div>'
            f'<div class="threshold-summary-value">'
            f'通過 {number_input_value(pass_thresholds.get(feature)):.3f}'
            f'&nbsp;／&nbsp; 備用 {number_input_value(standby_thresholds.get(feature)):.3f}'
            f'</div></div>'
        )
    st.markdown(
        f'<div class="threshold-summary-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def hardware_settings_page() -> None:
    page_heading(
        "硬體連接設定",
        "掃碼器、Arduino 燈號與 RS485/USB 秤重設備可在同一畫面完成設定與測試。",
    )

    scanner_col, light_col, scale_col = st.columns(3, vertical_alignment="top")

    with scanner_col:
        with st.container(border=True):
            st.subheader("掃碼器")
            st.caption("USB HID 模式等同鍵盤輸入，不需要指定 COM。")
            st.text_input(
                "掃碼器輸入測試",
                placeholder="C001",
                help="進入此頁後可直接掃描；掃描槍送出 Enter 後會自動記錄最後掃描內容。",
                key="scanner_test_input",
                on_change=commit_scanner_input,
                args=("scanner_test_input", "scanner_last_value"),
            )
            install_n_scanner_focus_hotkey()
            install_scanner_input_autofocus()
            if st.session_state.get("scanner_last_value"):
                st.success(f"最後掃描：{st.session_state.scanner_last_value}")
            else:
                st.info("等待掃描輸入")

    with light_col:
        with st.container(border=True):
            st.subheader("燈號 COM")
            hardware_enabled = st.checkbox(
                "啟用 Arduino 實體燈號",
                value=bool(st.session_state.get("arduino_light_enabled", True)),
                key="arduino_light_enabled",
            )
            ports = list_serial_ports()
            port_labels = [port.label for port in ports]
            port_devices = {port.label: port.device for port in ports}
            default_port = st.session_state.get("arduino_serial_port", "")
            if ports:
                default_index = default_arduino_port_index(port_labels, default_port)
                selected_label = st.selectbox(
                    "Arduino serial port",
                    port_labels,
                    index=default_index,
                    key="hardware_arduino_serial_label",
                )
                arduino_port = port_devices[selected_label]
            else:
                arduino_port = st.text_input(
                    "Arduino serial port",
                    value=default_port,
                    placeholder="例如 COM5 或 /dev/ttyACM0",
                    key="hardware_arduino_serial_port_input",
                ).strip()
            st.session_state.arduino_serial_port = arduino_port

            test_col1, test_col2 = st.columns(2)
            if test_col1.button("測試燈號", use_container_width=True, key="hardware_test_light"):
                if not arduino_port:
                    st.warning("請先選擇 Arduino serial port。")
                else:
                    try:
                        replies = send_arduino_commands(arduino_port, ["TEST"])
                        st.success("已送出 TEST 指令。" + (f" 回覆：{', '.join(replies)}" if replies else ""))
                    except Exception as exc:
                        st.error(f"Arduino 連線失敗：{exc}")
            if test_col2.button("關閉燈號", use_container_width=True, key="hardware_light_off"):
                if not arduino_port:
                    st.warning("請先選擇 Arduino serial port。")
                else:
                    try:
                        replies = send_arduino_commands(arduino_port, ["LIGHT:OFF"])
                        st.success("已送出 LIGHT:OFF 指令。" + (f" 回覆：{', '.join(replies)}" if replies else ""))
                    except Exception as exc:
                        st.error(f"Arduino 連線失敗：{exc}")
            if st.button("重新連線 Arduino", use_container_width=True, key="hardware_reconnect_arduino"):
                if not arduino_port:
                    st.warning("請先選擇 Arduino serial port。")
                else:
                    try:
                        replies = reconnect_arduino(arduino_port)
                        st.success("Arduino 重新連線成功。" + (f" 回覆：{', '.join(replies)}" if replies else ""))
                    except Exception as exc:
                        st.error(f"Arduino 重新連線失敗：{exc}")

            if hardware_enabled and arduino_port:
                st.info(f"目前連接：{arduino_port}")

    with scale_col:
        with st.container(border=True):
            st.subheader("秤重設備")
            st.caption(f"RS485 轉 USB · {SCALE_BAUD_RATE} bps · STX/ASCII/ETX")
            ports = list_serial_ports()
            port_labels = [port.label for port in ports]
            port_devices = {port.label: port.device for port in ports}
            default_port = st.session_state.get("scale_serial_port", "")
            if ports:
                default_index = default_scale_port_index(port_labels, default_port)
                selected_label = st.selectbox(
                    "秤重設備 serial port",
                    port_labels,
                    index=default_index,
                    key="hardware_scale_serial_label",
                )
                scale_port = port_devices[selected_label]
            else:
                scale_port = st.text_input(
                    "秤重設備 serial port",
                    value=default_port,
                    placeholder="例如 COM6 或 /dev/ttyUSB0",
                    key="hardware_scale_serial_port_input",
                ).strip()
            st.session_state.scale_serial_port = scale_port

            scale_test_col1, scale_test_col2 = st.columns(2)
            if scale_test_col1.button("測試讀值", use_container_width=True, key="hardware_test_scale"):
                if not scale_port:
                    st.warning("請先選擇秤重設備 serial port。")
                else:
                    try:
                        result = read_scale_average(scale_port, duration_seconds=2.0)
                        if result.average_g is None:
                            detail = f" 原始封包：{', '.join(result.raw_packets)}" if result.raw_packets else ""
                            errors = f" 錯誤：{'; '.join(result.errors)}" if result.errors else ""
                            st.warning(f"2 秒內沒有取得有效重量。{detail}{errors}")
                        else:
                            st.success(
                                f"平均 {result.average_g:.1f} g · 最新 {result.latest_g} g · {len(result.samples)} 筆"
                            )
                            if result.raw_packets:
                                st.caption(f"原始封包：{', '.join(result.raw_packets[-5:])}")
                            if result.errors:
                                st.warning("；".join(result.errors))
                    except Exception as exc:
                        st.error(f"秤重設備連線失敗：{exc}")
            if scale_test_col2.button("清除紀錄", use_container_width=True, key="hardware_clear_scale_history"):
                st.session_state.scale_recent_samples = []
                st.success("已清除秤重即時紀錄。")

            if scale_port:
                st.info(f"目前連接：{scale_port}")


def commit_scanner_input(input_key: str, confirmed_key: str) -> None:
    scanned_id = str(st.session_state.get(input_key, "") or "").strip()
    if scanned_id:
        st.session_state[confirmed_key] = scanned_id
        st.session_state[input_key] = ""


def resolve_qrcode_input(
    key_prefix: str,
    clear_on_submit: bool = False,
    auto_focus: bool = False,
    scanner_mode: bool = False,
    show_confirmation: bool = True,
) -> str:
    install_n_scanner_focus_hotkey()
    confirmed_key = f"{key_prefix}_confirmed_manual_id"
    input_version = int(st.session_state.get(f"{key_prefix}_input_version", 0))
    input_key = f"{key_prefix}_manual_id_{input_version}"

    if scanner_mode:
        st.session_state.setdefault(input_key, "")
        st.text_input(
            "USB 掃描槍掃描 / 手動輸入 chicken ID",
            placeholder="C001",
            help="此欄位會自動聚焦；掃描槍送出 Enter 後會直接確認 chicken ID。",
            key=input_key,
            on_change=commit_scanner_input,
            args=(input_key, confirmed_key),
        )
        if auto_focus:
            install_scanner_input_autofocus()
        confirmed_id = st.session_state.get(confirmed_key, "").strip()
        if confirmed_id and show_confirmation:
            st.success(f"已確認 chicken ID：{confirmed_id}")
        return confirmed_id

    with st.form(f"{key_prefix}_manual_id_form_{input_version}", clear_on_submit=clear_on_submit):
        manual_id = st.text_input(
            "USB 掃描槍掃描 / 手動輸入 chicken ID",
            placeholder="C001",
            help="按第一顆實體按鈕後，使用 USB 掃描槍掃描 chicken ID；掃描槍送出 Enter 或按「確定」即可確認。",
            key=input_key,
        )
        manual_confirmed = st.form_submit_button("確定", use_container_width=True)
    if auto_focus:
        install_scanner_input_autofocus()
    if manual_confirmed:
        st.session_state[confirmed_key] = manual_id.strip()

    confirmed_id = st.session_state.get(confirmed_key, "").strip()
    if confirmed_id and show_confirmation:
        st.success(f"已確認 chicken ID：{confirmed_id}")
    return confirmed_id


def calibration_display_image(image_path: Path, points: list[tuple[int, int]]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    radius = 7
    for index, (x, y) in enumerate(points, start=1):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
        draw.text((x + radius + 4, y - radius), str(index), fill="red")
    return image


def camera_adjustment_page() -> None:
    page_heading("相機調整", "確認雞隻側面、雞冠與腳脛皆完整進入畫面。")
    st.camera_input(
        "相機即時畫面",
        key="camera_adjustment_preview",
        label_visibility="collapsed",
    )


@st.fragment
def calibration_point_picker(image_path: Path, real_length_cm: float) -> None:
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size
    click_key_version = int(st.session_state.get("calibration_click_key_version", 0))
    points = st.session_state.calibration_points

    status_col, confirm_col, clear_col = st.columns([3, 1, 1], vertical_alignment="bottom")
    status_col.info(f"已選取 {len(points)}/2 點：{points or '請依序點選比例尺兩端'}")
    if len(points) == 2:
        install_n_click_button_hotkey("\u78ba\u5b9a\u6821\u6b63")
        (x1, y1), (x2, y2) = points
        pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if pixel_distance > 0:
            cm_per_pixel = float(real_length_cm) / pixel_distance
            if confirm_col.button("\u78ba\u5b9a\u6821\u6b63", type="primary", use_container_width=True):
                st.session_state.calibration_cm_per_pixel = cm_per_pixel
                st.session_state.phenotype_cm_per_pixel = cm_per_pixel
                st.session_state.phenotype_cm_per_pixel_version = int(
                    st.session_state.get("phenotype_cm_per_pixel_version", 0)
                ) + 1
                st.success(f"\u6821\u6b63\u5df2\u5957\u7528\uff1acm_per_pixel = {cm_per_pixel:.6f}")
        else:
            status_col.error("\u5169\u500b\u6821\u6b63\u9ede\u8ddd\u96e2\u70ba 0\uff0c\u8acb\u6e05\u9664\u5f8c\u91cd\u65b0\u9ede\u9078\u3002")
    else:
        confirm_col.button("\u78ba\u5b9a\u6821\u6b63", disabled=True, use_container_width=True)

    if clear_col.button("\u6e05\u9664\u6821\u6b63\u9ede", use_container_width=True):
        st.session_state.calibration_points = []
        st.session_state.calibration_last_click_unix_time = None
        st.session_state.calibration_click_key_version = click_key_version + 1
        st.rerun(scope="fragment")

    display_image = calibration_display_image(image_path, st.session_state.calibration_points)
    display_image = display_image.resize((CAMERA_PREVIEW_WIDTH_PX, CAMERA_PREVIEW_HEIGHT_PX))
    click_value = streamlit_image_coordinates(
        display_image,
        height=CAMERA_PREVIEW_HEIGHT_PX,
        width=CAMERA_PREVIEW_WIDTH_PX,
        key=f"calibration_click_{image_path.name}_{click_key_version}",
        image_format="JPEG",
        jpeg_quality=82,
        cursor="crosshair",
    )

    if click_value is not None:
        click_time = click_value.get("unix_time")
        if click_time != st.session_state.get("calibration_last_click_unix_time"):
            st.session_state.calibration_last_click_unix_time = click_time
            clicked_width = max(1, int(click_value.get("width") or CAMERA_PREVIEW_WIDTH_PX))
            clicked_height = max(1, int(click_value.get("height") or CAMERA_PREVIEW_HEIGHT_PX))
            point = (
                min(original_width - 1, max(0, int(round(click_value["x"] * original_width / clicked_width)))),
                min(original_height - 1, max(0, int(round(click_value["y"] * original_height / clicked_height)))),
            )
            if len(st.session_state.calibration_points) < 2:
                st.session_state.calibration_points.append(point)
                st.rerun(scope="fragment")

def scale_calibration_section() -> None:
    page_heading("比例尺校正", "拍攝比例尺照片，輸入真實長度並依序點選兩端。")
    install_n_click_button_hotkey("拍攝校正照片")

    st.session_state.setdefault("calibration_points", [])
    st.session_state.setdefault("calibration_cm_per_pixel", 0.01)
    st.session_state.setdefault("calibration_click_key_version", 0)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
    calibration_camera_index = ctrl_col1.number_input(
        "校正 camera ID",
        min_value=0,
        max_value=10,
        value=int(st.session_state.get("calibration_camera_index", 0)),
        step=1,
        key="calibration_camera_index",
    )
    real_length_cm = ctrl_col2.number_input(
        "比例尺真實長度 (cm)",
        min_value=0.001,
        value=float(st.session_state.get("calibration_real_length_cm", 10.0)),
        step=0.5,
        format="%.3f",
        key="calibration_real_length_cm",
    )
    ctrl_col3.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
    if ctrl_col3.button("拍攝校正照片", use_container_width=True):
        try:
            image_path = capture_calibration_image(int(calibration_camera_index))
            st.session_state.calibration_image_path = str(image_path)
            st.session_state.calibration_points = []
            st.session_state.calibration_last_click_unix_time = None
            st.session_state.calibration_click_key_version = 0
            st.success(f"已拍攝校正照片：{image_path.name}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get("calibration_image_path"):
        image_path = Path(st.session_state.calibration_image_path)
        if image_path.exists():
            calibration_point_picker(image_path, float(real_length_cm))
        else:
            st.warning("找不到校正照片，請重新拍攝。")


@st.fragment(run_every=0.5)
def live_scale_panel(scale_port: str, window_seconds: float = 2.0) -> None:
    st.subheader("即時秤重")
    if not scale_port:
        st.info("尚未設定秤重設備 serial port，請先到「硬體連接設定」選擇秤重設備。")
        st.session_state.scale_recent_samples = []
        return

    history = st.session_state.get("scale_recent_samples", [])
    try:
        result = read_scale_samples(scale_port, duration_seconds=0.25)
        history = append_recent_scale_samples(history, result.samples, window_seconds=window_seconds)
        st.session_state.scale_recent_samples = history
        average_g = average_recent_samples(history)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("最新重量 (g)", "-" if result.latest_g is None else f"{result.latest_g}")
        metric_col2.metric(f"{window_seconds:.0f} 秒平均 (g)", "-" if average_g is None else f"{average_g:.1f}")
        metric_col3.metric("有效樣本數", str(len(history)))
        if result.raw_packets:
            st.caption(f"最近封包：{', '.join(result.raw_packets[-3:])}")
        if result.errors:
            st.warning("；".join(result.errors))
    except Exception as exc:
        st.session_state.scale_recent_samples = history
        st.error(f"秤重設備讀取失敗：{exc}")


def phenotype_entry_page() -> None:
    page_heading(
        "表型資料輸入",
        "掃描 chicken ID、即時秤重並建立背景推論任務；前一隻推論時可繼續掃描下一隻。",
    )
    install_r_submit_hotkey("計算雞冠及腳脛數據")

    yolo_seg_errors = validate_yolo_seg_assets()
    if yolo_seg_errors:
        for error in yolo_seg_errors:
            st.error(error)
        return

    if st.session_state.get("phenotype_last_enqueued_job"):
        st.success(f"已建立背景推論任務：{st.session_state.phenotype_last_enqueued_job}")
        st.session_state.phenotype_last_enqueued_job = ""

    cm_per_pixel_value = float(
        st.session_state.get(
            "phenotype_cm_per_pixel",
            st.session_state.get("calibration_cm_per_pixel", 0.01),
        )
    )
    setting_col1, setting_col2, setting_col3 = st.columns(3)
    camera_index = setting_col1.number_input(
        "camera ID",
        min_value=0,
        max_value=10,
        value=int(st.session_state.get("phenotype_camera_id", 0)),
        step=1,
        key="phenotype_camera_id",
    )
    confidence_threshold = setting_col2.number_input(
        "YOLO confidence 門檻",
        min_value=0.01,
        max_value=1.0,
        value=float(st.session_state.get("phenotype_confidence_threshold", 0.5)),
        step=0.05,
        key="phenotype_confidence_threshold",
    )
    setting_col3.text_input(
        "目前 cm_per_pixel",
        value=f"{cm_per_pixel_value:.6f}",
        disabled=True,
    )

    query_id = resolve_qrcode_input(
        "phenotype_entry",
        auto_focus=True,
        scanner_mode=True,
        show_confirmation=False,
    )
    if not query_id:
        st.info("請掃描 QR code，或手動輸入 chicken ID 後按 Enter。")
        render_inference_queue_panel()
        return
    if st.session_state.get("scale_active_chicken_id") != query_id:
        st.session_state.scale_active_chicken_id = query_id
        st.session_state.scale_recent_samples = []

    selected = get_chicken(query_id)
    if not selected:
        st.info(f"資料庫尚無 chicken ID：{query_id}，儲存後會新增此雞隻。")

    scale_port = default_scale_port()
    measurement_col, queue_col = st.columns([2, 3], vertical_alignment="top")

    with measurement_col:
        live_scale_panel(scale_port, window_seconds=2.0)

        with st.form("phenotype_entry_form"):
            st.caption(f"目前 chicken ID：{query_id}")
            col1, col2 = st.columns(2)
            manual_weight_g = col1.number_input(
                "手動重量備援 (g)",
                min_value=0.0,
                value=number_input_value(selected["weight_g"]) if selected else 0.0,
                step=10.0,
                help="正常流程會讀取秤重設備 2 秒平均；讀不到有效重量時才使用此欄位。",
            )
            cm_per_pixel = col2.number_input(
                "cm_per_pixel",
                min_value=0.000001,
                value=cm_per_pixel_value,
                step=0.001,
                format="%.6f",
                key=f"phenotype_cm_per_pixel_input_{int(st.session_state.get('phenotype_cm_per_pixel_version', 0))}",
            )
            st.caption("錄影 1 秒 · 固定推論 10 張照片")
            submitted = st.form_submit_button("計算雞冠及腳脛數據", type="primary", use_container_width=True)

    with queue_col:
        completed_job = latest_completed_job_for_chicken(query_id)
        if completed_job and completed_job.result_row:
            st.subheader("最近完成結果")
            st.dataframe(
                pd.DataFrame([completed_job.result_row]),
                use_container_width=True,
                hide_index=True,
                height=105,
            )
            if completed_job.video_id:
                with st.expander("查看最近推論影格", expanded=False):
                    show_inference_sample_frames(video_ids=[completed_job.video_id])

        render_inference_queue_panel()

    if submitted:
        try:
            scale_result = None
            final_weight_g = float(manual_weight_g)
            if scale_port:
                scale_result = read_scale_average(scale_port, duration_seconds=2.0)
                if scale_result.average_g is not None:
                    final_weight_g = float(scale_result.average_g)
                    st.session_state.scale_recent_samples = []
                elif scale_result.errors:
                    st.warning("秤重設備未取得有效重量，改用手動重量備援。錯誤：" + "；".join(scale_result.errors))
                else:
                    st.warning("秤重設備 2 秒內沒有取得有效重量，改用手動重量備援。")
            else:
                st.warning("尚未設定秤重設備 serial port，改用手動重量備援。")

            st.session_state.phenotype_cm_per_pixel = float(cm_per_pixel)
            if selected:
                update_chicken(
                    query_id,
                    final_weight_g,
                    selected["comb_area_cm2"],
                    selected.get("shank_width_cm", 0.0),
                    selected.get("shank_length_cm", 0.0),
                )
            else:
                add_chicken(query_id, final_weight_g, 0.0, 0.0, 0.0)

            job_id = enqueue_inference_job(
                chicken_id=query_id,
                weight_g=float(final_weight_g),
                camera_index=int(camera_index),
                confidence_threshold=float(confidence_threshold),
                frame_stride=1,
                cm_per_pixel=float(cm_per_pixel),
                hardware_enabled=bool(st.session_state.get("arduino_light_enabled", True)),
                arduino_port=default_arduino_port(),
                target_photo_count=10,
            )
            st.session_state.phenotype_last_enqueued_job = job_id
            st.session_state.phenotype_entry_confirmed_manual_id = ""
            st.session_state.phenotype_entry_input_version = int(
                st.session_state.get("phenotype_entry_input_version", 0)
            ) + 1
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

def data_management_page() -> None:
    page_heading("雞隻資料管理", "集中檢視所有表型資料；新增、修改與刪除操作收納於下方。")

    chickens = get_all_chickens()
    total_col, sample_col, import_col, export_col = st.columns([0.8, 1, 1.25, 1])
    total_col.metric("資料筆數", len(chickens))
    with sample_col:
        if st.button("建立 20 筆範例資料", use_container_width=True):
            inserted = create_sample_data()
            st.success(f"已新增 {inserted} 筆範例資料。")
            st.rerun()
    with import_col:
        csv_file = st.file_uploader("匯入 CSV", type=["csv"], label_visibility="collapsed")
        if csv_file is not None:
            try:
                inserted, updated = import_csv(csv_file)
                st.success(f"CSV 匯入完成：新增 {inserted} 筆，更新 {updated} 筆。")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with export_col:
        st.download_button(
            "匯出 CSV",
            export_csv_bytes(),
            file_name="chicken_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    show_data_table(hide_result_columns=True, height=330)

    with st.expander("新增雞隻", expanded=False):
        with st.form("add_chicken_form", clear_on_submit=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            chicken_id = col1.text_input("chicken ID", placeholder="C001")
            weight_g = col2.number_input("體重 (g)", min_value=0.0, step=10.0)
            comb_area_cm2 = col3.number_input("雞冠面積 (cm²)", min_value=0.0, value=0.0, step=0.5)
            shank_width_cm = col4.number_input("腳脛寬度 (cm)", min_value=0.0, value=0.0, step=0.1)
            shank_length_cm = col5.number_input("腳脛長度 (cm)", min_value=0.0, value=0.0, step=0.1)
            submitted = st.form_submit_button("新增資料", use_container_width=True)
            if submitted:
                try:
                    add_chicken(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)
                    st.success(f"已新增雞隻：{chicken_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with st.expander("修改或刪除雞隻", expanded=False):
        if chickens:
            chicken_ids = [row["chicken_id"] for row in chickens]
            selected_id = st.selectbox("選擇 chicken ID", chicken_ids, key="manage_chicken_id")
            selected = get_chicken(selected_id)
            if selected:
                with st.form("edit_chicken_form"):
                    col1, col2, col3, col4 = st.columns(4)
                    edit_weight = col1.number_input("體重 (g)", value=number_input_value(selected["weight_g"]), step=10.0)
                    edit_comb = col2.number_input("雞冠面積 (cm²)", value=number_input_value(selected["comb_area_cm2"]), step=0.5)
                    edit_shank = col3.number_input(
                        "腳脛寬度 (cm)",
                        value=number_input_value(selected.get("shank_width_cm")),
                        step=0.1,
                    )
                    edit_shank_length = col4.number_input(
                        "腳脛長度 (cm)",
                        value=number_input_value(selected.get("shank_length_cm")),
                        step=0.1,
                    )
                    save_col, delete_col = st.columns(2)
                    save = save_col.form_submit_button("儲存修改", use_container_width=True)
                    remove = delete_col.form_submit_button("刪除此雞隻", use_container_width=True)
                    if save:
                        update_chicken(selected_id, edit_weight, edit_comb, edit_shank, edit_shank_length)
                        st.success(f"已更新雞隻：{selected_id}")
                        st.rerun()
                    if remove:
                        delete_chicken(selected_id)
                        st.warning(f"已刪除雞隻：{selected_id}")
                        st.rerun()
        else:
            st.info("目前沒有可修改的雞隻資料。")


def grading_page() -> None:
    page_heading(
        "分級門檻設定",
        "設定通過與備用門檻後，系統會更新所有雞隻的通過、備用及不通過結果。",
    )
    render_current_thresholds()

    pass_thresholds = {}
    standby_thresholds = {}
    feature_items = list(FEATURE_COLUMNS.items())
    pass_section, standby_section = st.columns(2, vertical_alignment="top")
    with pass_section:
        with st.container(border=True, key="pass_threshold_block"):
            st.markdown("<h3 style='text-align: center; margin:0 0 .35rem;'>通過門檻</h3>", unsafe_allow_html=True)
            for row_start in range(0, len(feature_items), 2):
                row_cols = st.columns(2)
                for col, (feature, label) in zip(row_cols, feature_items[row_start : row_start + 2]):
                    pass_thresholds[feature] = col.number_input(
                        label,
                        min_value=0.0,
                        step=0.1,
                        key=f"{feature}_pass_threshold",
                    )

    with standby_section:
        with st.container(border=True, key="standby_threshold_block"):
            st.markdown("<h3 style='text-align: center; margin:0 0 .35rem;'>備用門檻</h3>", unsafe_allow_html=True)
            for row_start in range(0, len(feature_items), 2):
                row_cols = st.columns(2)
                for col, (feature, label) in zip(row_cols, feature_items[row_start : row_start + 2]):
                    standby_thresholds[feature] = col.number_input(
                        label,
                        min_value=0.0,
                        step=0.1,
                        key=f"{feature}_standby_threshold",
                    )

    if st.button("確認設定門檻", type="primary", use_container_width=True):
        try:
            selected_features = list(FEATURE_COLUMNS.keys())
            validate_grading_config(selected_features, pass_thresholds, standby_thresholds)
            enabled_features = active_features(selected_features, pass_thresholds, standby_thresholds)
            rows = []
            for chicken in get_all_chickens():
                grade_result, light_result = grade_chicken(
                    chicken,
                    selected_features,
                    pass_thresholds,
                    standby_thresholds,
                )
                update_grading_result(
                    chicken["chicken_id"],
                    selected_features,
                    pass_thresholds,
                    standby_thresholds,
                    grade_result,
                    light_result,
                )
                below_pass_features = []
                below_standby_features = []
                for feature in enabled_features:
                    value = chicken.get(feature)
                    if value is None or value == "":
                        below_pass_features.append(feature)
                        below_standby_features.append(feature)
                    else:
                        if float(value) < float(pass_thresholds[feature]):
                            below_pass_features.append(feature)
                        if float(value) < float(standby_thresholds[feature]):
                            below_standby_features.append(feature)
                rows.append(
                    {
                        "chicken ID": chicken["chicken_id"],
                        "grade_result": grade_result,
                        "light_result": light_result,
                        "below_pass_features": ", ".join(below_pass_features),
                        "below_standby_features": ", ".join(below_standby_features),
                    }
                )
            st.success("分級完成，已更新資料庫。")
            summary = summarize_results(rows)
            metric_cols = st.columns(7)
            labels = [
                ("總筆數", summary["total"]),
                ("通過筆數", summary["pass_count"]),
                ("備用筆數", summary["standby_count"]),
                ("不通過筆數", summary["fail_count"]),
                ("通過比例", f"{summary['pass_rate']:.1f}%"),
                ("備用比例", f"{summary['standby_rate']:.1f}%"),
                ("不通過比例", f"{summary['fail_rate']:.1f}%"),
            ]
            for col, (label, val) in zip(metric_cols, labels):
                col.markdown(
                    f"<div style='text-align:center'><div style='font-weight:600'>{label}</div><div style='font-size:20px'>{val}</div></div>",
                    unsafe_allow_html=True,
                )
            with st.expander("查看本次分級明細", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=260)
        except Exception as exc:
            st.error(str(exc))

    with st.expander("查看目前雞隻資料", expanded=False):
        show_data_table(height=320)


def render_light(status: str, message: str) -> None:
    css_class = {
        GRADE_PASS: "light-green",
        GRADE_STANDBY: "light-yellow",
        GRADE_FAIL: "light-red",
    }.get(status, "light-gray")
    st.markdown(
        f"""
        <div class="light-panel">
            <div class="light-circle {css_class}"></div>
            <div>
                <div class="light-text">{message}</div>
                <div class="status-note">判定結果：{status}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def schedule_arduino_light_off(arduino_port: str) -> None:
    previous_timer = st.session_state.get("final_light_off_timer")
    if previous_timer and previous_timer.is_alive():
        previous_timer.cancel()

    def turn_off() -> None:
        try:
            send_arduino_commands(arduino_port, ["LIGHT:OFF"], delay_after_open=0.02)
        except Exception:
            pass

    timer = threading.Timer(FINAL_LIGHT_SECONDS, turn_off)
    timer.daemon = True
    timer.start()
    st.session_state.final_light_off_timer = timer


def execute_final_light_decision(query_id: str) -> bool:
    chicken = get_chicken(query_id)
    if not chicken:
        st.warning("找不到此 chicken ID。")
        return False
    try:
        grade_result, light_result, grade_warning = grade_with_latest_database_thresholds(chicken)
        if grade_warning:
            st.warning(grade_warning)
            grade_result = chicken["grade_result"]
        else:
            st.success("已依目前資料庫分級門檻更新最終篩選結果。")
    except Exception as exc:
        st.error(f"最終篩選失敗：{exc}")
        return False

    if grade_result == GRADE_PASS:
        render_light(GRADE_PASS, "通過 / 綠燈")
    elif grade_result == GRADE_STANDBY:
        render_light(GRADE_STANDBY, "備用 / 黃燈")
    elif grade_result == GRADE_FAIL:
        render_light(GRADE_FAIL, "不通過 / 紅燈")
    else:
        render_light("UNDEFINED", "尚未完成分級")

    if bool(st.session_state.get("arduino_light_enabled", True)):
        arduino_port = default_arduino_port()
        if not arduino_port:
            st.warning("已啟用實體燈號，但尚未選擇 Arduino serial port。")
        else:
            try:
                commands = commands_for_grade(grade_result)
                timed_commands = [
                    command.replace("LIGHT:", "LIGHT_FOR:", 1) + f":{int(FINAL_LIGHT_SECONDS * 1000)}"
                    if command.startswith("LIGHT:") and command != "LIGHT:OFF"
                    else command
                    for command in commands
                ]
                replies = send_arduino_commands(arduino_port, timed_commands)
                if any(reply.startswith("ERR:UNKNOWN:LIGHT_FOR") for reply in replies):
                    legacy_light_commands = [command for command in commands if command.startswith("LIGHT:")]
                    replies = send_arduino_commands(arduino_port, legacy_light_commands)
                    schedule_arduino_light_off(arduino_port)
                st.success(
                    f"已送出實體燈號指令：{', '.join(commands)}，將亮燈 {FINAL_LIGHT_SECONDS:.0f} 秒後自動關閉"
                    + (f"；Arduino 回覆：{', '.join(replies)}" if replies else "")
                )
            except Exception as exc:
                st.error(f"Arduino 燈號控制失敗：{exc}")

    return True


def render_single_chicken_row(query_id: str) -> None:
    st.dataframe(
        get_all_chickens_df().query("chicken_id == @query_id").rename(columns={"chicken_id": "chicken ID"}),
        use_container_width=True,
        hide_index=True,
        height=105,
    )


def final_scan_page() -> None:
    page_heading(
        "再次掃描與亮燈判定",
        "再次掃描既有 QR code，立即顯示最終分級並控制綠燈、黃燈或紅燈。",
    )

    query_id = resolve_qrcode_input("final_scan", auto_focus=True, scanner_mode=True)
    if not query_id:
        st.info("請輸入 chicken ID，或掃描既有 QR code。")
        st.session_state.final_scan_last_processed_token = ""
        last_id = st.session_state.get("final_scan_last_completed_id")
        if last_id:
            st.success(f"上一筆已完成第二階段掃描：{last_id}")
            render_single_chicken_row(last_id)
        return

    st.success(f"已選擇 chicken ID：{query_id}")
    final_scan_token = f"{query_id}|{repr(get_latest_grading_config())}"
    if st.session_state.get("final_scan_last_processed_token") != final_scan_token:
        if execute_final_light_decision(query_id):
            scanned_at = mark_final_scanned(query_id)
            st.success(f"已記錄第二階段掃描完成：{query_id}，時間：{scanned_at}")
            render_single_chicken_row(query_id)
            st.session_state.final_scan_last_completed_id = query_id
            st.session_state.final_scan_confirmed_manual_id = ""
            st.session_state.final_scan_input_version = int(st.session_state.get("final_scan_input_version", 0)) + 1
            st.session_state.final_scan_last_processed_token = ""
    else:
        chicken = get_chicken(query_id)
        if chicken:
            grade_result = chicken.get("grade_result")
            if grade_result == GRADE_PASS:
                render_light(GRADE_PASS, "通過 / 綠燈")
            elif grade_result == GRADE_STANDBY:
                render_light(GRADE_STANDBY, "備用 / 黃燈")
            elif grade_result == GRADE_FAIL:
                render_light(GRADE_FAIL, "不通過 / 紅燈")
            else:
                render_light("UNDEFINED", "尚未完成分級")
            render_single_chicken_row(query_id)


def main() -> None:
    st.markdown(
        """
        <div class="app-masthead">
            <div class="app-title">紅羽土雞表型分級系統</div>
            <div class="app-subtitle">掃描識別 · 表型量測 · 分級亮燈</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = {
        "硬體連接設定": hardware_settings_page,
        "雞隻資料管理": data_management_page,
        "分級門檻設定": grading_page,
        "相機調整": camera_adjustment_page,
        "比例尺校正": scale_calibration_section,
        "掃描 QR code 與表型輸入": phenotype_entry_page,
        "再次掃描與亮燈判定": final_scan_page,
    }
    page_names = list(pages.keys())
    if "main_page_navigation" not in st.session_state:
        st.session_state.main_page_navigation = page_names[0]
    install_p_page_cycle_hotkey(page_names, st.session_state.main_page_navigation)
    selected_page = st.segmented_control(
        "功能選擇",
        page_names,
        selection_mode="single",
        default=st.session_state.main_page_navigation,
        label_visibility="collapsed",
        key="main_page_navigation_segmented",
    )
    if selected_page is None:
        selected_page = st.session_state.main_page_navigation
    st.session_state.main_page_navigation = selected_page
    pages[selected_page]()


if __name__ == "__main__":
    main()


