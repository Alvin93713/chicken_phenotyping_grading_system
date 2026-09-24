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


CAMERA_PREVIEW_WIDTH_PX = 960
CAMERA_PREVIEW_HEIGHT_PX = 600
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
.light-panel {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 24px 0;
}
.light-circle {
    width: 156px;
    height: 156px;
    border-radius: 50%;
    border: 8px solid rgba(0, 0, 0, 0.12);
    box-shadow: inset 0 10px 26px rgba(255,255,255,0.55), 0 10px 26px rgba(0,0,0,0.14);
}
.light-green { background: #16a34a; }
.light-yellow { background: #facc15; }
.light-red { background: #dc2626; }
.light-gray { background: #9ca3af; }
.light-text {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.35;
}
.status-note {
    color: #4b5563;
    font-size: 16px;
    margin-top: 8px;
}
[data-testid="stSegmentedControl"] button {
    font-size: 1.5rem;
    min-height: 3rem;
    padding: 0.5rem 1rem;
}
[data-testid="stCameraInput"] {
    width: 100% !important;
    max-width: 960px !important;
    aspect-ratio: 16 / 10;
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
    aspect-ratio: 16 / 10;
}
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img,
[data-testid="stCameraInput"] canvas {
    width: 100% !important;
    height: auto !important;
    max-height: min(600px, calc((100vw - 3rem) / 1.6));
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
    padding: 16px;
    margin-bottom: 18px;
}
.st-key-standby_threshold_block {
    border: 4px solid #facc15;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 18px;
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def show_data_table(hide_result_columns: bool = False) -> None:
    df = get_all_chickens_df().rename(columns={"chicken_id": "chicken ID"})
    if hide_result_columns:
        df = df.drop(columns=["grade_result", "light_result"], errors="ignore")
    st.dataframe(df, use_container_width=True, hide_index=True)


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
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    completed_jobs = [job for job in jobs if job["狀態"] == STATUS_COMPLETED and job.get("影片")]
    if completed_jobs:
        selected_label = st.selectbox(
            "查看已完成推論影格",
            [f'{job["chicken ID"]} - {job["影片"]}' for job in completed_jobs],
            key="completed_inference_preview_select",
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
    st.subheader("目前門檻")
    if not config:
        st.info("目前尚未儲存分級門檻。請設定門檻後按「確認設定門檻」。")
        return

    selected_features, pass_thresholds, standby_thresholds = config
    rows = []
    for feature, label in FEATURE_COLUMNS.items():
        rows.append(
            {
                "表型": label,
                "是否啟用": "是" if feature in selected_features else "否",
                "通過門檻": number_input_value(pass_thresholds.get(feature)),
                "備用門檻": number_input_value(standby_thresholds.get(feature)),
            }
        )
    centered_thresholds = (
        pd.DataFrame(rows)
        .style.set_properties(**{"text-align": "center"})
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [
                        ("width", "100%"),
                        ("table-layout", "fixed"),
                        ("margin-left", "auto"),
                        ("margin-right", "auto"),
                    ],
                },
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ]
        )
    )
    st.markdown(
        f"<div style='width:100%; display:flex; justify-content:center;'>{centered_thresholds.hide(axis='index').to_html()}</div>",
        unsafe_allow_html=True,
    )


def hardware_settings_page() -> None:
    st.header("硬體連接設定")
    st.caption("設定掃碼器測試欄位、Arduino 燈號 COM 與 RS485/USB 秤重設備。掃碼器若為 USB HID 模式，通常等同鍵盤輸入，不需要另外指定 COM。")

    with st.container(border=True):
        st.subheader("掃碼器測試")
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
            st.success(f"最後掃描內容：{st.session_state.scanner_last_value}")

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

        test_col1, test_col2, test_col3 = st.columns(3)
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
        if test_col3.button("重新連線 Arduino", use_container_width=True, key="hardware_reconnect_arduino"):
            if not arduino_port:
                st.warning("請先選擇 Arduino serial port。")
            else:
                try:
                    replies = reconnect_arduino(arduino_port)
                    st.success("Arduino 重新連線成功。" + (f" 回覆：{', '.join(replies)}" if replies else ""))
                except Exception as exc:
                    st.error(f"Arduino 重新連線失敗：{exc}")

        if hardware_enabled and arduino_port:
            st.info(f"目前燈號 COM：{arduino_port}")

    with st.container(border=True):
        st.subheader("秤重設備測試")
        st.caption(f"秤重設備設定：RS485 轉 USB，{SCALE_BAUD_RATE} bps，封包格式為 0x02 + ASCII 克數 + 0x03。")
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
        if scale_test_col1.button("測試秤重讀值", use_container_width=True, key="hardware_test_scale"):
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
                            f"2 秒平均重量：{result.average_g:.1f} g；"
                            f"最新重量：{result.latest_g} g；樣本數：{len(result.samples)}"
                        )
                        if result.raw_packets:
                            st.caption(f"原始封包：{', '.join(result.raw_packets[-5:])}")
                        if result.errors:
                            st.warning("；".join(result.errors))
                except Exception as exc:
                    st.error(f"秤重設備連線失敗：{exc}")
        if scale_test_col2.button("清除秤重即時紀錄", use_container_width=True, key="hardware_clear_scale_history"):
            st.session_state.scale_recent_samples = []
            st.success("已清除秤重即時紀錄。")

        if scale_port:
            st.info(f"目前秤重設備 COM：{scale_port}")


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
        if confirmed_id:
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
    if confirmed_id:
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
    st.header("相機調整")
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

    st.info("\u8acb\u5728\u5716\u7247\u4e0a\u4f9d\u5e8f\u9ede\u9078\u6bd4\u4f8b\u5c3a\u5169\u7aef\u3002\u82e5\u9ede\u932f\uff0c\u8acb\u4f7f\u7528\u4e0b\u65b9\u300c\u6e05\u9664\u6821\u6b63\u9ede\u300d\u3002")
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

    points = st.session_state.calibration_points
    st.write(f"\u5df2\u9078\u53d6\u9ede\u4f4d\uff1a{points}")

    if len(points) == 2:
        install_n_click_button_hotkey("\u78ba\u5b9a\u6821\u6b63")
        (x1, y1), (x2, y2) = points
        pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if pixel_distance > 0:
            cm_per_pixel = float(real_length_cm) / pixel_distance
            if st.button("\u78ba\u5b9a\u6821\u6b63", type="primary", use_container_width=True):
                st.session_state.calibration_cm_per_pixel = cm_per_pixel
                st.session_state.phenotype_cm_per_pixel = cm_per_pixel
                st.session_state.phenotype_cm_per_pixel_version = int(
                    st.session_state.get("phenotype_cm_per_pixel_version", 0)
                ) + 1
                st.success(f"\u6821\u6b63\u5df2\u5957\u7528\uff1acm_per_pixel = {cm_per_pixel:.6f}")
        else:
            st.error("\u5169\u500b\u6821\u6b63\u9ede\u8ddd\u96e2\u70ba 0\uff0c\u8acb\u6e05\u9664\u5f8c\u91cd\u65b0\u9ede\u9078\u3002")

    if st.button("\u6e05\u9664\u6821\u6b63\u9ede", use_container_width=True):
        st.session_state.calibration_points = []
        st.session_state.calibration_last_click_unix_time = (
            click_value.get("unix_time") if click_value is not None else None
        )
        st.session_state.calibration_click_key_version = click_key_version + 1
        st.rerun(scope="fragment")

def scale_calibration_section() -> None:
    st.header("比例尺校正")
    st.caption("拍攝比例尺照片後，依序用滑鼠點選比例尺兩端，再輸入真實長度。")
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
    st.header("表型資料輸入")
    st.caption("確認 chicken ID 後，系統會讀取秤重設備並建立背景推論任務；前一隻雞推論時，可繼續掃描下一隻。")
    st.info("量測設定：錄影 1 秒，固定推論 10 張照片。")
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
    st.caption(f"目前套用 cm_per_pixel：{cm_per_pixel_value:.6f}")

    setting_col1, setting_col2 = st.columns(2)
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

    query_id = resolve_qrcode_input("phenotype_entry", auto_focus=True, scanner_mode=True)
    if not query_id:
        st.info("請掃描 QR code，或手動輸入 chicken ID 後按 Enter。")
        render_inference_queue_panel()
        return
    if st.session_state.get("scale_active_chicken_id") != query_id:
        st.session_state.scale_active_chicken_id = query_id
        st.session_state.scale_recent_samples = []

    selected = get_chicken(query_id)
    if selected:
        st.success(f"已選擇 chicken ID：{query_id}")
    else:
        st.info(f"資料庫尚無 chicken ID：{query_id}，儲存後會新增此雞隻。")

    scale_port = default_scale_port()
    live_scale_panel(scale_port, window_seconds=2.0)

    with st.form("phenotype_entry_form"):
        st.text_input("chicken ID", value=query_id, disabled=True)
        col1, col2, col3 = st.columns(3)
        manual_weight_g = col1.number_input(
            "手動重量備援 weight_g (g)",
            min_value=0.0,
            value=number_input_value(selected["weight_g"]) if selected else 0.0,
            step=10.0,
            help="正常流程會在按下推論時讀取秤重設備 2 秒平均；只有秤重設備未設定或讀不到有效重量時，才使用此欄位。",
        )
        cm_per_pixel = col2.number_input(
            "cm_per_pixel",
            min_value=0.000001,
            value=cm_per_pixel_value,
            step=0.001,
            format="%.6f",
            key=f"phenotype_cm_per_pixel_input_{int(st.session_state.get('phenotype_cm_per_pixel_version', 0))}",
        )
        col3.text_input(
            "推論照片張數",
            value="10",
            disabled=True,
        )
        submitted = st.form_submit_button("計算雞冠及腳脛數據", type="primary", use_container_width=True)

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

    completed_job = latest_completed_job_for_chicken(query_id)
    if completed_job and completed_job.result_row:
        st.subheader("目前 chicken ID 最近完成結果")
        st.dataframe(pd.DataFrame([completed_job.result_row]), use_container_width=True, hide_index=True)
        if completed_job.video_id:
            show_inference_sample_frames(video_ids=[completed_job.video_id])

    render_inference_queue_panel()


def data_management_page() -> None:
    st.header("雞隻資料管理")

    sample_col, import_col, export_col = st.columns(3)
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

    st.subheader("新增雞隻")
    with st.form("add_chicken_form", clear_on_submit=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        chicken_id = col1.text_input("chicken ID", placeholder="C001")
        weight_g = col2.number_input("體重 weight_g (g)", min_value=0.0, step=10.0)
        comb_area_cm2 = col3.number_input("雞冠面積 comb_area_cm2 (cm²)", min_value=0.0, value=0.0, step=0.5)
        shank_width_cm = col4.number_input("腳脛寬度 shank_width_cm (cm)", min_value=0.0, value=0.0, step=0.1)
        shank_length_cm = col5.number_input("腳脛長度 shank_length_cm (cm)", min_value=0.0, value=0.0, step=0.1)
        submitted = st.form_submit_button("新增資料")
        if submitted:
            try:
                add_chicken(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)
                st.success(f"已新增雞隻：{chicken_id}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.subheader("修改或刪除雞隻")
    chickens = get_all_chickens()
    if chickens:
        chicken_ids = [row["chicken_id"] for row in chickens]
        selected_id = st.selectbox("選擇 chicken ID", chicken_ids, key="manage_chicken_id")
        selected = get_chicken(selected_id)
        if selected:
            with st.form("edit_chicken_form"):
                col1, col2, col3, col4 = st.columns(4)
                edit_weight = col1.number_input("體重 weight_g (g)", value=number_input_value(selected["weight_g"]), step=10.0)
                edit_comb = col2.number_input("雞冠面積 comb_area_cm2 (cm²)", value=number_input_value(selected["comb_area_cm2"]), step=0.5)
                edit_shank = col3.number_input(
                    "腳脛寬度 shank_width_cm (cm)",
                    value=number_input_value(selected.get("shank_width_cm")),
                    step=0.1,
                )
                edit_shank_length = col4.number_input(
                    "腳脛長度 shank_length_cm (cm)",
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

    st.subheader("所有雞隻資料")
    show_data_table(hide_result_columns=True)


def grading_page() -> None:
    st.header("分級設定與執行")
    st.caption("設定各表型通過門檻與備用門檻後，系統會更新資料庫中的通過、備用、不通過結果。")
    render_current_thresholds()

    pass_thresholds = {}
    standby_thresholds = {}
    with st.container(border=True, key="pass_threshold_block"):
        st.markdown("<h3 style='text-align: center;'>通過門檻</h3>", unsafe_allow_html=True)
        pass_cols = st.columns(len(FEATURE_COLUMNS))
        for index, (feature, label) in enumerate(FEATURE_COLUMNS.items()):
            pass_thresholds[feature] = pass_cols[index].number_input(
                label,
                min_value=0.0,
                step=0.1,
                key=f"{feature}_pass_threshold",
            )

    with st.container(border=True, key="standby_threshold_block"):
        st.markdown("<h3 style='text-align: center;'>備用門檻</h3>", unsafe_allow_html=True)
        standby_cols = st.columns(len(FEATURE_COLUMNS))
        for index, (feature, label) in enumerate(FEATURE_COLUMNS.items()):
            standby_thresholds[feature] = standby_cols[index].number_input(
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
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))

    st.subheader("目前雞隻資料")
    show_data_table()


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
    )


def final_scan_page() -> None:
    st.header("再次掃描 QR code 與亮燈判定")
    st.caption("完成表型資料輸入與分級後，再次掃描既有 QR code，依分級結果顯示綠燈、黃燈或紅燈。")

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
    st.title("紅羽土雞表型分級系統")
    st.caption("掃描雞隻 QR code，根據設定門檻分級。")

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
    st.divider()
    pages[selected_page]()


if __name__ == "__main__":
    main()


