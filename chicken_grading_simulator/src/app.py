"""雞隻表型資料分級與亮燈篩選系統。"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

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
    update_grading_results,
)
from arduino_control import commands_for_grade, light_for, list_serial_ports, reconnect_arduino, send_arduino_commands
from scale_control import (
    SCALE_BAUD_RATE,
    average_recent_samples,
    scale_monitor,
)
from grading import (
    FEATURE_COLUMNS,
    GRADE_FAIL,
    GRADE_PASS,
    GRADE_STANDBY,
    active_features,
    grade_and_store_with_latest_config,
    grade_details,
    summarize_results,
    validate_grading_config,
)
from inference_utils import OUTPUTS_DIR
from inference_queue import (
    STATUS_COMPLETED,
    has_active_jobs,
    latest_completed_job_for_chicken,
    list_jobs,
)
from sample_data import create_sample_data
import i18n
from i18n import t


FINAL_LIGHT_SECONDS = 3.0


APP_ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "chicken_icon.png"


def app_icon_html() -> str:
    """The app icon (chicken on a white rounded tile) inline, for the sidebar logo."""
    import base64

    try:
        data = base64.b64encode(APP_ICON_PATH.read_bytes()).decode("ascii")
    except OSError:
        return "🐔"
    return f'<img src="data:image/png;base64,{data}" alt="" />'


st.set_page_config(
    page_title=t("雞隻表型資料分級與亮燈篩選系統"),
    page_icon=str(APP_ICON_PATH),  # chicken icon (assets/chicken_icon.png)
    layout="wide",
    initial_sidebar_state="expanded",  # the sidebar is the navigation: always open
)

init_db()

try:
    from secure_sync import start_sync_worker

    start_sync_worker()
except Exception:
    # The grading system remains fully local when secure sync is not configured.
    pass


def number_input_value(value, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return float(value)


def run_page_script(html: str) -> None:
    """Run a <script> in the page (hotkeys, scanner autofocus). st.html replaces the deprecated
    components.v1.html: no iframe, so `window.parent` is the page itself, which the scripts already use."""
    st.html(html, unsafe_allow_javascript=True)


# Shared by every hotkey / autofocus script (needs `parentDocument` in scope).
SCANNER_JS = """
const isScannerInput = (element) =>
    !!element && !!element.matches &&
    element.matches('input:not([type="file"])') &&
    (element.getAttribute("placeholder") || "").trim().toUpperCase() === "C001";
const findScannerInput = () =>
    Array.from(parentDocument.querySelectorAll('input:not([type="file"])'))
        .find((input) => !input.disabled && input.offsetParent !== null && isScannerInput(input));
// Hotkeys must not swallow real typing. The panel buttons type a lower-case key while the empty
// scanner field has focus, so that still counts as a hotkey; scanners type IDs into it (Shift /
// upper-case), and any other input, textarea or editable element is real typing.
const ignoreHotkey = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) {
        return true;
    }
    const element = event.target;
    const tag = ((element && element.tagName) || "").toUpperCase();
    if (!element || !(element.isContentEditable || tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT")) {
        return false;
    }
    const panelKeyInEmptyScanner = isScannerInput(element) && !element.value &&
        !event.shiftKey && event.key === event.key.toLowerCase();
    return !panelKeyInEmptyScanner;
};
"""


def install_r_submit_hotkey(button_text: str) -> None:
    # Match the button text as displayed (t() leaves already-translated text unchanged).
    escaped_text = json.dumps(t(button_text), ensure_ascii=False)
    run_page_script(
        f"""
        <script>
        (() => {{
            const buttonText = {escaped_text};
            try {{
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                {SCANNER_JS}
                if (parentWindow.__chickenrHotkeyHandler) {{
                    parentDocument.removeEventListener(
                        "keydown",
                        parentWindow.__chickenrHotkeyHandler,
                        true
                    );
                }}
                const handler = (event) => {{
                    if ((event.key.toLowerCase() !== "r" && event.code !== "KeyR") || ignoreHotkey(event)) {{
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
    )


def install_n_scanner_focus_hotkey() -> None:
    run_page_script(
        """
        <script>
        (() => {
            try {
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                """ + SCANNER_JS + """
                if (parentWindow.__chickennScannerFocusHotkeyHandler) {
                    parentDocument.removeEventListener(
                        "keydown",
                        parentWindow.__chickennScannerFocusHotkeyHandler,
                        true
                    );
                }
                const focusScannerInput = () => {
                    const target = findScannerInput();
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
                    if ((event.key.toLowerCase() !== "n" && event.code !== "KeyN") || ignoreHotkey(event)) {
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
    )


def install_scanner_input_autofocus() -> None:
    run_page_script(
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
            """ + SCANNER_JS + """
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
                    const target = findScannerInput();
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
    )


def install_n_click_button_hotkey(button_text: str) -> None:
    # Match the button text as displayed (t() leaves already-translated text unchanged).
    escaped_button_text = json.dumps(t(button_text), ensure_ascii=False)
    run_page_script(
        f"""
        <script>
        (() => {{
            try {{
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                {SCANNER_JS}
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
                    if ((event.key.toLowerCase() !== "n" && event.code !== "KeyN") || ignoreHotkey(event)) {{
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
    )


def install_p_page_cycle_hotkey(page_names: list[str], current_page: str) -> None:
    if not page_names:
        return
    escaped_page_names = json.dumps(page_names, ensure_ascii=False)
    escaped_current_page = json.dumps(current_page if current_page in page_names else page_names[0], ensure_ascii=False)
    run_page_script(
        f"""
        <script>
        (() => {{
            const pageNames = {escaped_page_names};
            const fallbackCurrentPage = {escaped_current_page};
            try {{
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                {SCANNER_JS}
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
                    if ((event.key.toLowerCase() !== "p" && event.code !== "KeyP") || ignoreHotkey(event)) {{
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
    )


LIGHT_CSS = """
<style>
:root {
    --content-width: 1760px;
    --panel-border: #263246;
    --surface: #131b27;
    --surface-2: #182233;
    --surface-hover: #1d2940;
    --line: #263246;
    --line-strong: #34445e;
    --muted: #93a1b5;
    --text-strong: #f3f6fb;
    --accent: #4f8cff;
    --accent-soft: rgba(79, 140, 255, 0.14);
    --pass: #22c55e;
    --standby: #facc15;
    --fail: #ef4444;
}

/* ---------- page frame ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 420px at 50% -180px, rgba(79, 140, 255, 0.10), transparent 70%),
        #0b1018;
}
header[data-testid="stHeader"] {
    height: 2rem;
    background: transparent;
}
[data-testid="stMainBlockContainer"] {
    width: 100%;  /* of the area right of the sidebar */
    max-width: var(--content-width);
    padding: 0.55rem 1.4rem 1.2rem;
}
[data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.85rem;
}

/* ---------- masthead ---------- */
.app-masthead {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin: 0 0 0.35rem;
}
.app-logo {
    display: grid;
    place-items: center;
    width: 2.3rem;
    height: 2.3rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #4f8cff, #7c5cff);
    box-shadow: 0 6px 18px rgba(79, 140, 255, 0.35);
    font-size: 1.25rem;
}
.app-title {
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: 0.02em;
    color: var(--text-strong);
}
.app-subtitle {
    color: var(--muted);
    font-size: 0.86rem;
    padding-left: 0.85rem;
    border-left: 1px solid var(--line-strong);
}

/* ---------- left sidebar navigation (Neon sidebar) ----------
   The page switch is still the same segmented control (key main_page_navigation_segmented), only moved into
   st.sidebar and laid out as a vertical list. Icons per tab come from nav_icon_css(); the springy pop-out
   transition is added by the 動態 style (ui_style.MOTION_CSS). */
/* the sidebar can't be collapsed or resized (no collapse / expand buttons, no drag handle) */
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarResizeHandle"] {
    display: none !important;
}
/* On a narrow window Streamlit marks the sidebar collapsed and would lay it over the page: keep it shown,
   fixed at the left, and move the page to its right instead. */
section[data-testid="stSidebar"][aria-expanded="false"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
    transform: none !important;
    margin-left: 0 !important;
    z-index: 100;
}
[data-testid="stAppViewContainer"]:has(> section[data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stMain"] {
    margin-left: 330px;
    width: calc(100vw - 330px);
    max-width: calc(100vw - 330px);
    flex: 0 0 auto;
}
section[data-testid="stSidebar"] {
    visibility: visible !important;
    width: 330px !important;
    min-width: 330px !important;
    max-width: 330px !important;
    background: transparent !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {
    margin: 14px 0 14px 14px;
    width: calc(100% - 14px);
    box-sizing: border-box;
    padding-left: 0 !important;
    padding-right: 0 !important;
    height: calc(100% - 28px);
    border-radius: 24px;
    background: linear-gradient(180deg, #111927 0%, #0d141f 100%);
    border: 1px solid #1d2838;
    box-shadow: 0 20px 50px -20px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
[data-testid="stSidebarHeader"] { height: 2.4rem; padding: 0.6rem 0.9rem 0; }
[data-testid="stSidebarUserContent"] { padding: 0 12px 18px 12px; }
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 4px 14px 4px;
    margin-bottom: 6px;
    border-bottom: 1px solid #1d2838;
}
.sidebar-brand .app-logo { width: 48px; height: 48px; border-radius: 11px; font-size: 1.4rem; overflow: hidden;
    background: none; box-shadow: none; }  /* white line-art chicken straight on the dark panel */
.sidebar-brand .app-logo img { width: 100%; height: 100%; display: block; }
.sidebar-brand-name { font-size: 18px; font-weight: 900; letter-spacing: 0.02em; color: #e7ecf3; line-height: 1.25;
    margin-bottom: 6px; }  /* a little air above 表型分級系統 */
.sidebar-brand-sub { font-size: 13px; font-weight: 700; color: #7c8ba3; line-height: 1.3; }
.sidebar-foot {
    margin-top: 10px;
    padding: 14px 6px 2px;
    border-top: 1px solid #1d2838;
    font-size: 13px;
    font-weight: 700;
    color: #7c8ba3;
}
.st-key-main_page_navigation_segmented,
.st-key-main_page_navigation_segmented [data-testid="stButtonGroup"],
.st-key-main_page_navigation_segmented [data-testid="stButtonGroup"] > div {
    width: 100% !important;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    flex-wrap: nowrap;
    width: 100%;
    max-width: none !important;
    gap: 8px;
    padding: 4px 12px 4px 14px;
    background: transparent;
    border: none;
    overflow: visible;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button {
    position: relative;
    display: flex;
    justify-content: flex-start;
    align-items: center;
    gap: 14px;
    width: 100%;
    min-height: 56px;
    margin: 0 !important;
    padding: 0 16px 0 18px;
    border: 1px solid transparent !important;
    border-radius: 14px !important;
    background: transparent;
    color: #9fb0c8;
    white-space: nowrap;
    text-align: left;
    overflow: visible !important;  /* the glowing bar sits just left of the active tab */
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button > div {
    justify-content: flex-start;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button p {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: inherit;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button::before {
    content: "";
    flex: 0 0 auto;
    width: 24px;
    height: 24px;
    background-color: currentColor;
    -webkit-mask: var(--nav-icon) center / contain no-repeat;
    mask: var(--nav-icon) center / contain no-repeat;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button:hover {
    color: #e7ecf3;
    background: rgba(79, 140, 255, 0.08);
    border-color: rgba(79, 140, 255, 0.18) !important;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button:focus-visible {
    outline: 2px solid #7aa7ff;
    outline-offset: 2px;
}
/* the page you are on: bigger, popped out to the right, neon blue */
.st-key-main_page_navigation_segmented [role="radiogroup"] button[aria-checked="true"] {
    color: #ffffff;
    transform: scale(1.045) translateX(6px);
    background: linear-gradient(100deg, rgba(79, 140, 255, 0.30) 0%, rgba(79, 140, 255, 0.10) 55%, rgba(79, 140, 255, 0.04) 100%);
    border-color: rgba(122, 167, 255, 0.65) !important;
    box-shadow: 0 0 0 1px rgba(79, 140, 255, 0.25), 0 10px 28px -6px rgba(79, 140, 255, 0.55),
        0 0 26px rgba(79, 140, 255, 0.28), inset 0 0 18px rgba(79, 140, 255, 0.18);
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button[aria-checked="true"]::after {
    content: "";
    position: absolute;
    left: -12px;
    top: 12px;
    bottom: 12px;
    width: 4px;
    border-radius: 4px;
    background: #7aa7ff;
    box-shadow: 0 0 10px #4f8cff, 0 0 22px #4f8cff;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button[aria-checked="true"]::before {
    filter: drop-shadow(0 0 6px rgba(122, 167, 255, 0.9));
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button[aria-checked="true"] p {
    color: #ffffff;
}

/* 模型維護與更新: the two cards in a row are the same height (stretched to the taller one) */
.st-key-maint_row_sync [data-testid="stHorizontalBlock"],
.st-key-maint_row_models [data-testid="stHorizontalBlock"] {
    align-items: stretch;
}
.st-key-maint_row_sync [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
.st-key-maint_row_models [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    height: 100%;
}
.st-key-maint_row_sync [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > div:has(> [class*="st-key-card_"]),
.st-key-maint_row_models [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > div:has(> [class*="st-key-card_"]),
.st-key-maint_row_sync .st-key-card_sync_upload, .st-key-maint_row_sync .st-key-card_manual_bundle,
.st-key-maint_row_models .st-key-card_model_select, .st-key-maint_row_models .st-key-card_model_import {
    flex: 1 1 auto;
}
.st-key-card_model_import [data-testid="stForm"] { padding: 0; background: transparent; border: none; }
/* text / date fields would otherwise blend into the card */
.st-key-card_model_import [data-baseweb="input"],
.st-key-card_model_import [data-testid="stTextInputRootElement"],
.st-key-card_model_import [data-testid="stFileUploaderDropzone"] {
    background: #0f1622 !important;
    border: 1px solid #263246 !important;
    border-radius: 10px;
}
.st-key-card_model_import [data-baseweb="input"] > div,
.st-key-card_model_import [data-baseweb="input"] input { background: transparent !important; }

/* 硬體連接設定: three cards of equal height, titles bigger and centred */
.st-key-hw_row [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-hw_row [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height: 100%; }
.st-key-hw_row [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > div:has(> [class*="st-key-card_hardware"]),
.st-key-hw_row [class*="st-key-card_hardware"] { flex: 1 1 auto; }
[class*="st-key-card_hardware"] [data-testid="stHeading"] h3 {
    text-align: center;
    font-size: 1.75rem !important;  /* the global h3 rule is !important too */
    font-weight: 800 !important;
    padding-bottom: 0.4rem;
}
/* the heading's hover link icon would push the centred title to the left */
[class*="st-key-card_hardware"] [data-testid="stHeaderActionElements"] { display: none; }

/* other segmented controls (review filters etc.) */
[data-testid="stButtonGroup"] button p {
    font-size: 1rem;
    font-weight: 650;
}
[data-testid="stButtonGroup"] button {
    min-height: 2.5rem;
    white-space: nowrap;
}

/* ---------- page heading ---------- */
.page-heading {
    display: flex;
    align-items: baseline;
    gap: 0.9rem;
    margin: 0.4rem 0 0.75rem;
    padding-left: 0.75rem;
    border-left: 4px solid var(--accent);
}
.page-heading h2 {
    font-size: 1.4rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0;
    padding: 0;
    color: var(--text-strong);
}
.page-heading p {
    color: var(--muted);
    font-size: 0.86rem;
    margin: 0;
}
h3 {
    font-size: 1.15rem !important;
    font-weight: 750 !important;
    letter-spacing: 0.01em;
}

/* ---------- cards ---------- */
div[class*="st-key-card_"] {
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    border: 1px solid var(--line) !important;
    border-radius: 14px;
    padding: 0.9rem 1rem !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
}
div[class*="st-key-card_"] h3 {
    margin-top: 0;
    padding-top: 0;
}

/* ---------- metrics: the important numbers ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.6rem 0.95rem 0.65rem;
    min-height: 4.7rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: var(--muted);
    font-size: 0.86rem;
    font-weight: 650;
    letter-spacing: 0.02em;
}
[data-testid="stMetricValue"] {
    font-size: 2.05rem;
    font-weight: 800;
    line-height: 1.15;
    color: var(--text-strong);
    font-variant-numeric: tabular-nums;
}

/* ---------- grading light (final result) ---------- */
.light-panel {
    display: flex;
    align-items: center;
    gap: 30px;
    padding: 22px 30px;
    margin: 4px 0;
    border-radius: 18px;
    border: 1px solid var(--line-strong);
    background: linear-gradient(90deg, rgba(148, 163, 184, 0.10), transparent 70%), var(--surface);
}
.light-panel:has(.light-green) {
    border-color: rgba(34, 197, 94, 0.65);
    background: linear-gradient(90deg, rgba(34, 197, 94, 0.20), transparent 72%), var(--surface);
}
.light-panel:has(.light-yellow) {
    border-color: rgba(250, 204, 21, 0.65);
    background: linear-gradient(90deg, rgba(250, 204, 21, 0.18), transparent 72%), var(--surface);
}
.light-panel:has(.light-red) {
    border-color: rgba(239, 68, 68, 0.65);
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.20), transparent 72%), var(--surface);
}
.light-circle {
    flex: 0 0 auto;
    width: 132px;
    height: 132px;
    border-radius: 50%;
    border: 6px solid rgba(255, 255, 255, 0.10);
    box-shadow: inset 0 10px 26px rgba(255, 255, 255, 0.35), 0 0 46px var(--glow, transparent);
}
.light-green { background: radial-gradient(circle at 35% 30%, #86efac, #16a34a 62%); --glow: rgba(34, 197, 94, 0.65); }
.light-yellow { background: radial-gradient(circle at 35% 30%, #fef08a, #eab308 62%); --glow: rgba(250, 204, 21, 0.6); }
.light-red { background: radial-gradient(circle at 35% 30%, #fca5a5, #dc2626 62%); --glow: rgba(239, 68, 68, 0.65); }
.light-gray { background: radial-gradient(circle at 35% 30%, #cbd5e1, #64748b 62%); }
.light-text {
    font-size: 46px;
    font-weight: 850;
    line-height: 1.2;
    color: var(--text-strong);
    letter-spacing: 0.02em;
}
.light-panel:has(.light-green) .light-text { color: #22c55e; }
.light-panel:has(.light-yellow) .light-text { color: #eab308; }
.light-panel:has(.light-red) .light-text { color: #ef4444; }
.status-note {
    color: var(--muted);
    font-size: 19px;
    font-weight: 650;
    margin-top: 6px;
}

/* ---------- alerts ---------- */
[data-testid="stAlert"] {
    padding: 0;
}
[data-testid="stAlertContainer"] {
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    font-size: 1rem;
    font-weight: 600;
}

/* ---------- buttons ---------- */
[data-testid="stButton"] > button,
[data-testid="stFormSubmitButton"] > button,
[data-testid="stDownloadButton"] > button {
    min-height: 3rem;
    padding: 0.55rem 1rem;
    border-width: 1px;
    font-size: 1.05rem;
    font-weight: 750;
    transition: transform 80ms ease, box-shadow 120ms ease, border-color 120ms ease;
}
[data-testid="stButton"] > button p,
[data-testid="stFormSubmitButton"] > button p,
[data-testid="stDownloadButton"] > button p {
    font-size: 1.05rem;
    font-weight: 750;
}
[data-testid="stButton"] > button[kind="secondary"],
[data-testid="stFormSubmitButton"] > button[kind="secondaryFormSubmit"],
[data-testid="stDownloadButton"] > button[kind="secondary"] {
    background: var(--surface-2);
    border-color: var(--line-strong);
}
[data-testid="stButton"] > button[kind="secondary"]:hover:not(:disabled),
[data-testid="stFormSubmitButton"] > button[kind="secondaryFormSubmit"]:hover:not(:disabled),
[data-testid="stDownloadButton"] > button[kind="secondary"]:hover:not(:disabled) {
    border-color: var(--accent);
    background: var(--surface-hover);
}
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
    background: linear-gradient(180deg, #5b95ff, #3f7cf0);
    border-color: #5b95ff;
    box-shadow: 0 6px 18px rgba(79, 140, 255, 0.30);
}
[data-testid="stButton"] > button:disabled,
[data-testid="stFormSubmitButton"] > button:disabled,
[data-testid="stDownloadButton"] > button:disabled {
    background: var(--surface) !important;
    border-color: var(--line) !important;
    box-shadow: none !important;
    opacity: 0.55;
    cursor: not-allowed;
}
[data-testid="stButton"] > button:active:not(:disabled),
[data-testid="stFormSubmitButton"] > button:active:not(:disabled) {
    transform: translateY(1px);
}

/* ---------- inputs ---------- */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    min-height: 3rem;
    font-size: 1.05rem;
    font-weight: 650;
    font-variant-numeric: tabular-nums;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 3rem;
    font-size: 1.02rem;
    font-weight: 650;
}
[data-testid="stWidgetLabel"] p {
    font-size: 0.9rem;
    font-weight: 650;
    color: #c5cfdc;
}
/* scanner field: the operator's main input */
input[placeholder="C001"] {
    font-size: 1.3rem !important;
    letter-spacing: 0.06em;
}
[data-testid="stTextInputRootElement"]:has(input[placeholder="C001"]) {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft);
}

/* ---------- expanders, forms, tables ---------- */
[data-testid="stExpander"] details {
    border-radius: 12px;
    border-color: var(--line);
    background: var(--surface);
}
[data-testid="stExpander"] details summary {
    min-height: 2.4rem;
}
[data-testid="stExpander"] details summary p {
    font-weight: 650;
}
[data-testid="stForm"] {
    border-color: var(--line);
    border-radius: 14px;
    background: var(--surface);
    padding: 0.8rem 0.9rem;
}
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stDivider"] {
    margin: 0.2rem 0 0.35rem;
}
[data-testid="stCaptionContainer"] p {
    color: var(--muted);
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
    background: var(--surface);
}

/* ---------- camera previews & result images ---------- */
.st-key-dual_preview_comb img, .st-key-dual_preview_shank img {
    max-height: 30vh;
    object-fit: contain;
    background: #000;
    border-radius: 10px;
    border: 1px solid var(--line);
}
.st-key-adjust_previews .st-key-dual_preview_comb img,
.st-key-adjust_previews .st-key-dual_preview_shank img { max-height: 45vh; }  /* 相機調整: 1.5× the usual 30vh */
.st-key-snapshot_result_images img {
    max-height: 19.5vh;
    cursor: zoom-in;  /* click = full screen (ui_style.IMAGE_ZOOM_HTML) */
    object-fit: contain;
    background: #000;
    border-radius: 10px;
}
.st-key-photo_mode_uploads img {
    max-height: 30vh;
    object-fit: contain;
    background: #000;
    border-radius: 10px;
}
/* script-only st.html blocks (hotkeys, autofocus, progress-ring driver) take no space */
[data-testid="stElementContainer"]:has(> [data-testid="stHtml"] > script:only-child) {
    position: absolute; width: 0; height: 0; overflow: hidden; }
/* hidden helper iframes (scanner auto-focus) must not add a gap, so the scan box lines up with the settings row */
.st-key-pheno_operator [data-testid="stElementContainer"][height="0px"] { position: absolute; }
.st-key-pheno_operator { padding-top: 7px; }  /* scan label on the same line as the photo-mode toggle */
/* ---------- units shown inside number boxes (after the number, before the − + buttons) ---------- */
[data-testid="stNumberInputContainer"] > div { order: 3; }
[data-testid="stNumberInputContainer"]::after { order: 2; align-self: center; padding: 0 10px 0 4px; color: var(--muted);
    font-size: .95rem; font-weight: 600; white-space: nowrap; pointer-events: none; }
[data-testid="stNumberInputContainer"]:has(input[aria-label$="(g)"])::after,
.st-key-weight_g_pass_threshold [data-testid="stNumberInputContainer"]::after,
.st-key-weight_g_standby_threshold [data-testid="stNumberInputContainer"]::after { content: "g"; }
[data-testid="stNumberInputContainer"]:has(input[aria-label$="(cm²)"])::after,
.st-key-comb_area_cm2_pass_threshold [data-testid="stNumberInputContainer"]::after,
.st-key-comb_area_cm2_standby_threshold [data-testid="stNumberInputContainer"]::after { content: "cm²"; }
[data-testid="stNumberInputContainer"]:has(input[aria-label$="(cm)"])::after,
.st-key-shank_width_cm_pass_threshold [data-testid="stNumberInputContainer"]::after,
.st-key-shank_width_cm_standby_threshold [data-testid="stNumberInputContainer"]::after,
.st-key-shank_length_cm_pass_threshold [data-testid="stNumberInputContainer"]::after,
.st-key-shank_length_cm_standby_threshold [data-testid="stNumberInputContainer"]::after { content: "cm"; }

/* ---------- 雞隻資料管理: four top boxes as tall as 資料筆數, content centred ---------- */
.st-key-data_toolbar [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-data_toolbar [data-testid="stMetric"],
.st-key-data_toolbar [data-testid="stButton"] > button,
.st-key-data_toolbar [data-testid="stDownloadButton"] > button,
.st-key-data_toolbar [data-testid="stFileUploaderDropzone"] {
    height: 96px; min-height: 96px; box-sizing: border-box; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center; margin: 0; }
.st-key-data_toolbar [data-testid="stMetric"] > div { align-items: center; justify-content: center; width: 100%; }
.st-key-data_toolbar [data-testid="stMetricLabel"], .st-key-data_toolbar [data-testid="stMetricValue"] {
    justify-content: center; text-align: center; width: 100%; align-self: center; }
/* the label is a CSS grid with an auto-width first column (label + help icon): make it a centred flex row */
.st-key-data_toolbar [data-testid="stMetricLabel"] { display: flex !important; justify-content: center; }
.st-key-data_toolbar [data-testid="stFileUploaderDropzone"] { flex-direction: row; gap: 14px; padding: 0 16px;
    justify-content: center !important; align-items: center !important; }
.st-key-data_toolbar [data-testid="stFileUploaderDropzone"] > * { flex: 0 0 auto !important; margin: 0 !important; }
.st-key-data_toolbar [data-testid="stFileUploader"] section { margin: 0; }

/* phenotype result numbers (static style): clear item names, large values */
.st-key-snapshot_metrics [data-testid="stMetricLabel"] p { font-size: 1.15rem; font-weight: 700; color: var(--text-strong); }
.st-key-snapshot_metrics [data-testid="stMetricValue"] { font-size: 2.3rem; }
.st-key-snapshot_status p {
    font-size: 1.15rem;
    margin: 0.15rem 0 0;
}

/* ---------- grading thresholds ---------- */
.threshold-summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 0.35rem;
}
.threshold-summary-card {
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
    text-align: center;
}
.threshold-summary-label {
    font-size: 0.86rem;
    font-weight: 650;
    color: var(--muted);
}
.threshold-summary-value {
    font-size: 1.2rem;
    font-weight: 800;
    margin-top: 0.2rem;
    font-variant-numeric: tabular-nums;
}
.threshold-summary-disabled {
    opacity: 0.5;
}
.summary-stat {
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.55rem 0.45rem;
    text-align: center;
}
/* the stats are markdown (Streamlit gives it a -1rem bottom margin): keep the row clear of the expander below */
.st-key-grading_summary { padding-bottom: 1rem; }
.summary-stat-label {
    color: var(--muted);
    font-size: 0.82rem;
    font-weight: 650;
}
.summary-stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.25;
    margin-top: 0.1rem;
    font-variant-numeric: tabular-nums;
}
.st-key-pass_threshold_block {
    border: 2px solid rgba(34, 197, 94, 0.75) !important;
    border-radius: 14px;
    padding: 10px 12px;
    margin-bottom: 4px;
    background: linear-gradient(180deg, rgba(34, 197, 94, 0.10), rgba(34, 197, 94, 0.02));
}
.st-key-standby_threshold_block {
    border: 2px solid rgba(250, 204, 21, 0.75) !important;
    border-radius: 14px;
    padding: 10px 12px;
    margin-bottom: 4px;
    background: linear-gradient(180deg, rgba(250, 204, 21, 0.10), rgba(250, 204, 21, 0.02));
}
.st-key-pass_threshold_block, .st-key-standby_threshold_block { margin-top: 0.75rem; }
/* ---------- feature checkboxes (grading page) ---------- */
.st-key-feature_select_block { margin-top: 1.6rem; padding: 14px 20px 10px; }
.feature-select-title { font-size: 1.05rem; font-weight: 700; color: var(--text-strong); margin-bottom: 4px; }
.st-key-feature_select_block [data-testid="stCheckbox"] label { gap: 12px; align-items: center; min-height: 44px; cursor: pointer; }
/* Streamlit 1.59: label[data-selected] > span (hidden input) + div (the box; check svg only when ticked) + stWidgetLabel */
.st-key-feature_select_block [data-testid="stCheckbox"] label > div:not([data-testid]) {
    width: 26px; height: 26px; border-radius: 7px; border: 2px solid var(--line-strong); background: transparent;
    margin: 0; flex: none; transition: background-color .15s ease, border-color .15s ease; }
.st-key-feature_select_block [data-testid="stCheckbox"] label > div:not([data-testid]) svg {
    width: 15px; height: 12px; stroke: #ffffff; stroke-width: 2.2; fill: none; }
.st-key-feature_select_block [data-testid="stCheckbox"] label[data-selected="true"] > div:not([data-testid]) {
    background-color: var(--accent); border-color: var(--accent); }
.st-key-feature_select_block [data-testid="stCheckbox"] label:not([data-selected="true"]) > div:not([data-testid]) svg { opacity: 0; }
.st-key-feature_select_block [data-testid="stCheckbox"] label[data-focus-visible="true"] > div:not([data-testid]) {
    box-shadow: 0 0 0 3px var(--accent-soft); }
.st-key-feature_select_block [data-testid="stCheckbox"] label p { font-size: 1.2rem; font-weight: 650; color: var(--text-strong); }
.st-key-feature_select_block [data-testid="stCheckbox"] label:not([data-selected="true"]) p { color: var(--muted); }
.st-key-pass_threshold_block h3 { color: #4ade80; }
.st-key-standby_threshold_block h3 { color: #fde047; }

/* ---------- camera input (unchanged behaviour) ---------- */
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
    .app-subtitle {
        border-left: none;
        padding-left: 0;
    }
    .threshold-summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    [data-testid="stButton"] > button,
    [data-testid="stFormSubmitButton"] > button {
        min-height: 2.8rem;
    }
    .light-text {
        font-size: 32px;
    }
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)
# Optional 動態 style / 米白 theme (外觀 menu). "原版 + 深色" adds nothing here.
import ui_style  # noqa: E402
ui_style.apply()


def page_heading(title: str, description: str = "") -> None:
    # t() is applied here so callers may pass either the Chinese source or translated text.
    title = t(title)
    description_html = ""  # page descriptions are no longer shown (compact pages)
    st.markdown(
        f'<div class="page-heading"><h2>{title}</h2>{description_html}</div>',
        unsafe_allow_html=True,
    )


DISPLAY_VALUE_COLUMNS = ("grade_result", "分級結果", "狀態", "初步分級")


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Translate grade/status values and column names for display only (no-op in 中文)."""
    if not i18n.is_english():
        return df
    df = df.copy()
    for column in DISPLAY_VALUE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(lambda value: t(value) if isinstance(value, str) else value)
    return i18n.t_columns(df)


def show_data_table(height: int | None = None, rows: list[dict] | None = None) -> None:
    df = get_all_chickens_df(rows).rename(columns={"chicken_id": "chicken ID"})
    if "grade_result" in df.columns:
        # 通過 / 備用 / 不通過 right after chicken ID, coloured like the lights; the light column (GREEN / YELLOW /
        # RED) says the same thing, so it is not shown.
        df = df.drop(columns=["light_result"], errors="ignore").rename(columns={"grade_result": "分級結果"})
        df.insert(1, "分級結果", df.pop("分級結果"))
    frame = display_frame(df)
    st.dataframe(ui_style.style_grade_columns(frame, [t("分級結果")]), width="stretch", hide_index=True,
                 height=height)


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
        st.info(t("目前沒有可顯示的推論影格。"))
        return

    st.subheader(t("YOLO推論影格"))
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
    st.subheader(t("背景推論任務佇列"))
    if not jobs:
        st.info(t("目前沒有背景推論任務。"))
        return

    display_df = pd.DataFrame(jobs).drop(columns=["Arduino", "影片", "錯誤", "建立時間", "初步燈號"], errors="ignore")
    st.dataframe(ui_style.style_grade_columns(display_frame(display_df), [t("初步分級")]),
                 width="stretch", hide_index=True, height=190)

    completed_jobs = [job for job in jobs if job["狀態"] == STATUS_COMPLETED and job.get("影片")]
    if completed_jobs:
        with st.expander(t("查看已完成推論影格"), expanded=False):
            selected_label = st.selectbox(
                t("選擇推論任務"),
                [f'{job["chicken ID"]} - {job["影片"]}' for job in completed_jobs],
                key="completed_inference_preview_select",
                label_visibility="collapsed",
            )
            selected_index = [f'{job["chicken ID"]} - {job["影片"]}' for job in completed_jobs].index(selected_label)
            selected_video = Path(str(completed_jobs[selected_index]["影片"])).stem
            show_inference_sample_frames(video_ids=[selected_video])

    if has_active_jobs():
        st.info(t("背景任務執行中"))


SCALE_PORT_KEYWORDS = ("ft232", "rs485", "usb serial", "uart", "serial")


def is_arduino_port_label(label: str) -> bool:
    return "leonardo" in label.lower()


def is_scale_port_label(label: str) -> bool:
    lowered = label.lower()
    return any(keyword in lowered for keyword in SCALE_PORT_KEYWORDS) and not is_arduino_port_label(label)


def preferred_port_index(port_labels: list[str], saved_port: str, matches) -> int:
    """Selectbox index: the saved port first, else the first label `matches` accepts, else 0."""
    for index, label in enumerate(port_labels):
        if label.startswith(f"{saved_port} ") or label == saved_port:
            return index
    return next((index for index, label in enumerate(port_labels) if matches(label)), 0)


def default_arduino_port(ports=None) -> str:
    existing_port = str(st.session_state.get("arduino_serial_port", "") or "").strip()
    if existing_port:
        return existing_port
    ports = list_serial_ports() if ports is None else ports
    match = next((port.device for port in ports if is_arduino_port_label(port.label)), None)
    return match or (ports[0].device if ports else "")


def default_scale_port(ports=None) -> str:
    existing_port = str(st.session_state.get("scale_serial_port", "") or "").strip()
    if existing_port:
        return existing_port
    ports = list_serial_ports() if ports is None else ports
    return next((port.device for port in ports if is_scale_port_label(port.label)), "")


FEATURE_UNITS = {"weight_g": "g", "comb_area_cm2": "cm²", "shank_width_cm": "cm", "shank_length_cm": "cm"}


def feature_selector() -> dict[str, bool]:
    """Checkboxes: which features take part in grading. Starts from the saved thresholds (a feature is
    ticked when it was in use), or all ticked when nothing has been saved yet."""
    config = get_latest_grading_config()
    in_use = set(active_features(*config)) if config else set(FEATURE_COLUMNS)
    if config:  # threshold inputs start from the saved values (not 0)
        _, saved_pass, saved_standby = config
        for feature in FEATURE_COLUMNS:
            st.session_state.setdefault(f"{feature}_pass_threshold", float(saved_pass.get(feature) or 0.0))
            st.session_state.setdefault(f"{feature}_standby_threshold", float(saved_standby.get(feature) or 0.0))
    with st.container(border=True, key="feature_select_block"):
        st.markdown(f"<div class='feature-select-title'>{t('納入判斷的特徵')}</div>", unsafe_allow_html=True)
        chosen = {}
        for column, (feature, label) in zip(st.columns(len(FEATURE_COLUMNS)), FEATURE_COLUMNS.items()):
            key = f"use_feature_{feature}"
            st.session_state.setdefault(key, feature in in_use)
            chosen[feature] = column.checkbox(f"{t(label)} ({FEATURE_UNITS[feature]})", key=key)
    return chosen


def render_current_thresholds() -> None:
    config = get_latest_grading_config()
    if not config:
        st.info(t("尚未設定分級門檻"))
        return

    selected_features, pass_thresholds, standby_thresholds = config
    in_use = set(active_features(selected_features, pass_thresholds, standby_thresholds))
    cards = []
    for feature, label in FEATURE_COLUMNS.items():
        enabled = feature in in_use
        disabled_class = "" if enabled else " threshold-summary-disabled"
        enabled_label = t("啟用") if enabled else t("停用")
        cards.append(
            f'<div class="threshold-summary-card{disabled_class}">'
            f'<div class="threshold-summary-label">{t(label)} · {enabled_label}</div>'
            f'<div class="threshold-summary-value">'
            f'{t("通過")} {number_input_value(pass_thresholds.get(feature)):.2f}'
            f'&nbsp;／&nbsp; {t("備用")} {number_input_value(standby_thresholds.get(feature)):.2f}'
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

    scanner_col, light_col, scale_col = st.container(key="hw_row").columns(3, gap="medium")  # equal-height cards
    # Enumerate serial ports once per render; both selectboxes use the same list.
    ports = list_serial_ports()
    port_labels = [port.label for port in ports]
    port_devices = {port.label: port.device for port in ports}

    with scanner_col:
        with st.container(border=True, key="card_hardware_scanner"):
            st.subheader(t("掃碼器"))
            st.text_input(
                t("掃碼器輸入測試"),
                placeholder="C001",
                help=t("進入此頁後可直接掃描；掃描槍送出 Enter 後會自動記錄最後掃描內容。"),
                key="scanner_test_input",
                on_change=commit_scanner_input,
                args=("scanner_test_input", "scanner_last_value"),
            )
            install_n_scanner_focus_hotkey()
            install_scanner_input_autofocus()
            if st.session_state.get("scanner_last_value"):
                st.success(t("最後掃描：{value}", value=st.session_state.scanner_last_value))
            else:
                st.info(t("等待掃描輸入"))

    with light_col:
        with st.container(border=True, key="card_hardware_light"):
            st.subheader(t("燈號 COM"))
            hardware_enabled = st.checkbox(
                t("啟用 Arduino 實體燈號"),
                value=bool(st.session_state.get("arduino_light_enabled", True)),
                key="arduino_light_enabled",
            )
            default_port = st.session_state.get("arduino_serial_port", "")
            if ports:
                default_index = preferred_port_index(port_labels, default_port, is_arduino_port_label)
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
                    placeholder=t("例如 COM5 或 /dev/ttyACM0"),
                    key="hardware_arduino_serial_port_input",
                ).strip()
            st.session_state.arduino_serial_port = arduino_port

            test_col1, test_col2 = st.columns(2)
            if test_col1.button(t("測試燈號"), width="stretch", key="hardware_test_light"):
                if not arduino_port:
                    st.warning(t("請先選擇 Arduino serial port。"))
                else:
                    try:
                        replies = send_arduino_commands(arduino_port, ["TEST"])
                        st.success(t("已送出 TEST 指令。") + (t(" 回覆：{replies}", replies=", ".join(replies)) if replies else ""))
                    except Exception as exc:
                        st.error(t("Arduino 連線失敗：{error}", error=exc))
            if test_col2.button(t("關閉燈號"), width="stretch", key="hardware_light_off"):
                if not arduino_port:
                    st.warning(t("請先選擇 Arduino serial port。"))
                else:
                    try:
                        replies = send_arduino_commands(arduino_port, ["LIGHT:OFF"])
                        st.success(t("已送出 LIGHT:OFF 指令。") + (t(" 回覆：{replies}", replies=", ".join(replies)) if replies else ""))
                    except Exception as exc:
                        st.error(t("Arduino 連線失敗：{error}", error=exc))
            if st.button(t("重新連線 Arduino"), width="stretch", key="hardware_reconnect_arduino"):
                if not arduino_port:
                    st.warning(t("請先選擇 Arduino serial port。"))
                else:
                    try:
                        replies = reconnect_arduino(arduino_port)
                        st.success(t("Arduino 重新連線成功。") + (t(" 回覆：{replies}", replies=", ".join(replies)) if replies else ""))
                    except Exception as exc:
                        st.error(t("Arduino 重新連線失敗：{error}", error=exc))

            if hardware_enabled and arduino_port:
                st.info(t("目前連接：{port}", port=arduino_port))

    with scale_col:
        with st.container(border=True, key="card_hardware_scale"):
            st.subheader(t("秤重設備"))
            default_port = st.session_state.get("scale_serial_port", "")
            if ports:
                default_index = preferred_port_index(port_labels, default_port, is_scale_port_label)
                selected_label = st.selectbox(
                    t("秤重設備 serial port"),
                    port_labels,
                    index=default_index,
                    key="hardware_scale_serial_label",
                )
                scale_port = port_devices[selected_label]
            else:
                scale_port = st.text_input(
                    t("秤重設備 serial port"),
                    value=default_port,
                    placeholder=t("例如 COM6 或 /dev/ttyUSB0"),
                    key="hardware_scale_serial_port_input",
                ).strip()
            st.session_state.scale_serial_port = scale_port

            scale_test_col1, scale_test_col2 = st.columns(2)
            if scale_test_col1.button(t("測試讀值"), width="stretch", key="hardware_test_scale"):
                if not scale_port:
                    st.warning(t("請先選擇秤重設備 serial port。"))
                else:
                    try:
                        monitor = scale_monitor(scale_port)
                        time.sleep(2.0)  # collect 2 s of readings
                        result = monitor.snapshot(2.0)
                        if result.error:
                            raise RuntimeError(result.error)
                        average_g = average_recent_samples(result.samples)
                        if average_g is None:
                            detail = t(" 原始封包：{packets}", packets=", ".join(result.raw_packets)) if result.raw_packets else ""
                            errors = t(" 錯誤：{errors}", errors="; ".join(result.errors)) if result.errors else ""
                            st.warning(t("2 秒內沒有取得有效重量。") + f"{detail}{errors}")
                        else:
                            st.success(
                                t(
                                    "平均 {average} g · 最新 {latest} g · {count} 筆",
                                    average=f"{average_g:.2f}",
                                    latest=f"{result.latest_g:.2f}",
                                    count=len(result.samples),
                                )
                            )
                            if result.raw_packets:
                                st.caption(t("原始封包：{packets}", packets=", ".join(result.raw_packets[-5:])))
                            if result.errors:
                                st.warning("；".join(result.errors))
                    except Exception as exc:
                        st.error(t("秤重設備連線失敗：{error}", error=exc))
            if scale_test_col2.button(t("清除紀錄"), width="stretch", key="hardware_clear_scale_history"):
                st.session_state.scale_recent_samples = []
                st.success(t("已清除秤重即時紀錄。"))

            if scale_port:
                st.info(t("目前連接：{port}", port=scale_port))


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
            t("USB 掃描槍掃描 / 手動輸入 chicken ID"),
            placeholder="C001",
            help=t("此欄位會自動聚焦；掃描槍送出 Enter 後會直接確認 chicken ID。"),
            key=input_key,
            on_change=commit_scanner_input,
            args=(input_key, confirmed_key),
        )
        if auto_focus:
            install_scanner_input_autofocus()
        confirmed_id = st.session_state.get(confirmed_key, "").strip()
        if confirmed_id and show_confirmation:
            st.success(t("已確認 chicken ID：{chicken_id}", chicken_id=confirmed_id))
        return confirmed_id

    with st.form(f"{key_prefix}_manual_id_form_{input_version}", clear_on_submit=clear_on_submit):
        manual_id = st.text_input(
            t("USB 掃描槍掃描 / 手動輸入 chicken ID"),
            placeholder="C001",
            help=t("按第一顆實體按鈕後，使用 USB 掃描槍掃描 chicken ID；掃描槍送出 Enter 或按「確定」即可確認。"),
            key=input_key,
        )
        manual_confirmed = st.form_submit_button(t("確定"), width="stretch")
    if auto_focus:
        install_scanner_input_autofocus()
    if manual_confirmed:
        st.session_state[confirmed_key] = manual_id.strip()

    confirmed_id = st.session_state.get(confirmed_key, "").strip()
    if confirmed_id and show_confirmation:
        st.success(t("已確認 chicken ID：{chicken_id}", chicken_id=confirmed_id))
    return confirmed_id


def camera_adjustment_page() -> None:
    from dual_camera_ui import adjustment_page
    adjustment_page(page_heading)


def scale_calibration_section() -> None:
    from dual_camera_ui import calibration_page
    calibration_page(page_heading, install_n_click_button_hotkey)


@st.fragment(run_every=0.5)
def live_scale_panel(scale_port: str, window_seconds: float = 2.0) -> None:
    st.subheader(t("即時秤重"))
    if not scale_port:
        st.info(t("尚未設定秤重設備"))
        st.session_state.scale_recent_samples = []
        return

    phase = st.session_state.get("scale_collection_phase", "collecting")
    if phase == "finalizing":
        st.info(t("拍攝中…"))
        return
    if phase == "stopped":
        frozen_weight = st.session_state.get("scale_final_weight_g")
        sample_count = int(st.session_state.get("scale_final_sample_count", 0) or 0)
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(t("本次最終重量 (g)"), "-" if frozen_weight is None else f"{float(frozen_weight):.2f}")
        metric_col2.metric(t("最終平均樣本數"), str(sample_count))
        st.caption(t("重量已鎖定"))
        return

    try:
        # A background thread keeps reading the port; this refresh only takes its latest readings, so it
        # never blocks the page (switching pages used to wait for a 0.35 s serial read).
        result = scale_monitor(scale_port).snapshot(window_seconds)
        if result.error:
            raise RuntimeError(result.error)
        history = result.samples
        st.session_state.scale_recent_samples = history
        average_g = average_recent_samples(history)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(t("最新重量 (g)"), "-" if result.latest_g is None else f"{result.latest_g:.2f}")
        metric_col2.metric(
            t("{seconds} 秒平均 (g)", seconds=f"{window_seconds:.0f}"),
            "-" if average_g is None else f"{average_g:.2f}",
        )
        metric_col3.metric(t("有效樣本數"), str(len(history)))
        if result.raw_packets:
            st.caption(t("最近封包：{packets}", packets=", ".join(result.raw_packets[-3:])))
        if result.errors:
            st.warning("；".join(result.errors))
    except Exception as exc:
        st.session_state.scale_recent_samples = []
        st.error(t("秤重設備讀取失敗：{error}", error=exc))


def phenotype_entry_page() -> None:
    from dual_camera_ui import phenotype_page
    phenotype_page(globals())

def data_management_page() -> None:
    page_heading("雞隻資料管理", "集中檢視所有表型資料；新增、修改與刪除操作收納於下方。")

    chickens = get_all_chickens()
    toolbar = st.container(key="data_toolbar")
    if ui_style.SHOW_DEMO_TOOLS:
        total_col, sample_col, import_col, export_col = toolbar.columns([0.8, 1, 1.25, 1], vertical_alignment="center")
        with sample_col:
            if st.button(t("建立 20 筆範例資料"), width="stretch"):
                inserted = create_sample_data()
                st.session_state.data_page_flash = t("已新增 {count} 筆範例資料。", count=inserted)
                st.rerun()
    else:  # product: no demo-data button
        total_col, import_col, export_col = toolbar.columns([0.8, 1.25, 1], vertical_alignment="center")
    total_col.metric(t("資料筆數"), len(chickens))
    with import_col:
        # The uploader keeps returning the same file on every rerun, so give it a versioned key and bump the
        # version after a successful import — otherwise import → rerun → import … loops forever.
        uploader_version = int(st.session_state.get("csv_uploader_version", 0))
        csv_file = st.file_uploader(t("匯入 CSV"), type=["csv"], label_visibility="collapsed",
                                    key=f"csv_uploader_{uploader_version}")
        if csv_file is not None:
            try:
                inserted, updated = import_csv(csv_file)
                st.session_state.csv_uploader_version = uploader_version + 1
                st.session_state.data_page_flash = t("CSV 匯入完成：新增 {inserted} 筆，更新 {updated} 筆。",
                                                     inserted=inserted, updated=updated)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with export_col:
        st.download_button(
            t("匯出 CSV"),
            export_csv_bytes,  # built only when the button is clicked
            file_name="chicken_data.csv",
            mime="text/csv",
            width="stretch",
        )
    flash = st.session_state.pop("data_page_flash", None)
    if flash:
        st.success(flash)

    show_data_table(height=500, rows=chickens)

    with st.expander(t("新增雞隻"), expanded=False):
        with st.form("add_chicken_form", clear_on_submit=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            chicken_id = col1.text_input("chicken ID", placeholder="C001")
            weight_g = col2.number_input(t("體重 (g)"), min_value=0.0, step=10.0)
            comb_area_cm2 = col3.number_input(t("雞冠面積 (cm²)"), min_value=0.0, value=0.0, step=0.5)
            shank_width_cm = col4.number_input(t("腳脛寬度 (cm)"), min_value=0.0, value=0.0, step=0.1)
            shank_length_cm = col5.number_input(t("腳脛長度 (cm)"), min_value=0.0, value=0.0, step=0.1)
            submitted = st.form_submit_button(t("新增資料"), width="stretch")
            if submitted:
                try:
                    add_chicken(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)
                    st.success(t("已新增雞隻：{chicken_id}", chicken_id=chicken_id))
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with st.expander(t("修改或刪除雞隻"), expanded=False):
        if chickens:
            chicken_ids = [row["chicken_id"] for row in chickens]
            selected_id = st.selectbox(t("選擇 chicken ID"), chicken_ids, key="manage_chicken_id")
            selected = get_chicken(selected_id)
            if selected:
                with st.form("edit_chicken_form"):
                    col1, col2, col3, col4 = st.columns(4)
                    edit_weight = col1.number_input(t("體重 (g)"), value=number_input_value(selected["weight_g"]), step=10.0)
                    edit_comb = col2.number_input(t("雞冠面積 (cm²)"), value=number_input_value(selected["comb_area_cm2"]), step=0.5)
                    edit_shank = col3.number_input(
                        t("腳脛寬度 (cm)"),
                        value=number_input_value(selected.get("shank_width_cm")),
                        step=0.1,
                    )
                    edit_shank_length = col4.number_input(
                        t("腳脛長度 (cm)"),
                        value=number_input_value(selected.get("shank_length_cm")),
                        step=0.1,
                    )
                    save_col, delete_col = st.columns(2)
                    save = save_col.form_submit_button(t("儲存修改"), width="stretch")
                    remove = delete_col.form_submit_button(t("刪除此雞隻"), width="stretch")
                    if save:
                        update_chicken(selected_id, edit_weight, edit_comb, edit_shank, edit_shank_length)
                        st.success(t("已更新雞隻：{chicken_id}", chicken_id=selected_id))
                        st.rerun()
                    if remove:
                        delete_chicken(selected_id)
                        st.warning(t("已刪除雞隻：{chicken_id}", chicken_id=selected_id))
                        st.rerun()
        else:
            st.info(t("目前沒有可修改的雞隻資料。"))


def grading_page() -> None:
    page_heading(
        "分級門檻設定",
        "設定通過與備用門檻後，系統會更新所有雞隻的通過、備用及不通過結果。",
    )
    # Reserved at the top, filled at the end of the page run: a click on 確認設定門檻 saves first, so the
    # cards already show the new thresholds (they used to lag one click behind).
    thresholds_slot = st.container()
    use_feature = feature_selector()

    pass_thresholds = {}
    standby_thresholds = {}
    feature_items = list(FEATURE_COLUMNS.items())
    pass_section, standby_section = st.columns(2, vertical_alignment="top")
    with pass_section:
        with st.container(border=True, key="pass_threshold_block"):
            st.markdown(f"<h3 style='text-align: center; margin:0 0 .35rem;'>{t('通過門檻')}</h3>", unsafe_allow_html=True)
            for row_start in range(0, len(feature_items), 2):
                row_cols = st.columns(2)
                for col, (feature, label) in zip(row_cols, feature_items[row_start : row_start + 2]):
                    pass_thresholds[feature] = col.number_input(
                        t(label),
                        min_value=0.0,
                        step=0.1,
                        key=f"{feature}_pass_threshold",
                        disabled=not use_feature[feature],
                    )

    with standby_section:
        with st.container(border=True, key="standby_threshold_block"):
            st.markdown(f"<h3 style='text-align: center; margin:0 0 .35rem;'>{t('備用門檻')}</h3>", unsafe_allow_html=True)
            for row_start in range(0, len(feature_items), 2):
                row_cols = st.columns(2)
                for col, (feature, label) in zip(row_cols, feature_items[row_start : row_start + 2]):
                    standby_thresholds[feature] = col.number_input(
                        t(label),
                        min_value=0.0,
                        step=0.1,
                        key=f"{feature}_standby_threshold",
                        disabled=not use_feature[feature],
                    )

    if st.button(t("確認設定門檻"), type="primary", width="stretch"):
        try:
            selected_features = [feature for feature in FEATURE_COLUMNS if use_feature[feature]]
            if not selected_features:
                raise ValueError(t("請至少勾選一個分級特徵。"))
            for feature in selected_features:
                if not pass_thresholds[feature] and not standby_thresholds[feature]:
                    raise ValueError(t("{feature} 已勾選，請設定大於 0 的門檻，或取消勾選。",
                                       feature=t(FEATURE_COLUMNS[feature])))
            validate_grading_config(selected_features, pass_thresholds, standby_thresholds)
            enabled_features = active_features(selected_features, pass_thresholds, standby_thresholds)
            # Config validated once above; grade every chicken, then write all results in one transaction.
            rows = []
            for chicken in get_all_chickens():
                grade_result, light_result, below_pass_features, below_standby_features = grade_details(
                    chicken, enabled_features, pass_thresholds, standby_thresholds
                )
                rows.append(
                    {
                        "chicken ID": chicken["chicken_id"],
                        "grade_result": grade_result,
                        "light_result": light_result,
                        "below_pass_features": ", ".join(below_pass_features),
                        "below_standby_features": ", ".join(below_standby_features),
                    }
                )
            update_grading_results(
                [(row["chicken ID"], row["grade_result"], row["light_result"]) for row in rows],
                selected_features, pass_thresholds, standby_thresholds,
            )
            st.success(t("分級完成，已更新資料庫。"))
            summary = summarize_results(rows)
            metric_cols = st.container(key="grading_summary").columns(7)
            labels = [
                ("總筆數", summary["total"]),
                ("通過筆數", summary["pass_count"]),
                ("備用筆數", summary["standby_count"]),
                ("不通過筆數", summary["fail_count"]),
                ("通過比例", f"{summary['pass_rate']:.2f}%"),
                ("備用比例", f"{summary['standby_rate']:.2f}%"),
                ("不通過比例", f"{summary['fail_rate']:.2f}%"),
            ]
            for col, (label, val) in zip(metric_cols, labels):
                col.markdown(
                    f'<div class="summary-stat">'
                    f'<div class="summary-stat-label">{t(label)}</div>'
                    f'<div class="summary-stat-value">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with st.expander(t("查看本次分級明細"), expanded=False):
                details = pd.DataFrame(rows).drop(columns=["light_result"]).rename(columns={"grade_result": "分級結果"})
                st.dataframe(ui_style.style_grade_columns(display_frame(details), [t("分級結果")]),
                             width="stretch", hide_index=True, height=260)
        except Exception as exc:
            st.error(str(exc))

    with st.expander(t("查看目前雞隻資料"), expanded=False):
        show_data_table(height=320)
    with thresholds_slot:
        render_current_thresholds()


def render_light(grade_result: str | None) -> None:
    """Light circle + grade word only (通過 / 備用 / 不通過) — no explanatory text."""
    css_class, message = {
        GRADE_PASS: ("light-green", GRADE_PASS),
        GRADE_STANDBY: ("light-yellow", GRADE_STANDBY),
        GRADE_FAIL: ("light-red", GRADE_FAIL),
    }.get(grade_result, ("light-gray", "尚未完成分級"))
    st.markdown(
        f"""
        <div class="light-panel">
            <div class="light-circle {css_class}"></div>
            <div class="light-text">{t(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def execute_final_light_decision(query_id: str) -> bool:
    chicken = get_chicken(query_id)
    if not chicken:
        st.warning(t("找不到此 chicken ID。"))
        return False
    try:
        grade_result, _ = grade_and_store_with_latest_config(chicken)
        if grade_result is None:
            st.warning(t("目前資料庫沒有已儲存的分級門檻，請先在「分級門檻設定」執行一次分級。"))
            grade_result = chicken["grade_result"]
    except Exception as exc:
        st.error(t("最終篩選失敗：{error}", error=exc))
        return False

    render_light(grade_result)

    if bool(st.session_state.get("arduino_light_enabled", True)):
        arduino_port = default_arduino_port()
        if not arduino_port:
            st.warning(t("已啟用實體燈號，但尚未選擇 Arduino serial port。"))
        else:
            try:
                light_for(arduino_port, commands_for_grade(grade_result), FINAL_LIGHT_SECONDS)
            except Exception as exc:
                st.error(t("Arduino 燈號控制失敗：{error}", error=exc))

    return True



def final_scan_page() -> None:
    page_heading(
        "最終種雞判斷",
        "再次掃描既有 QR code，立即顯示最終分級並控制綠燈、黃燈或紅燈。",
    )

    # Only the light + grade word is shown here (no confirmation/explanation text, no data row).
    query_id = resolve_qrcode_input("final_scan", auto_focus=True, scanner_mode=True, show_confirmation=False)
    if not query_id:
        st.session_state.final_scan_last_processed_token = ""
        last_id = st.session_state.get("final_scan_last_completed_id")
        if last_id:
            chicken = get_chicken(last_id)
            if chicken:
                render_light(chicken.get("grade_result"))
        return

    final_scan_token = f"{query_id}|{repr(get_latest_grading_config())}"
    if st.session_state.get("final_scan_last_processed_token") != final_scan_token:
        if execute_final_light_decision(query_id):
            mark_final_scanned(query_id)
            st.session_state.final_scan_last_completed_id = query_id
            st.session_state.final_scan_confirmed_manual_id = ""
            st.session_state.final_scan_input_version = int(st.session_state.get("final_scan_input_version", 0)) + 1
            st.session_state.final_scan_last_processed_token = ""
    else:
        chicken = get_chicken(query_id)
        if chicken:
            render_light(chicken.get("grade_result"))


# Sidebar tab names. The page keys (values of the navigation control, used by session state and tests) stay
# as they were; only what the sidebar shows changed.
NAV_LABELS = {
    "掃描 QR code 與表型輸入": "表型數據辨識",
    "再次掃描與亮燈判定": "最終種雞判斷",
    "資料與模型維護": "模型維護與更新",
}
DEFAULT_PAGE = "雞隻資料管理"  # the page a new session opens on (the first tab)

# Stroke icons (24×24) per page, drawn in the button's text colour through a CSS mask.
NAV_ICONS = {
    "硬體連接設定": '<path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0zM12 17v4"/>',
    "雞隻資料管理": '<ellipse cx="12" cy="5.5" rx="7" ry="2.5"/><path d="M5 5.5v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6'
                  'M5 11.5v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6"/>',
    "分級門檻設定": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/>'
                  '<circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "相機調整": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "比例尺校正": '<path d="M3 16 16 3l5 5L8 21zM7 12l2 2M10 9l2 2M13 6l2 2"/>',
    "掃描 QR code 與表型輸入": '<path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M8 8h3v3H8zM13 13h3v3h-3zM13 8h3M8 13v3"/>',
    "再次掃描與亮燈判定": '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z"/>',
    "模型品質監控": '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
    "案例審核": '<path d="M9 4h6v3H9zM7 5H5v16h14V5h-2M8.5 14l2.5 2.5 4.5-5"/>',
    "資料與模型維護": '<path d="M21 8 12 3 3 8l9 5zM3 8v8l9 5 9-5V8M12 13v8"/>',
}


def nav_icon_css(page_names) -> str:
    """One icon per sidebar tab, by position (the tabs keep a fixed order)."""
    from urllib.parse import quote

    rules = []
    for position, name in enumerate(page_names, start=1):
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
               'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">' + NAV_ICONS.get(name, "") + "</svg>")
        rules.append(f'.st-key-main_page_navigation_segmented [role="radiogroup"] button:nth-of-type({position})'
                     f'{{--nav-icon:url("data:image/svg+xml,{quote(svg)}")}}')
    return "<style>" + "\n".join(rules) + "</style>"


def main() -> None:
    _, language_column, appearance_column = st.columns([7, 1.1, 1], vertical_alignment="center")
    with language_column:
        ui_style.language_switch()
    with appearance_column:
        ui_style.controls()

    from product_review_ui import model_maintenance_page, quality_dashboard_page, review_cases_page

    pages = {
        "雞隻資料管理": data_management_page,
        "分級門檻設定": grading_page,
        "相機調整": camera_adjustment_page,
        "比例尺校正": scale_calibration_section,
        "掃描 QR code 與表型輸入": phenotype_entry_page,
        "再次掃描與亮燈判定": final_scan_page,
        "硬體連接設定": hardware_settings_page,
        "模型品質監控": lambda: quality_dashboard_page(page_heading),
        "案例審核": lambda: review_cases_page(page_heading),
        "資料與模型維護": lambda: model_maintenance_page(page_heading),
    }
    page_names = list(pages.keys())
    if "main_page_navigation" not in st.session_state:
        st.session_state.main_page_navigation = DEFAULT_PAGE
    # Option values stay the page keys; labels are resolved now (a dict lookup, so the formatter does not
    # depend on the language state when it is called later, e.g. by AppTest).
    page_labels = {name: t(NAV_LABELS.get(name, name)) for name in page_names}
    # The P hotkey matches nav buttons by their visible text, so pass the displayed names.
    install_p_page_cycle_hotkey(list(page_labels.values()), page_labels.get(st.session_state.main_page_navigation, ""))
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="app-logo">{app_icon_html()}</div>
                <div>
                    <div class="sidebar-brand-name">{t("紅羽土雞")}</div>
                    <div class="sidebar-brand-sub">{t("表型分級系統")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(nav_icon_css(page_names), unsafe_allow_html=True)
        selected_page = st.segmented_control(
            t("功能選擇"),
            page_names,
            format_func=page_labels.__getitem__,
            selection_mode="single",
            default=st.session_state.main_page_navigation,
            label_visibility="collapsed",
            key="main_page_navigation_segmented",
        )
        st.markdown(f'<div class="sidebar-foot">{t("快捷鍵 P：切換到下一頁")}</div>', unsafe_allow_html=True)
    if selected_page is None:
        selected_page = st.session_state.main_page_navigation
    st.session_state.main_page_navigation = selected_page
    pages[selected_page]()


if __name__ == "__main__":
    main()


