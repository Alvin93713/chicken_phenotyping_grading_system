"""Optional look for the grading station: 原版/動態 style and 深色/米白 theme.

"原版 + 深色" is the default and renders exactly like the app did before this
module existed (no extra CSS, the theme from .streamlit/config.toml).  The
other choices only add CSS / swap the Streamlit theme; no feature changes.

The choice is stored in data/ui_style.json so it survives restarts.  The
Streamlit theme is a server-wide option, so every browser connected to this
station follows the same choice.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import streamlit as st
from streamlit import config as _st_config

from i18n import LANG_EN, LANG_ZH, LANGUAGES, t

STYLE_CLASSIC = "原版"
STYLE_MOTION = "動態"
THEME_DARK = "深色"
THEME_CREAM = "米白"
STYLES = (STYLE_CLASSIC, STYLE_MOTION)
THEMES = (THEME_DARK, THEME_CREAM)
DEFAULT_PREFS = {"style": STYLE_CLASSIC, "theme": THEME_DARK, "lang": LANG_ZH}
# What the 外觀 menu shows for each stored value (stored values stay the same, so saved preferences keep working).
OPTION_LABELS = {STYLE_CLASSIC: "靜態", STYLE_MOTION: "動態", THEME_DARK: "深色", THEME_CREAM: "淺色"}


def _option_label(value: str) -> str:
    return t(OPTION_LABELS.get(value, value))

# Demo-only buttons (「建立 20 筆範例資料」「載入五筆真實測試案例」) are hidden in the product; set
# CHICKEN_SHOW_DEMO_TOOLS=1 to show them again for a demonstration.
SHOW_DEMO_TOOLS = os.environ.get("CHICKEN_SHOW_DEMO_TOOLS", "").strip() == "1"
_DEFAULT_PREFS_PATH = Path(__file__).resolve().parent.parent / "data" / "ui_style.json"

# Identical to .streamlit/config.toml, so "深色" never changes the original look.
STREAMLIT_THEMES = {
    THEME_DARK: {
        "base": "dark",
        "primaryColor": "#4f8cff",
        "backgroundColor": "#0b1018",
        "secondaryBackgroundColor": "#151d2a",
        "textColor": "#e7ecf3",
        "borderColor": "#263246",
        "dataframeBorderColor": "#223047",
        "dataframeHeaderBackgroundColor": "#111927",
        "greenColor": "#22c55e",
        "yellowColor": "#facc15",
        "redColor": "#ef4444",
        "blueColor": "#4f8cff",
        "linkColor": "#7aa7ff",
    },
    THEME_CREAM: {
        "base": "light",
        "primaryColor": "#2f6fe4",
        "backgroundColor": "#f6f1e7",
        "secondaryBackgroundColor": "#efe8da",
        "textColor": "#2b261f",
        "borderColor": "#d9cfbb",
        "dataframeBorderColor": "#e2d9c7",
        "dataframeHeaderBackgroundColor": "#efe8da",
        "greenColor": "#16a34a",
        "yellowColor": "#ca8a04",
        "redColor": "#dc2626",
        "blueColor": "#2f6fe4",
        "linkColor": "#2f5fc4",
    },
}


# ---------------------------------------------------------------- preferences
def prefs_path() -> Path:
    return Path(os.environ.get("CHICKEN_UI_PREFS_PATH") or _DEFAULT_PREFS_PATH)


def load_prefs() -> dict:
    prefs = dict(DEFAULT_PREFS)
    try:
        saved = json.loads(prefs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return prefs
    if saved.get("style") in STYLES:
        prefs["style"] = saved["style"]
    if saved.get("theme") in THEMES:
        prefs["theme"] = saved["theme"]
    if saved.get("lang") in LANGUAGES:
        prefs["lang"] = saved["lang"]
    return prefs


def save_prefs(prefs: dict) -> None:
    path = prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")


def current() -> dict:
    return st.session_state.get("_ui_prefs") or load_prefs()


def is_motion() -> bool:
    return current()["style"] == STYLE_MOTION


# ---------------------------------------------------------------- theme + CSS
def _ensure_streamlit_theme(theme: str) -> bool:
    """Point Streamlit's theme options at `theme`. Returns True if anything changed.

    The theme is sent to the browser at the start of each script run, so a
    change only shows after the next rerun (the caller reruns once).
    """
    changed = False
    for name, value in STREAMLIT_THEMES[theme].items():
        key = f"theme.{name}"
        if _st_config.get_option(key) != value:
            _st_config.set_option(key, value, "ui_style")
            changed = True
    return changed


def apply() -> None:
    """Call once per run, right after the base CSS. Adds CSS for the chosen look."""
    prefs = load_prefs()
    st.session_state["_ui_prefs"] = prefs
    st.session_state["_ui_lang"] = prefs["lang"]
    if _ensure_streamlit_theme(prefs["theme"]):
        st.rerun()
    css = ""
    if prefs["style"] == STYLE_MOTION:
        css += MOTION_CSS
    if prefs["theme"] == THEME_CREAM:
        css += CREAM_CSS  # last, so its palette (incl. --mo-* result-card colours) wins over the dark defaults
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _on_change() -> None:
    previous = current()
    # Clicking the already-selected option deselects it (None): keep the previous choice.
    style = st.session_state.get("ui_style_choice") or previous["style"]
    theme = st.session_state.get("ui_theme_choice") or previous["theme"]
    st.session_state.ui_style_choice = style
    st.session_state.ui_theme_choice = theme
    save_prefs({"style": style, "theme": theme, "lang": previous["lang"]})


def _restore_classic() -> None:
    st.session_state.ui_style_choice = STYLE_CLASSIC
    st.session_state.ui_theme_choice = THEME_DARK
    save_prefs({**DEFAULT_PREFS, "lang": current()["lang"]})


def controls() -> None:
    """Small 外觀 popover: style and theme switches, plus a one-click restore."""
    prefs = current()
    # Streamlit's transparent top header (≈60px) sits over the masthead and swallows clicks
    # on this button; let clicks pass through it except on its own toolbar buttons.
    st.markdown(
        "<style>header[data-testid='stHeader'],header[data-testid='stHeader'] *{pointer-events:none}"
        "header[data-testid='stHeader'] button,header[data-testid='stHeader'] button *,"
        "header[data-testid='stHeader'] a,header[data-testid='stHeader'] a *{pointer-events:auto}</style>",
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("ui_style_choice", prefs["style"])
    st.session_state.setdefault("ui_theme_choice", prefs["theme"])
    with st.popover(t("外觀"), width="stretch"):
        st.segmented_control(t("介面樣式"), STYLES, key="ui_style_choice", on_change=_on_change, format_func=_option_label)
        st.segmented_control(t("色彩主題"), THEMES, key="ui_theme_choice", on_change=_on_change, format_func=_option_label)
        is_original = prefs["style"] == STYLE_CLASSIC and prefs["theme"] == THEME_DARK
        st.button(t("恢復預設外觀"), key="ui_restore_classic", width="stretch",
                  disabled=is_original, on_click=_restore_classic)


def _on_language_change() -> None:
    previous = current()
    lang = st.session_state.get("ui_lang_choice") or previous["lang"]
    st.session_state.ui_lang_choice = lang
    st.session_state["_ui_lang"] = lang
    save_prefs({**previous, "lang": lang})


def language_switch() -> None:
    """中文 / English switch shown next to the 外觀 button."""
    st.session_state.setdefault("ui_lang_choice", current()["lang"])
    st.segmented_control(
        "Language",
        LANGUAGES,
        key="ui_lang_choice",
        on_change=_on_language_change,
        format_func=lambda value: "EN" if value == LANG_EN else value,
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------- motion HTML
_GRADE_CLASS = {"通過": "mo-pass", "備用": "mo-standby", "不通過": "mo-fail"}
_MAX_DELAY_STEPS = 40  # 50 ms each


def _odometer(text: str, start_step: int) -> str:
    parts = []
    step = start_step
    for char in text:
        if char.isdigit():
            digits = "".join(f"<span>{d}</span>" for d in range(10))
            delay = min(step, _MAX_DELAY_STEPS)
            parts.append(f'<span class="mo-col"><span class="mo-strip mo-n{char} mo-dl{delay}">{digits}</span></span>')
            step += 1
        else:
            parts.append(f'<span class="mo-sep">{html.escape(char)}</span>')
    return f'<span class="mo-odo">{"".join(parts)}</span>'


def _split_label(label: str) -> tuple[str, str]:
    """'雞冠面積 (cm²)' -> ('雞冠面積', 'cm²')."""
    if label.endswith(")") and " (" in label:
        name, unit = label[:-1].split(" (", 1)
        return name, unit
    return label, ""


def metric_cards_html(metrics, extra_class: str = "") -> str:
    """metrics: iterable of (label, display_text, tone) where tone is weight/comb/shank."""
    cards = []
    for index, (label, display, tone) in enumerate(metrics):
        name, unit = _split_label(label)
        number = _odometer(display, 3 + index * 3) if display not in ("-", "") else '<span class="mo-odo">-</span>'
        cards.append(
            f'<div class="mo-card mo-cd{index} mo-{tone}">'
            f'<div class="mo-label"><span class="mo-dot"></span>{html.escape(name)}</div>'
            f'<div class="mo-value">{number}<span class="mo-unit">{html.escape(unit)}</span></div>'
            f'<div class="mo-track"><span class="mo-bar"></span></div></div>'
        )
    return f'<div class="mo-cards {extra_class}">{"".join(cards)}</div>'


def result_band_html(chicken_id: str, grade: str, metrics) -> str:
    grade_class = _GRADE_CLASS.get(grade, "mo-neutral")
    badge = f'<span class="mo-badge {grade_class}">{html.escape(t(grade) if grade else "-")}</span>'
    return (
        '<div class="mo-band">'
        f'<div class="mo-head"><span class="mo-eyebrow">{html.escape(t("最近完成"))}</span>'
        f'<span class="mo-id">{html.escape(str(chicken_id))}</span>{badge}</div>'
        f'{metric_cards_html(metrics)}</div>'
    )


# --- measurement progress ring (phenotype page, while a chicken is being measured) ---------------------
# Stages and the percentage reached when each one is done. The ring shows the last finished stage and
# highlights the one running now.
RING_STAGES = (("鎖定體重", 20), ("拍攝雙相機", 40), ("推論雞冠面積", 60), ("推論腳脛長寬", 80), ("寫入並分級", 100))
_RING_CIRCUMFERENCE = 578.05  # 2·π·92

RING_CSS = """
@property --ring-p { syntax: '<number>'; inherits: true; initial-value: 0; }
@property --ring-pi { syntax: '<integer>'; inherits: true; initial-value: 0; }
.pr-band { background: var(--ring-band, #0e141e); border: 1px solid var(--ring-border, #222d3f); border-radius: 18px;
    padding: 14px 18px 16px; display: flex; flex-direction: column; gap: 12px; color: var(--ring-text, #e7ecf4);
    font-family: Inter, system-ui, sans-serif; }
.pr-head { display: flex; align-items: center; gap: 12px; }
.pr-eyebrow { font-size: .78rem; color: var(--ring-muted, #93a1b5); letter-spacing: .16em; }
.pr-id { font-family: 'Space Grotesk', Inter, system-ui, sans-serif; font-size: 1.6rem; font-weight: 700;
    color: var(--ring-num, #ffffff); }
.pr-body { display: flex; align-items: center; justify-content: center; gap: 28px; flex-wrap: wrap; }
/* --ring-p / --ring-pi are set inline (start value) and then driven frame by frame by RING_DRIVER_JS. */
.pr-ring { position: relative; width: 208px; height: 208px;
    flex: none; border-radius: 50%;
    background: var(--ring-plate, radial-gradient(circle at 35% 30%, #18222f, #0f1620 70%));
    box-shadow: var(--ring-plate-shadow, 0 16px 36px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05)); }
.pr-ring svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.pr-track { fill: none; stroke: var(--ring-track, #1c2636); stroke-width: 16; }
.pr-arc { fill: none; stroke: url(#pr-grad); stroke-width: 16; stroke-linecap: round; stroke-dasharray: 578.05;
    stroke-dashoffset: calc(578.05px * (1 - var(--ring-p) / 100)); filter: var(--ring-glow, drop-shadow(0 0 7px rgba(79,140,255,.45))); }
.pr-core { position: absolute; inset: 32px; border-radius: 50%; background: var(--ring-core, #121a26);
    box-shadow: var(--ring-core-shadow, inset 0 6px 16px rgba(0,0,0,.5)); display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 2px; }
.pr-num { font-family: 'Space Grotesk', Inter, system-ui, sans-serif; font-size: 48px; font-weight: 600; line-height: 1;
    letter-spacing: -.02em; color: var(--ring-num, #ffffff); font-variant-numeric: tabular-nums; counter-reset: pct var(--ring-pi); }
.pr-num::before { content: counter(pct); }
.pr-num small { font-size: 20px; font-weight: 500; color: var(--ring-muted, #93a1b5); margin-left: 2px; }
.pr-now { font-size: 12px; color: var(--ring-muted, #93a1b5); }
/* While Streamlit re-runs the page it keeps the previous run's elements (data-stale) until replaced. An
   outdated ring (e.g. the capture ring) is hidden only while a current ring is on screen, so there is never
   a second ring and never a moment without one. */
body:has([data-stale="false"] .pr-band) [data-stale="true"] .pr-band { display: none; }
"""


def progress_ring_html(chicken_id: str, pct: int, previous_pct: int, job: str = "", continue_from: str = "") -> str:
    """Ring for a measurement in progress. `pct` = the last finished stage (0–100, multiples of 20); the
    browser (RING_DRIVER_JS) animates the shown number from `previous_pct` and creeps toward the next stage
    on its own, so this HTML only changes when the stage changes. `continue_from` names an earlier ring (the
    capture ring) whose shown value this one starts from, so the number never steps back."""
    pct = max(0, min(100, int(pct)))
    previous_pct = max(0, min(pct, int(previous_pct)))
    current = next((label for label, done_at in RING_STAGES if done_at > pct), RING_STAGES[-1][0])
    return (
        f'<style>{RING_CSS}</style>'
        '<div class="pr-band" role="status">'
        f'<div class="pr-head"><span class="pr-eyebrow">{html.escape(t("計算中"))}</span>'
        f'<span class="pr-id">{html.escape(str(chicken_id))}</span></div>'
        '<div class="pr-body">'
        # style stays identical for a job (only data-floor changes), so a stage update never resets the values
        # the browser is animating; until the driver's first frame the ring shows its start value.
        f'<div class="pr-ring" style="--ring-p:{previous_pct};--ring-pi:{previous_pct}" '
        f'data-job="{html.escape(str(job))}" data-continue="{html.escape(str(continue_from))}" data-floor="{pct}" '
        f'data-from="{previous_pct}" role="progressbar" '
        f'aria-label="{html.escape(t("計算進度"))}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{pct}">'
        '<svg viewBox="0 0 236 236" aria-hidden="true"><defs><linearGradient id="pr-grad" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" style="stop-color: var(--ring-accent, #4f8cff)"></stop>'
        '<stop offset="1" style="stop-color: var(--ring-accent-2, #6fd3ff)"></stop>'
        '</linearGradient></defs><circle class="pr-track" cx="118" cy="118" r="92"></circle>'
        '<circle class="pr-arc" cx="118" cy="118" r="92" transform="rotate(-90 118 118)"></circle></svg>'
        f'<div class="pr-core"><span class="pr-num"><small>%</small></span>'
        f'<span class="pr-now">{html.escape(t(current))}</span></div></div>'
        '</div></div>'
    )


# Runs in the page (not per Streamlit rerun): animates every .pr-ring at display frame rate. The server only
# sets data-floor when a stage finishes; between stages the number creeps toward the next stage (slower and
# slower, never reaching it), and a new floor is glided to from wherever the number is. Inference speed
# therefore only changes where the number waits, never how smoothly it moves.
RING_DRIVER_JS = r"""
(() => {
  const VERSION = 11;
  if (window.__chickenRingDriver === VERSION) return;
  window.__chickenRingDriver = VERSION;
  const states = (window.__chickenRingStates = window.__chickenRingStates || {});
  const CREEP_MS = 4000;   // time constant of the creep inside a stage
  const FOLLOW_MS = 180;   // how quickly the shown value follows its target (glide)
  const paint = (el, value) => {
    el.style.setProperty('--ring-p', value.toFixed(2));
    el.style.setProperty('--ring-pi', String(Math.floor(value + 1e-6)));
    el.setAttribute('aria-valuenow', String(Math.floor(value + 1e-6)));
  };
  const adopt = (el, now) => {
    let start = Number(el.dataset.from) || 0;
    const earlier = el.dataset.continue && states[el.dataset.continue];
    if (earlier) start = Math.max(start, earlier.value);  // pick up where the capture ring was
    const s = (states[el.dataset.job] = { value: start, floor: Number(el.dataset.floor) || 0, since: now });
    paint(el, start);
    return s;
  };
  // Instant feedback: the moment 計算 is pressed (mouse or the R hotkey) the scale panel + form are swapped
  // for a ring in the same place, without waiting for the server. The server's capture ring then takes over
  // from this ring's value (data-continue="instant"); an error brings the form back.
  let instantAt = 0;
  const removeInstant = (restoreForm) => {
    document.querySelectorAll('.pr-instant').forEach((node) => node.remove());
    if (restoreForm) {
      document.querySelectorAll('.st-key-measure_form_area').forEach((node) => { node.style.visibility = ''; });
    }
    delete states.instant;
    instantAt = 0;
  };
  document.addEventListener('click', (event) => {
    const button = event.target.closest && event.target.closest('.st-key-measure_form_area [data-testid="stFormSubmitButton"] button');
    if (!button || button.disabled) return;
    const area = button.closest('.st-key-measure_form_area');
    const id = ((area.innerText.match(/Chicken ID[：:]\s*(\S+)/) || [])[1]) || '';
    setTimeout(() => {  // after Streamlit has handled the click
      removeInstant(false);
      // An overlay in the scroll container (Streamlit re-renders the form's block and would drop a node put
      // inside it). It covers the form area exactly; the area keeps its height, so nothing below moves.
      const host = document.querySelector('[data-testid="stMain"]') || document.body;
      if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
      const box = area.getBoundingClientRect();
      const hostBox = host.getBoundingClientRect();
      const holder = document.createElement('div');
      holder.className = 'pr-instant';
      const page = document.querySelector('[data-testid="stAppViewContainer"]');
      Object.assign(holder.style, {
        position: 'absolute', zIndex: 5, left: (box.left - hostBox.left + host.scrollLeft) + 'px',
        top: (box.top - hostBox.top + host.scrollTop) + 'px', width: box.width + 'px', minHeight: box.height + 'px',
        background: page ? getComputedStyle(page).backgroundColor : 'transparent',
      });
      holder.innerHTML = INSTANT_TEMPLATE.replace('__CHICKEN_ID__', id.replace(/[<>&"]/g, ''));
      host.appendChild(holder);
      area.style.visibility = 'hidden';
      instantAt = performance.now();
      wake();
    }, 0);
  }, true);
  // A ring inserted by Streamlit gets its start value before it is first painted (no 1-frame flash).
  new MutationObserver(() => {
    const found = document.querySelectorAll('.pr-ring[data-job]');
    if (found.length) wake();
    found.forEach((el) => {
      if (!states[el.dataset.job]) adopt(el, performance.now());
      else paint(el, Math.min(100, states[el.dataset.job].value));
    });
    if (instantAt && document.querySelector('.st-key-measure_form_area [data-testid="stAlert"]')) {
      removeInstant(true);  // the server refused (e.g. not calibrated): show the form and its message again
    }
  }).observe(document.body, { childList: true, subtree: true });
  // The overlay stays until the server's ring is actually ON SCREEN (it can sit in the page, not yet shown,
  // for a while during the re-run), so there is never a moment without a ring.
  const serverRingShown = () => [...document.querySelectorAll('.pr-ring[data-job]:not([data-job="instant"])')]
    .some((el) => el.getClientRects().length && el.offsetHeight > 0);
  // Safety net: no answer from the server within 15 s -> show the form again.
  setInterval(() => { if (instantAt && performance.now() - instantAt > 15000) removeInstant(true); }, 1000);
  let last = performance.now();
  let idleTimer = 0;       // set while the loop only polls (no ring on screen)
  // A new ring (instant overlay or one from the server) restarts per-frame animation at once instead of
  // waiting for the next idle poll, so it never sits at 0 % and hand-overs are seen on their first frame.
  const wake = () => {
    lastSeen = performance.now();
    if (!idleTimer) return;
    clearTimeout(idleTimer);
    idleTimer = 0;
    last = performance.now();
    requestAnimationFrame(frame);
  };
  let lastSeen = 0;  // keep animating for a while after the last ring, so a ring that replaces another
                    // (capture ring -> job ring) is picked up on its very first frame
  const frame = (now) => {
    if (instantAt && serverRingShown()) removeInstant(false);
    const rings = document.querySelectorAll('.pr-ring[data-job]');
    const dt = Math.min(100, now - last);
    last = now;
    rings.forEach((el) => {
      const floor = Number(el.dataset.floor) || 0;
      let s = states[el.dataset.job];
      if (!s) s = adopt(el, now);
      if (floor !== s.floor) { s.floor = floor; s.since = now; }
      const target = floor >= 100 ? 100 : floor + 19 * (1 - Math.exp(-(now - s.since) / CREEP_MS));
      if (target > s.value) s.value += (target - s.value) * (1 - Math.exp(-dt / FOLLOW_MS));
      if (target - s.value < 0.02) s.value = Math.max(s.value, target);
      paint(el, Math.min(100, s.value));
    });
    if (rings.length) lastSeen = now;
    if (now - lastSeen < 3000) {
      requestAnimationFrame(frame);
    } else {
      idleTimer = setTimeout(() => { idleTimer = 0; last = performance.now(); requestAnimationFrame(frame); }, 150);  // idle
    }
  };
  requestAnimationFrame(frame);
})();
"""


# Clicking an inference photo opens it full screen (same as its small 「Fullscreen」 button); clicking it there,
# or that button, closes it again. Installed once per page.
IMAGE_ZOOM_HTML = """<script>(() => {
  if (window.__chickenImageZoom) return;
  window.__chickenImageZoom = true;
  document.addEventListener('click', (event) => {
    const image = event.target.closest && event.target.closest('.st-key-snapshot_result_images [data-testid="stImage"] img');
    if (!image) return;
    const frame = image.closest('[data-testid="stElementContainer"]');
    const button = frame && frame.querySelector('button[data-testid="stBaseButton-elementToolbar"]');
    if (button) { event.preventDefault(); button.click(); }
  }, true);
})();</script>"""


def ring_driver_html() -> str:
    """Zero-height component that installs RING_DRIVER_JS into the app page once (runs in the page itself,
    so it keeps running when Streamlit re-renders or removes this helper)."""
    import json

    template = progress_ring_html("__CHICKEN_ID__", 0, 0, "instant")
    script = "const INSTANT_TEMPLATE = " + json.dumps(template) + ";\n" + RING_DRIVER_JS
    # st.html sanitizes its HTML and drops a <script> whose text looks like markup ("<div"), so every "<"
    # inside the embedded JS string is written as < (the browser decodes it back).
    embedded = json.dumps(script).replace("<", "\\u003c")
    return (
        "<script>(() => { try { const d = window.parent.document;"
        " if (window.parent.__chickenRingDriver === 11) return;"
        " const s = d.createElement('script'); s.textContent = " + embedded + ";"
        " d.head.appendChild(s); } catch (e) { console.warn('ring driver not installed', e); } })();</script>"
    )


GRADE_TEXT_COLORS = {"通過": "#22c55e", "備用": "#eab308", "不通過": "#ef4444"}


def style_grade_columns(frame, columns):
    """Tables: show 通過 / 備用 / 不通過 in the light's colour (green / yellow / red). `columns` are the
    displayed column names; numbers are shown with 2 decimals."""
    colors = {t(grade): color for grade, color in GRADE_TEXT_COLORS.items()}
    present = [column for column in columns if column in frame.columns]
    styler = frame.style
    if present:
        styler = styler.map(lambda value: f"color: {colors[value]}; font-weight: 700" if value in colors else "",
                            subset=present)
    numeric = frame.select_dtypes("number").columns
    return styler.format(lambda value: f"{value:.2f}", subset=numeric, na_rep="None")  # 2 decimals everywhere


def _delay_rules() -> str:
    rules = [f".mo-dl{i}{{animation-delay:{i * 50}ms}}" for i in range(_MAX_DELAY_STEPS + 1)]
    rules += [f"@keyframes mo-roll{n}{{from{{transform:translateY(0)}}to{{transform:translateY(-{n * 10}%)}}}}"
              f".mo-n{n}{{animation-name:mo-roll{n}}}" for n in range(10)]
    rules += [f".mo-cd{i}{{animation-delay:{i * 90}ms}}.mo-cd{i} .mo-bar{{animation-delay:{300 + i * 140}ms}}"
              for i in range(8)]
    return "".join(rules)


MOTION_CSS = """
/* sidebar tabs: the current one springs out (static style: same highlight, no animation) */
.st-key-main_page_navigation_segmented [role="radiogroup"] button {
    transition: transform .28s cubic-bezier(.34,1.56,.64,1), background .25s ease, color .2s ease,
        box-shadow .3s ease, border-color .25s ease;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button::before { transition: filter .3s ease; }
@media (prefers-reduced-motion: reduce) {
    .st-key-main_page_navigation_segmented [role="radiogroup"] button { transition: none; }
}
:root {
    --mo-card: #121a26; --mo-band: #0e141e; --mo-border: #222d3f; --mo-num: #ffffff;
    --mo-label: #c5cfdc; --mo-muted: #93a1b5; --mo-track: #1c2636;
    --mo-weight: #e7ecf4; --mo-comb: #ff9a4d; --mo-shank: #4f8cff;
    --mo-pass-fg: #4ade80; --mo-pass-bg: rgba(34,197,94,.14); --mo-pass-bd: rgba(34,197,94,.5);
    --mo-standby-fg: #fde047; --mo-standby-bg: rgba(250,204,21,.12); --mo-standby-bd: rgba(250,204,21,.45);
    --mo-fail-fg: #fca5a5; --mo-fail-bg: rgba(239,68,68,.14); --mo-fail-bd: rgba(239,68,68,.5);
}
.mo-band { background: var(--mo-band); border: 1px solid var(--mo-border); border-radius: 18px;
    padding: 16px 18px; display: flex; flex-direction: column; gap: 14px; }
.mo-head { display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }
.mo-eyebrow { font-size: .78rem; color: var(--mo-muted); letter-spacing: .16em; }
.mo-id { font-family: 'Space Grotesk', Inter, system-ui, sans-serif; font-size: 2.1rem; font-weight: 700;
    color: var(--mo-num); letter-spacing: .02em; animation: mo-slide .7s cubic-bezier(.16,1,.3,1) both; }
.mo-badge { display: inline-flex; align-items: center; height: 2.1rem; padding: 0 1rem; border-radius: 999px;
    font-size: 1.1rem; font-weight: 800; letter-spacing: .08em; border: 1px solid transparent;
    animation: mo-stamp .55s cubic-bezier(.2,.8,.2,1) .7s both; }
.mo-pass { color: var(--mo-pass-fg); background: var(--mo-pass-bg); border-color: var(--mo-pass-bd); }
.mo-standby { color: var(--mo-standby-fg); background: var(--mo-standby-bg); border-color: var(--mo-standby-bd); }
.mo-fail { color: var(--mo-fail-fg); background: var(--mo-fail-bg); border-color: var(--mo-fail-bd); }
.mo-neutral { color: var(--mo-label); border-color: var(--mo-border); }
.mo-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.mo-card { background: var(--mo-card); border: 1px solid var(--mo-border); border-radius: 16px;
    padding: 14px 16px 12px; display: flex; flex-direction: column; gap: 10px; min-width: 0;
    animation: mo-slide .7s cubic-bezier(.16,1,.3,1) both; }
.mo-label { display: flex; align-items: center; gap: 8px; font-size: .9rem; font-weight: 650; color: var(--mo-label); }
.mo-dot { width: 10px; height: 10px; border-radius: 5px; background: var(--mo-tone); flex: none; }
.mo-weight { --mo-tone: var(--mo-weight); } .mo-comb { --mo-tone: var(--mo-comb); } .mo-shank { --mo-tone: var(--mo-shank); }
.mo-value { display: flex; align-items: baseline; gap: 6px; min-width: 0; }
.mo-unit { font-size: 1.05rem; color: var(--mo-muted); }
.mo-odo { display: inline-flex; align-items: flex-end; font-family: 'Space Grotesk', Inter, system-ui, sans-serif;
    font-variant-numeric: tabular-nums; font-size: clamp(2.1rem, 2.9vw, 3.6rem); font-weight: 650;
    line-height: 1; letter-spacing: -.02em; color: var(--mo-num); }
.mo-col { display: inline-block; height: 1em; overflow: hidden; }
.mo-strip { display: flex; flex-direction: column; animation-duration: 1.3s;
    animation-timing-function: cubic-bezier(.16,1,.3,1); animation-fill-mode: both; }
.mo-strip span { display: block; height: 1em; line-height: 1em; text-align: center; }
.mo-sep { display: inline-block; height: 1em; line-height: 1em; }
.mo-track { height: 3px; background: var(--mo-track); border-radius: 2px; overflow: hidden; }
.mo-bar { display: block; height: 100%; background: var(--mo-tone); transform-origin: left;
    animation: mo-sweep 1.1s cubic-bezier(.16,1,.3,1) both; }
.mo-single .mo-card { max-width: 340px; }
@keyframes mo-slide { from { opacity: 0; transform: translateY(26px); } to { opacity: 1; transform: none; } }
@keyframes mo-sweep { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes mo-stamp { 0% { opacity: 0; transform: scale(1.6); } 60% { opacity: 1; transform: scale(.94); } 100% { transform: scale(1); } }
/* native metrics (live scale, calibration…): bigger numbers, slide in when they appear */
[data-testid="stMetric"] { animation: mo-slide .6s cubic-bezier(.16,1,.3,1) both; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', Inter, system-ui, sans-serif; font-size: 2.4rem; letter-spacing: -.01em; }
.st-key-snapshot_result_images img { max-height: 15vh; }
/* phenotype page: result sits in the narrow right column → numbers 2×2, tighter spacing */
.st-key-snapshot_status .mo-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.st-key-snapshot_status .mo-band { padding: 12px 14px; gap: 10px; }
.st-key-snapshot_status { padding-bottom: 1rem; }  /* markdown's -1rem bottom margin would let the band overlap the images */
.st-key-snapshot_status .mo-id { font-size: 1.8rem; }
.st-key-snapshot_status .mo-card { padding: 10px 14px 9px; gap: 4px; }
.st-key-snapshot_status .mo-label { font-size: 1.22rem; font-weight: 700; color: var(--mo-num); gap: 9px; }
.st-key-snapshot_status .mo-dot { width: 12px; height: 12px; border-radius: 6px; }
.st-key-snapshot_status .mo-unit { font-size: 1.15rem; }
.st-key-snapshot_result_images [data-testid="stImageCaption"] { margin-top: 2px; padding: 0; }
@media (prefers-reduced-motion: reduce) {
    .mo-strip { animation-duration: 0s; } .mo-card, .mo-id, .mo-badge, .mo-bar, [data-testid="stMetric"] { animation: none; }
}
""" + _delay_rules()


CREAM_CSS = """
:root {
    --panel-border: #e2d9c7; --surface: #fffdf8; --surface-2: #f9f5ec; --surface-hover: #f1eadc;
    --line: #e2d9c7; --line-strong: #d3c7b0; --muted: #6b6358; --text-strong: #1f1b16;
    --accent: #2f6fe4; --accent-soft: rgba(47, 111, 228, 0.16);
    --pass: #16a34a; --standby: #ca8a04; --fail: #dc2626;
    --mo-card: #f9f5ec; --mo-band: #fffdf8; --mo-border: #e2d9c7; --mo-num: #1f1b16;
    --mo-label: #3a342c; --mo-muted: #6b6358; --mo-track: #ebe3d3;
    --mo-weight: #3a342c; --mo-comb: #e0701f; --mo-shank: #2f6fe4;
    --mo-pass-fg: #15803d; --mo-pass-bg: rgba(22,163,74,.12); --mo-pass-bd: rgba(22,163,74,.45);
    --mo-standby-fg: #92400e; --mo-standby-bg: rgba(234,179,8,.18); --mo-standby-bd: rgba(202,138,4,.5);
    --mo-fail-fg: #b91c1c; --mo-fail-bg: rgba(220,38,38,.10); --mo-fail-bd: rgba(220,38,38,.45);
    /* measurement progress ring */
    --ring-band: #fffdf8; --ring-border: #e2d9c7; --ring-text: #1f1b16; --ring-muted: #6b6358; --ring-num: #1f1b16;
    --ring-plate: #f9f5ec; --ring-plate-shadow: 10px 12px 26px rgba(120,100,60,.18), -10px -10px 22px rgba(255,255,255,.9);
    --ring-track: #ebe3d3; --ring-core: #fffdf8;
    --ring-core-shadow: inset 5px 6px 14px rgba(120,100,60,.14), inset -4px -4px 10px rgba(255,255,255,.95);
    --ring-glow: none; --ring-accent: #2f6fe4; --ring-accent-2: #5cb8f0; --ring-step: #f9f5ec; --ring-dot: #d3c7b0;
}
[data-testid="stAppViewContainer"] { background: #f6f1e7; }
.app-logo { background: #2f6fe4; box-shadow: 0 6px 16px rgba(47, 111, 228, 0.25); }
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
    background: linear-gradient(180deg, #3b7bec, #2a63d4); border-color: #2f6fe4;
    box-shadow: 0 4px 12px rgba(47, 111, 228, 0.22); color: #ffffff;
}
[data-testid="stButton"] > button[kind="primary"] p,
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] p { color: #ffffff; }
div[class*="st-key-card_"], [data-testid="stMetric"], .threshold-summary-card, .summary-stat {
    box-shadow: 0 1px 2px rgba(60, 45, 20, 0.05), 0 6px 18px rgba(60, 45, 20, 0.06);
}
[data-testid="stWidgetLabel"] p { color: #3a342c; }
.light-circle { border-color: rgba(60, 45, 20, 0.10); }
.st-key-pass_threshold_block h3 { color: #15803d; }
.st-key-standby_threshold_block h3 { color: #a16207; }
.st-key-pass_threshold_block { border-color: rgba(22, 163, 74, 0.6) !important; }
.st-key-standby_threshold_block { border-color: rgba(202, 138, 4, 0.6) !important; }
.st-key-card_model_import [data-baseweb="input"], .st-key-card_model_import [data-testid="stTextInputRootElement"],
.st-key-card_model_import [data-testid="stFileUploaderDropzone"] { background: #fffdf8 !important; border-color: #e2d9c7 !important; }
/* sidebar in cream */
section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #fffdf8 0%, #f9f5ec 100%);
    border-color: #e2d9c7;
    box-shadow: 0 1px 2px rgba(60, 45, 20, 0.05), 0 14px 34px -14px rgba(60, 45, 20, 0.22);
}
.sidebar-brand, .sidebar-foot { border-color: #e2d9c7; }
.sidebar-brand-name { color: #1f1b16; }
.sidebar-brand-sub, .sidebar-foot { color: #6b6358; }
.sidebar-brand .app-logo { background: #2f6fe4; padding: 4px; box-sizing: border-box;
    box-shadow: 0 6px 16px rgba(47, 111, 228, 0.25); }  /* white chicken needs a dark tile on cream */
.st-key-main_page_navigation_segmented [role="radiogroup"] button { color: #5a5248; }
.st-key-main_page_navigation_segmented [role="radiogroup"] button:hover {
    color: #1f1b16; background: rgba(47, 111, 228, 0.07); border-color: rgba(47, 111, 228, 0.20) !important;
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button[aria-checked="true"] {
    color: #1d4fb8;
    background: linear-gradient(100deg, rgba(47, 111, 228, 0.18) 0%, rgba(47, 111, 228, 0.07) 60%, rgba(47, 111, 228, 0.03) 100%);
    border-color: rgba(47, 111, 228, 0.55) !important;
    box-shadow: 0 10px 24px -8px rgba(47, 111, 228, 0.40), 0 0 0 1px rgba(47, 111, 228, 0.12);
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button[aria-checked="true"] p { color: #1d4fb8; }
.st-key-main_page_navigation_segmented [role="radiogroup"] button[aria-checked="true"]::after {
    background: #2f6fe4; box-shadow: 0 0 8px rgba(47, 111, 228, 0.6);
}
.st-key-main_page_navigation_segmented [role="radiogroup"] button[aria-checked="true"]::before { filter: none; }
"""
