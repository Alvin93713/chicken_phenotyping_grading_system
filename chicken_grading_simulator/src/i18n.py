"""UI language switch (中文 / English).

Chinese is the source language: every UI string in the code stays Chinese and is
passed through `t()` at render time.  `t()` returns the Chinese text unchanged
when the language is 中文 (the default, so the original UI is untouched) and the
English translation when it is English.

Rules for callers
- Call `t()` when rendering, never at import time (the language can change per run).
- Never translate values that are stored or compared (DB values, job statuses,
  grades, review states, widget option values).  Translate them only for display,
  e.g. `st.segmented_control(..., options=STATES, format_func=t)` or `t(row["狀態"])`.
- Strings with variables: `t("最近完成：{chicken_id} · {grade}", chicken_id=x, grade=t(g))`.
  The dictionary key is the template with the `{name}` placeholders.

English text lives in the `locale_en_*.py` modules (one per UI area), merged here.
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import streamlit as st

LANG_ZH = "中文"
LANG_EN = "English"
LANGUAGES = (LANG_ZH, LANG_EN)

_CJK = re.compile(r"[㐀-鿿＀-￯　-〿]")

# Shared data values (stored in the DB / compared in code; translate for display only).
VALUES_EN = {
    # grading
    "通過": "Pass",
    "備用": "Standby",
    "不通過": "Fail",
    # inference jobs
    "等待中": "Waiting",
    "錄影中": "Recording",
    "推論中": "Inferring",
    "已完成": "Completed",
    "失敗": "Failed",
    # case review
    "待審核": "Pending",
    "正確": "Correct",
    "不正確": "Incorrect",
    "無法判斷": "Unsure",
    "全部": "All",
    # appearance / language switch
    "原版": "Classic",
    "動態": "Motion",
    "深色": "Dark",
    "米白": "Cream",
    "中文": "中文",
    "English": "English",
}

_EN: dict[str, str] | None = None
_MISSING: set[str] = set()


def _load() -> dict[str, str]:
    global _EN
    if _EN is None:
        merged = dict(VALUES_EN)
        folder = Path(__file__).resolve().parent
        for module in sorted(pkgutil.iter_modules([str(folder)]), key=lambda m: m.name):
            if module.name.startswith("locale_en_"):
                merged.update(getattr(importlib.import_module(module.name), "EN", {}))
        _EN = merged
    return _EN


def current_language() -> str:
    try:
        return st.session_state.get("_ui_lang") or LANG_ZH
    except Exception:  # outside a Streamlit run (worker threads, tests)
        return LANG_ZH


def is_english() -> bool:
    return current_language() == LANG_EN


def t(text, **values) -> str:
    """Translate one UI string (Chinese source text) into the current language."""
    if text is None:
        return ""
    source = str(text)
    result = source
    if is_english() and source:
        english = _load().get(source)
        if english is not None:
            result = english
        elif _CJK.search(source):
            _MISSING.add(source)
    if values:
        try:
            return result.format(**values)
        except (KeyError, IndexError, ValueError):
            return source.format(**values)
    return result


def t_columns(frame):
    """Translate DataFrame column names (and nothing else) for display."""
    if not is_english():
        return frame
    return frame.rename(columns={column: t(column) for column in frame.columns})


def t_records(rows):
    """Translate the keys of a list of dict rows (for st.dataframe on lists of dicts)."""
    if not is_english():
        return rows
    return [{t(key): value for key, value in row.items()} for row in rows]


def missing_translations() -> list[str]:
    """Chinese strings shown while English was selected but with no translation (for QA/tests)."""
    return sorted(_MISSING)
