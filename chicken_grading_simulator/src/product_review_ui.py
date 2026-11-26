"""Streamlit pages for product quality monitoring and lightweight review."""
from __future__ import annotations

import base64
import mimetypes
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import model_registry
from dual_camera import MODEL_PATHS
from i18n import t, t_records
from product_review import (
    bundle_summary,
    export_counts,
    export_review_bundle,
    list_cases,
    review_case_role,
    review_summary,
    role_note,
    seed_demo_cases,
)
import secure_sync
import ui_style
from secure_sync import SyncConfig


REVIEW_ROLES = ("comb", "shank")
GALLERY_PAGE_SIZE = 30
GALLERY_COLUMNS = 5


def _display_local_image(path: Path, width: str = "100%") -> None:
    """Embed local QA images without relying on Streamlit's media endpoint."""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        f'<img src="data:{mime_type};base64,{encoded}" '
        f'style="display:block;width:{width};margin:0 auto;aspect-ratio:16/9;object-fit:contain;'
        'background:#151922;border:1px solid #343a46;border-radius:6px;" />',
        unsafe_allow_html=True,
    )


def _model_rows() -> list[dict[str, object]]:
    rows = []
    for role, path in MODEL_PATHS.items():
        if path.is_file():
            digest = secure_sync.file_hash12(path)  # cached per file version (comb model is >100 MB)
            rows.append({"模型": role.upper(), "檔案": path.name, "版本雜湊": digest, "大小 (MB)": f"{path.stat().st_size / 1024 / 1024:.2f}", "狀態": t("已載入")})
        else:
            rows.append({"模型": role.upper(), "檔案": path.name, "版本雜湊": "-", "大小 (MB)": "-", "狀態": t("缺少檔案")})
    return t_records(rows)


def _t_reason(reason) -> str:
    """Display text for a stored case reason (e.g. 「最低信心 0.412」)."""
    text = str(reason or "")
    prefix = "最低信心 "
    if text.startswith(prefix):
        value = text[len(prefix):]
        try:
            value = f"{float(value):.2f}"  # 2 decimals, also for reasons stored with 3
        except ValueError:
            pass
        return t("最低信心 {value}", value=value)
    return t(text)


def quality_dashboard_page(page_heading) -> None:
    page_heading(t("模型品質監控"), t("集中追蹤低信心、推論失敗與人工確認結果。"))
    summary = review_summary()
    metrics = (
        ("待審核", summary["待審核"]),
        ("確認正確", summary["正確"]),
        ("確認錯誤", summary["不正確"]),
        ("無法判斷", summary["無法判斷"]),
        ("總案例", summary["總案例"]),
    )
    for (label, value), column in zip(metrics, st.columns(5)):
        column.metric(t(label), value)

    st.subheader(t("目前模型"))
    st.dataframe(pd.DataFrame(_model_rows()), width="stretch", hide_index=True, height=112)

    cases = list_cases()
    st.subheader(t("最近品質事件"))
    if not cases:
        st.info(t("尚無品質事件。"))
        return
    rows = [
        {
            "case ID": case["case_id"],
            "chicken ID": case["chicken_id"],
            "優先度": t(case["priority"]),
            "原因": _t_reason(case["reason"]),
            "審核狀態": t(case["review_state"]),
            "最低信心": "-" if case["minimum_confidence"] is None else f"{float(case['minimum_confidence']):.2f}",
            "建立時間": case["created_at"],
        }
        for case in cases[:20]
    ]
    st.dataframe(t_records(rows), width="stretch", hide_index=True, height=270)


def _role_items(status: str, role_filter: str | None = None) -> list[tuple[dict[str, object], str]]:
    items = []
    for case in list_cases():
        for role in REVIEW_ROLES:
            if role_filter and role != role_filter:
                continue
            role_state = case[f"{role}_review_state"]
            if status == "全部" or role_state == status:
                items.append((case, role))
    return items


def _case_image(case: dict[str, object], role: str) -> Path:
    result_path = Path(case["result_paths"].get(role, ""))
    source_path = Path(case["source_paths"].get(role, ""))
    return result_path if result_path.is_file() else source_path


REVISE_OPTIONS = ("正確", "不正確", "無法判斷", "待審核")


def _review_flash(kind: str, message: str) -> None:
    st.session_state["review_flash"] = (kind, message)


def _show_review_flash() -> None:
    flash = st.session_state.pop("review_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])


def _undo_last_review() -> None:
    """One-click undo for the most recent decision made on the pending queue."""
    last = st.session_state.get("last_review_action")
    if not last:
        return
    text_column, button_column = st.columns([4, 1], vertical_alignment="center")
    text_column.info(
        t(
            "上一張：{chicken_id} · {role} → 判定為「{decision}」。選錯了可以復原。",
            chicken_id=last["chicken_id"],
            role=str(last["role"]).upper(),
            decision=t(last["decision"]),
        )
    )
    if button_column.button(t("↩ 復原上一張"), width="stretch", key="undo_last_review"):
        review_case_role(last["case_id"], last["role"], "待審核", last["previous_note"])
        st.session_state.pop("last_review_action", None)
        _review_flash(
            "success",
            t(
                "已復原：{chicken_id} · {role} 回到待審核，請重新判定。",
                chicken_id=last["chicken_id"],
                role=str(last["role"]).upper(),
            ),
        )
        st.rerun()


def _revise_decision(case: dict[str, object], role: str) -> None:
    """Change an earlier decision (or send the image back to the pending queue)."""
    key = f"{case['case_id']}_{role}"
    current = str(case[f"{role}_review_state"])
    with st.popover(t("修改判定"), width="stretch"):
        choice = st.radio(
            t("改為"),
            REVISE_OPTIONS,
            index=REVISE_OPTIONS.index(current) if current in REVISE_OPTIONS else 0,
            format_func=lambda option: t("退回待審核（重新判定）") if option == "待審核" else t(option),
            key=f"revise_choice_{key}",
        )
        note = st.text_input(t("備註"), value=role_note(case, role), key=f"revise_note_{key}")
        if st.button(t("儲存修改"), type="primary", width="stretch", key=f"revise_save_{key}"):
            if choice == current and note == role_note(case, role):
                st.info(t("沒有變更。"))
            else:
                review_case_role(str(case["case_id"]), role, choice, note)
                if choice == "待審核":
                    message = t(
                        "{chicken_id} · {role} 已退回待審核，可在「待審核」重新判定。",
                        chicken_id=case["chicken_id"],
                        role=role.upper(),
                    )
                else:
                    message = t(
                        "{chicken_id} · {role} 已由「{current}」改為「{choice}」。",
                        chicken_id=case["chicken_id"],
                        role=role.upper(),
                        current=t(current),
                        choice=t(choice),
                    )
                _review_flash("success", message)
                st.rerun()


def _pending_review() -> None:
    _undo_last_review()
    items = _role_items("待審核")
    if not items:
        st.success(t("已無待審核影像。"))
        return

    case, role = items[0]
    confidence = case["role_confidences"].get(role)
    info_columns = st.columns(4)
    info_columns[0].metric("Chicken ID", case["chicken_id"])
    info_columns[1].metric(t("影像類型"), role.upper())
    info_columns[2].metric(t("信心"), "-" if confidence is None else f"{float(confidence):.2f}")
    info_columns[3].metric(t("剩餘待審核"), len(items))

    image_column, action_column = st.columns(2, gap="large")
    with image_column:
        display_path = _case_image(case, role)
        if display_path.is_file():
            _display_local_image(display_path)
        else:
            st.warning(t("影像不存在"))

    with action_column:
        if case["error"] and role == "comb":
            st.error(t(case["error"]))
        note = st.text_input(
            t("備註"),
            value=role_note(case, role),
            key=f"review_note_{role}_{case['case_id']}",
        )
        actions = (("正確", "primary"), ("不正確", "secondary"), ("無法判斷", "secondary"))
        for decision, button_type in actions:
            if st.button(
                t(decision),
                type=button_type,
                width="stretch",
                key=f"decision_{role}_{decision}_{case['case_id']}",
            ):
                st.session_state["last_review_action"] = {
                    "case_id": case["case_id"],
                    "chicken_id": case["chicken_id"],
                    "role": role,
                    "decision": decision,
                    "previous_note": role_note(case, role),
                }
                review_case_role(case["case_id"], role, decision, note)
                st.rerun()


def _review_gallery(status: str, role: str) -> None:
    items = _role_items(status, role)
    if not items:
        st.info(t("目前沒有符合條件的案例影像。"))
        return

    page_count = (len(items) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE
    page = st.selectbox(
        t("頁數"),
        range(1, page_count + 1),
        format_func=lambda value: t("第 {page} / {page_count} 頁", page=value, page_count=page_count),
        key=f"review_gallery_page_{status}",
    )
    start = (page - 1) * GALLERY_PAGE_SIZE
    visible_items = items[start : start + GALLERY_PAGE_SIZE]
    st.caption(
        t(
            "共 {count} 張",
            count=len(items),
            role=role.upper(),
            page_size=GALLERY_PAGE_SIZE,
        )
    )

    for index, (case, role) in enumerate(visible_items):
        if index % GALLERY_COLUMNS == 0:
            gallery_columns = st.columns(GALLERY_COLUMNS)
        with gallery_columns[index % GALLERY_COLUMNS]:
            with st.container(border=True):
                display_path = _case_image(case, role)
                if display_path.is_file():
                    _display_local_image(display_path)
                else:
                    st.warning(t("影像不存在"))
                confidence = case["role_confidences"].get(role)
                role_state = case[f"{role}_review_state"]
                st.markdown(f"**{role.upper()} · {case['chicken_id']}**")
                st.markdown(t("**判定：{state}**", state=t(role_state)))
                st.write(t("信心：{value}", value="-" if confidence is None else f"{float(confidence):.2f}"))
                st.caption(_t_reason(case["reason"]))
                _revise_decision(case, role)


def _keep_selection(key: str, fallback: str) -> None:
    """Clicking the highlighted segment again deselects it (value None); keep the previous choice instead."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(f"{key}__last", fallback)
    st.session_state[f"{key}__last"] = st.session_state[key]


def review_cases_page(page_heading) -> None:
    page_heading(t("案例審核"), t("COMB 與 SHANK 採獨立審核佇列。"))
    status = st.segmented_control(
        t("審核篩選"),
        ["待審核", "不正確", "無法判斷", "正確", "全部"],
        default="待審核",
        format_func=t,
        key="review_status_filter",
        on_change=_keep_selection,
        args=("review_status_filter", "待審核"),
    )
    _show_review_flash()
    if status == "待審核":
        _pending_review()
    else:
        role_key = f"review_gallery_role_{status}"
        role_label = st.segmented_control(
            t("影像類型"),
            ["COMB", "SHANK"],
            default="COMB",
            key=role_key,
            on_change=_keep_selection,
            args=(role_key, "COMB"),
        )
        _review_gallery(status, role_label.lower())


def _mask(value: str) -> str:
    value = str(value or "")
    return t("（未設定）") if not value else ("•" * 8 + value[-4:] if len(value) > 4 else "•" * len(value))


def _flash(kind: str, message: str) -> None:
    """Show a message after st.rerun() (messages written right before a rerun are otherwise lost)."""
    st.session_state["sync_flash"] = (kind, message)


def _show_flash() -> None:
    flash = st.session_state.pop("sync_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])


def _sync_settings_panel(config: SyncConfig, configured: bool) -> None:
    with st.expander(t("連線設定"), expanded=not configured):
        overrides = secure_sync.environment_overrides()
        if overrides:
            st.caption(t("注意：系統環境變數 {names} 已設定，會優先於此處儲存的設定。", names=", ".join(overrides)))
        version = int(st.session_state.get("sync_settings_version", 0))
        paste_tab, manual_tab = st.tabs([t("貼上連線設定碼（建議）"), t("手動輸入")])
        with paste_tab:
            code = st.text_area(
                t("連線設定碼"),
                placeholder=t("CHKSYNC1:...（由廠商提供，請視同密碼保管）"),
                height=90,
                key=f"sync_setup_code_{version}",
            )
            if st.button(t("儲存連線設定碼"), type="primary", width="stretch", key="sync_save_code"):
                try:
                    secure_sync.save_config(secure_sync.decode_setup_code(code))
                    secure_sync.start_sync_worker()
                    st.session_state["sync_settings_version"] = version + 1
                    _flash("success", t("已儲存，請按「測試連線」確認。"))
                    st.rerun()
                except ValueError as exc:
                    st.error(secure_sync.display_validation_error(str(exc)))
        with manual_tab:
            saved = secure_sync.load_saved_config()
            with st.form(f"sync_manual_form_{version}"):
                endpoint = st.text_input(t("上傳網址（https://…）"), value=saved.get("endpoint", ""))
                id_columns = st.columns(2)
                site_id = id_columns[0].text_input(t("場域代號"), value=saved.get("site_id", ""))
                device_id = id_columns[1].text_input(t("裝置代號"), value=saved.get("device_id", ""))
                key_columns = st.columns(2)
                api_key = key_columns[0].text_input(t("API 金鑰"), type="password", value=saved.get("api_key", ""))
                hmac_secret = key_columns[1].text_input(t("簽章密鑰"), type="password", value=saved.get("hmac_secret", ""))
                ca_bundle = st.text_input(t("CA 憑證檔路徑（選填，私有憑證才需要）"), value=saved.get("ca_bundle", ""))
                if st.form_submit_button(t("儲存手動設定"), width="stretch"):
                    try:
                        secure_sync.save_config(
                            {
                                "endpoint": endpoint,
                                "site_id": site_id,
                                "device_id": device_id,
                                "api_key": api_key,
                                "hmac_secret": hmac_secret,
                                "ca_bundle": ca_bundle,
                            }
                        )
                        secure_sync.start_sync_worker()
                        st.session_state["sync_settings_version"] = version + 1
                        _flash("success", t("已儲存連線設定。請按「測試連線」確認。"))
                        st.rerun()
                    except ValueError as exc:
                        st.error(secure_sync.display_validation_error(str(exc)))
        if configured:
            st.caption(
                t(
                    "目前設定：{endpoint} · 場域 {site_id} · 裝置 {device_id} · "
                    "API 金鑰 {api_key} · 簽章密鑰 {hmac_secret}",
                    endpoint=config.endpoint,
                    site_id=config.site_id,
                    device_id=config.device_id,
                    api_key=_mask(config.api_key),
                    hmac_secret=_mask(config.hmac_secret),
                )
            )
        if secure_sync.config_path().is_file() and st.button(t("清除已儲存的連線設定"), key="sync_clear_config"):
            secure_sync.clear_saved_config()
            st.session_state["sync_settings_version"] = version + 1
            _flash("info", t("已清除連線設定，之後無法透過網路回傳。"))
            st.rerun()


def _megabytes(value) -> str:
    return f"{(int(value or 0)) / 1024 / 1024:.2f} MB"


@st.fragment(run_every=2)
def _snapshot_progress() -> None:
    """Live status of the snapshot upload (refreshes itself while the page is open)."""
    status = secure_sync.snapshot_status()
    active, done = status["active"], status["last_done"]
    if active:
        if active["status"] == "failed":
            reason = secure_sync.describe_error(active["last_error"])
            message = t(
                "回傳暫時失敗：{reason}。下次自動重試：{next_attempt}",
                reason=reason,
                next_attempt=active["next_attempt_at"],
            )
            if active["stalled"]:
                st.error(message + t("（已連續失敗超過約 3 小時，請依原因排除問題）"))
            else:
                st.warning(message)
        if not active["parts_total"]:
            st.progress(0.0, text=t("正在打包所有已審核影像…"))
        else:
            done_parts, total_parts = int(active["parts_done"]), int(active["parts_total"])
            sent = min(int(active["size_bytes"] or 0), done_parts * int(active["part_size"] or 0))
            st.progress(
                done_parts / total_parts,
                text=t(
                    "正在回傳：{sent} / {total}（{images} 張影像）",
                    sent=_megabytes(sent),
                    total=_megabytes(active["size_bytes"]),
                    images=active["summary"].get("images", 0),
                ),
            )
    if done:
        summary = done["summary"]
        st.success(
            t(
                "最近一次回傳完成：{completed_at} · {images} 張影像"
                "（可直接訓練 {train_ready}、需重新標註 {to_relabel}） · {size}",
                completed_at=done["completed_at"],
                images=summary.get("images", 0),
                train_ready=summary.get("train_ready", 0),
                to_relabel=summary.get("to_relabel", 0),
                size=_megabytes(done["size_bytes"]),
            )
        )
        if done.get("stored_as"):
            st.caption(t("廠商端檔案：{name}", name=done["stored_as"]))
    elif not active:
        st.info(t("尚未回傳過"))


def _start_snapshot(exportable_total: int) -> None:
    secure_sync.request_snapshot()
    st.toast(t("開始回傳 {count} 張影像", count=exportable_total))


def _sync_panel(exportable_total: int) -> None:
    config = SyncConfig.from_environment()
    configured = not config.validation_error()
    if configured:
        secure_sync.start_sync_worker()

    st.subheader(t("一鍵回傳"))
    _show_flash()
    if configured:
        st.caption(
            t("場域 {site_id} · 裝置 {device_id}", site_id=config.site_id, device_id=config.device_id)
        )
    else:
        st.warning(config.friendly_status() or t("尚未連線到廠商。"))
        st.markdown(
            t(
                "向廠商索取連線設定碼 → 貼到下方「連線設定」並儲存 → 按「測試連線」"
            )
        )

    _snapshot_progress()

    action_columns = st.columns([2, 1, 1])
    # on_click callback (not `if button: … st.rerun()`): one rerun instead of two, and a toast
    # instead of a message inserted above, so the page doesn't jump when the button is pressed.
    action_columns[0].button(
        t("回傳測試影像"),
        type="primary",
        width="stretch",
        disabled=not configured or exportable_total == 0,
        key="sync_send_snapshot",
        on_click=_start_snapshot,
        args=(exportable_total,),
    )
    if action_columns[1].button(t("測試連線"), width="stretch", disabled=not configured, key="sync_test_connection"):
        with st.spinner(t("正在測試與廠商伺服器的連線...")):
            ok, message = secure_sync.check_connection(config)
        _flash("success" if ok else "error", message)
        st.rerun()
    active = secure_sync.snapshot_status()["active"]
    if action_columns[2].button(
        t("立即重試"),
        width="stretch",
        disabled=not (configured and active and active["status"] == "failed"),
        key="sync_retry_now",
    ):
        secure_sync.retry_now()
        _flash("info", t("已立即重新嘗試回傳。"))
        st.rerun()
    _sync_settings_panel(config, configured)


def model_maintenance_page(page_heading) -> None:
    page_heading(t("資料與模型維護"), t("把所有已審核影像回傳給廠商，由廠商重新標註並微調模型。"))
    summary = review_summary()
    cases = list_cases()  # read once; both counts below reuse it
    exportable = export_counts(cases=cases)
    metric_columns = st.columns(4)
    metric_columns[0].metric(t("可匯出正確影像"), exportable["正確"])
    metric_columns[1].metric(t("可匯出不確定影像"), exportable["無法判斷"])
    metric_columns[2].metric(t("可匯出錯誤影像"), exportable["不正確"])
    metric_columns[3].metric(t("待審核案例"), summary["待審核"])

    sync_column, manual_column = st.columns([1.45, 1], gap="medium")
    with sync_column, st.container(border=True, key="card_sync_upload"):
        _sync_panel(sum(exportable.values()))
    with manual_column:
        with st.container(border=True, key="card_manual_bundle"):
            st.subheader(t("手動下載資料包"))
            st.caption(t("含雞隻編號，請用加密隨身碟等安全管道交付，勿用 Email／LINE。"))
            # Built only when the button is clicked, not on every rerun of this page.
            st.download_button(
                t("下載審核資料包"),
                data=export_review_bundle,
                file_name="chicken_review_bundle.zip",
                mime="application/zip",
                width="stretch",
                disabled=sum(exportable.values()) == 0,
            )
            layout = bundle_summary(cases=cases)
            missing = (
                t("（其中 {count} 張缺標註）", count=layout["missing_label"]) if layout["missing_label"] else ""
            )
            st.caption(
                t(
                    "可直接訓練 {train_ready} 張 · 需重新標註 {to_relabel} 張{missing}",
                    train_ready=layout["train_ready"],
                    to_relabel=layout["to_relabel"],
                    missing=missing,
                )
            )
        if ui_style.SHOW_DEMO_TOOLS:  # demo only (hidden in the product)
            with st.container(border=True, key="card_demo_cases"):
                st.subheader(t("流程展示"))
                if st.button(t("載入五筆真實測試案例"), width="stretch"):
                    seed_demo_cases()
                    st.rerun()

    _model_versions_panel()


def _validate_model(path: Path, role: str) -> None:
    """Reject a file that is not a YOLO segmentation model with the role's class."""
    from dual_camera import load_model, target_classes

    target_classes(load_model(str(path), path.stat().st_mtime_ns), role)


def _model_versions_panel() -> None:
    """Model version control: every imported model is kept as best_<role>_<雞舍>_<日期>.pt; pick the one in use."""
    st.subheader(t("模型版本"))
    flash = st.session_state.pop("model_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])
    rows = []
    for role in REVIEW_ROLES:
        in_use = model_registry.active_path(role)
        for version in model_registry.list_versions(role):
            rows.append({
                "模型": role.upper(),
                "檔案": version.path.name,
                "雞舍": version.farm,
                "更新日期": version.updated_text,
                "大小 (MB)": f"{version.path.stat().st_size / 1024 / 1024:.2f}",
                "版本雜湊": secure_sync.file_hash12(version.path),
                "狀態": t("使用中") if version.path == in_use else "",
            })
    if rows:
        st.dataframe(pd.DataFrame(t_records(rows)), width="stretch", hide_index=True,
                     height=min(36 * (len(rows) + 1) + 4, 300))
    else:
        st.warning(t("找不到任何模型檔"))

    choose_column, import_column = st.columns(2, gap="medium")
    with choose_column, st.container(border=True, key="card_model_select"):
        st.markdown(f"**{t('使用中的版本')}**")
        for role in REVIEW_ROLES:
            versions = model_registry.list_versions(role)
            options = [model_registry.AUTO] + [v.path.name for v in versions]
            current = model_registry.selection(role)
            choice = st.selectbox(
                role.upper(),
                options,
                index=options.index(current) if current in options else 0,
                format_func=lambda name: t("自動（最新日期）") if name == model_registry.AUTO else name,
                key=f"model_choice_{role}",
            )
            if choice != current:
                model_registry.set_active(role, choice)
                _model_flash("success", t("{role} 模型已切換為 {name}", role=role.upper(),
                                    name=model_registry.active_path(role).name))
                st.rerun()
        originals = [role for role in REVIEW_ROLES
                     if model_registry.LEGACY_PATHS[role].is_file()
                     and all(v.legacy for v in model_registry.list_versions(role))]
        if originals and st.button(t("把目前的模型加入版本清單"), width="stretch", key="model_register_original"):
            farm = _default_farm_name()
            if not farm:
                _model_flash("warning", t("請先完成連線設定（場域代號），或用右側「匯入新模型」並填寫雞舍名稱"))
                st.rerun()
            try:
                for role in originals:
                    model_registry.register_original(role, farm, _validate_model)
                _model_flash("success", t("已加入版本清單"))
            except Exception as exc:
                _model_flash("error", t("加入失敗：{error}", error=exc))
            st.rerun()

    with import_column, st.container(border=True, key="card_model_import"):
        st.markdown(f"**{t('匯入新模型')}**")
        with st.form("model_import_form", clear_on_submit=True):
            role = st.segmented_control(t("模型"), ["COMB", "SHANK"], default="COMB", key="model_import_role")
            uploaded = st.file_uploader(t("模型檔 (.pt)"), type=["pt"], key="model_import_file")
            farm_column, date_column = st.columns(2)
            farm = farm_column.text_input(t("雞舍名稱"), value=_default_farm_name())
            updated = date_column.date_input(t("模型更新日期"), value=date.today())
            submitted = st.form_submit_button(t("匯入"), type="primary", width="stretch")
        if submitted:
            if uploaded is None or not role:
                st.warning(t("請選擇模型檔"))
            elif not farm.strip():
                st.warning(t("請輸入雞舍名稱"))
            else:
                try:
                    with st.spinner(t("驗證並儲存模型中…")):
                        path = model_registry.import_model(role.lower(), uploaded.getvalue(), farm, updated,
                                                           _validate_model)
                    _model_flash("success", t("已匯入 {name}", name=path.name))
                    st.rerun()
                except Exception as exc:
                    st.error(t("匯入失敗：{error}", error=exc))
        st.caption(t("檔名：`best_<模型>_<雞舍名稱>_<日期>.pt`；舊版本保留，可隨時切回。"))


def _model_flash(kind: str, message: str) -> None:
    st.session_state["model_flash"] = (kind, message)


def _default_farm_name() -> str:
    try:
        return SyncConfig.from_environment().site_id or ""
    except Exception:
        return ""
