"""Streamlit pages for product quality monitoring and lightweight review."""
from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path

import pandas as pd
import streamlit as st

from dual_camera import MODEL_PATHS
from product_review import (
    export_review_bundle,
    get_case,
    list_cases,
    review_case,
    review_summary,
    seed_demo_cases,
)


def _display_local_image(path: Path) -> None:
    """Embed local QA images without relying on Streamlit's media endpoint."""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        f'<img src="data:{mime_type};base64,{encoded}" '
        'style="display:block;width:100%;aspect-ratio:16/9;object-fit:contain;'
        'background:#151922;border:1px solid #343a46;border-radius:6px;" />',
        unsafe_allow_html=True,
    )


def _model_rows() -> list[dict[str, object]]:
    rows = []
    for role, path in MODEL_PATHS.items():
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            rows.append({"模型": role.upper(), "檔案": path.name, "版本雜湊": digest, "大小 (MB)": round(path.stat().st_size / 1024 / 1024, 2), "狀態": "已載入"})
        else:
            rows.append({"模型": role.upper(), "檔案": path.name, "版本雜湊": "-", "大小 (MB)": "-", "狀態": "缺少檔案"})
    return rows


def quality_dashboard_page(page_heading) -> None:
    page_heading("模型品質監控", "集中追蹤低信心、推論失敗與人工確認結果。")
    summary = review_summary()
    metrics = (
        ("待審核", summary["待審核"]),
        ("確認正確", summary["正確"]),
        ("確認錯誤", summary["不正確"]),
        ("無法判斷", summary["無法判斷"]),
        ("總案例", summary["總案例"]),
    )
    for (label, value), column in zip(metrics, st.columns(5)):
        column.metric(label, value)

    st.subheader("目前模型")
    st.dataframe(pd.DataFrame(_model_rows()), use_container_width=True, hide_index=True, height=112)

    cases = list_cases()
    st.subheader("最近品質事件")
    if not cases:
        st.info("尚無品質事件。")
        return
    rows = [
        {
            "case ID": case["case_id"],
            "chicken ID": case["chicken_id"],
            "優先度": case["priority"],
            "原因": case["reason"],
            "審核狀態": case["review_state"],
            "最低信心": case["minimum_confidence"],
            "建立時間": case["created_at"],
        }
        for case in cases[:20]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True, height=270)


def review_cases_page(page_heading) -> None:
    page_heading("案例審核", "操作人員只需確認結果，不需繪製 segmentation mask。")
    status = st.segmented_control("審核篩選", ["待審核", "不正確", "無法判斷", "正確", "全部"], default="待審核")
    cases = list_cases(status)
    if not cases:
        st.info("目前沒有符合條件的案例。")
        return

    labels = {case["case_id"]: f"{case['chicken_id']} · {case['reason']} · {case['created_at']}" for case in cases}
    selected_id = st.selectbox("案例", list(labels), format_func=labels.get)
    case = get_case(selected_id)
    if not case:
        st.error("找不到案例資料。")
        return

    info_columns = st.columns(4)
    info_columns[0].metric("Chicken ID", case["chicken_id"])
    info_columns[1].metric("優先度", case["priority"])
    confidence = case["minimum_confidence"]
    info_columns[2].metric("最低信心", "-" if confidence is None else f"{float(confidence):.3f}")
    info_columns[3].metric("目前狀態", case["review_state"])
    if case["error"]:
        st.error(case["error"])

    for role, column in zip(("comb", "shank"), st.columns(2)):
        result_path = Path(case["result_paths"].get(role, ""))
        source_path = Path(case["source_paths"].get(role, ""))
        display_path = result_path if result_path.is_file() else source_path
        with column:
            st.markdown(f"**{role.upper()}**")
            if display_path.is_file():
                _display_local_image(display_path)
            else:
                st.warning("影像不存在")
            role_confidence = case["role_confidences"].get(role)
            st.caption("信心：-" if role_confidence is None else f"信心：{float(role_confidence):.3f}")

    note = st.text_input("備註", value=str(case["reviewer_note"] or ""), key=f"review_note_{selected_id}")
    columns = st.columns(3)
    actions = (("正確", "primary"), ("不正確", "secondary"), ("無法判斷", "secondary"))
    for (decision, button_type), column in zip(actions, columns):
        if column.button(decision, type=button_type, use_container_width=True, key=f"decision_{decision}_{selected_id}"):
            review_case(selected_id, decision, note)
            st.rerun()


def model_maintenance_page(page_heading) -> None:
    page_heading("資料與模型維護", "匯出錯誤案例，由供應商完成精確標註、驗證與模型發布。")
    summary = review_summary()
    metric_columns = st.columns(3)
    metric_columns[0].metric("可匯出錯誤案例", summary["不正確"])
    metric_columns[1].metric("可匯出不確定案例", summary["無法判斷"])
    metric_columns[2].metric("待審核", summary["待審核"])

    left, right = st.columns(2)
    with left:
        st.subheader("流程展示")
        if st.button("載入三筆示範案例", use_container_width=True):
            seed_demo_cases()
            st.rerun()
    with right:
        st.subheader("供應商資料包")
        bundle = export_review_bundle()
        st.download_button(
            "下載待標註資料包",
            data=bundle,
            file_name="chicken_review_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            disabled=summary["不正確"] + summary["無法判斷"] == 0,
        )

    st.subheader("發布中的模型")
    st.dataframe(pd.DataFrame(_model_rows()), use_container_width=True, hide_index=True, height=112)
    st.info("模型訓練與發布由供應商管理；雞舍端只負責確認案例。")
