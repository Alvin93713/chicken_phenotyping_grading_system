"""Chicken phenotype grading logic."""

try:
    from i18n import t
except Exception:  # pragma: no cover - i18n/streamlit unavailable
    def t(text, **values):
        return text.format(**values) if values else text

FEATURE_COLUMNS = {
    "weight_g": "體重",
    "comb_area_cm2": "雞冠面積",
    "shank_width_cm": "腳脛寬度",
    "shank_length_cm": "腳脛長度",
}

GRADE_PASS = "通過"
GRADE_STANDBY = "備用"
GRADE_FAIL = "不通過"

LIGHT_GREEN = "GREEN"
LIGHT_YELLOW = "YELLOW"
LIGHT_RED = "RED"


def is_feature_enabled(feature, pass_thresholds, standby_thresholds):
    """A feature is active only when either threshold is greater than zero."""
    return float(pass_thresholds.get(feature, 0) or 0) > 0 or float(standby_thresholds.get(feature, 0) or 0) > 0


def active_features(selected_features, pass_thresholds, standby_thresholds):
    return [
        feature
        for feature in selected_features
        if is_feature_enabled(feature, pass_thresholds, standby_thresholds)
    ]


def validate_grading_config(selected_features, pass_thresholds, standby_thresholds):
    """Validate selected grading features and two-level thresholds."""
    if not selected_features:
        raise ValueError(t("請至少選擇一個分級特徵。"))

    enabled = active_features(selected_features, pass_thresholds, standby_thresholds)
    if not enabled:
        raise ValueError(t("請至少設定一個大於 0 的通過門檻或備用門檻。"))

    for feature in selected_features:
        if feature not in FEATURE_COLUMNS:
            raise ValueError(t("不支援的分級欄位：{feature}", feature=feature))
        pass_value = pass_thresholds.get(feature)
        standby_value = standby_thresholds.get(feature)
        if pass_value is None:
            raise ValueError(t("請設定 {feature} 的通過門檻。", feature=t(FEATURE_COLUMNS[feature])))
        if standby_value is None:
            raise ValueError(t("請設定 {feature} 的備用門檻。", feature=t(FEATURE_COLUMNS[feature])))
        if pass_value < 0 or standby_value < 0:
            raise ValueError(t("{feature} 門檻不可小於 0。", feature=t(FEATURE_COLUMNS[feature])))
        if standby_value > pass_value:
            raise ValueError(t("{feature} 的備用門檻不可高於通過門檻。", feature=t(FEATURE_COLUMNS[feature])))


def grade_details(chicken, enabled_features, pass_thresholds, standby_thresholds):
    """Grade one chicken against an already validated config.
    Returns (grade, light, below_pass_features, below_standby_features); a missing value is below both."""
    below_pass, below_standby = [], []
    for feature in enabled_features:
        value = chicken.get(feature)
        if value is None or value == "":
            below_pass.append(feature)
            below_standby.append(feature)
            continue
        number = float(value)
        if number < float(pass_thresholds[feature]):
            below_pass.append(feature)
        if number < float(standby_thresholds[feature]):
            below_standby.append(feature)
    if not below_pass:
        return GRADE_PASS, LIGHT_GREEN, below_pass, below_standby
    if not below_standby:
        return GRADE_STANDBY, LIGHT_YELLOW, below_pass, below_standby
    return GRADE_FAIL, LIGHT_RED, below_pass, below_standby


def grade_and_store_with_latest_config(chicken):
    """Grade one chicken row with its breed's latest saved thresholds and store the result.
    Returns (grade, light), or (None, None) when no thresholds have been saved for that breed yet."""
    from database import get_chicken, get_latest_grading_config, update_grading_result

    breed = chicken.get("breed") or (get_chicken(chicken["chicken_id"]) or {}).get("breed")
    config = get_latest_grading_config(breed)
    if not config:
        return None, None
    selected_features, pass_thresholds, standby_thresholds = config
    grade, light = grade_chicken(chicken, selected_features, pass_thresholds, standby_thresholds)
    update_grading_result(chicken["chicken_id"], selected_features, pass_thresholds, standby_thresholds, grade, light)
    return grade, light


def grade_chicken(chicken, selected_features, pass_thresholds, standby_thresholds):
    """Grade one chicken and return grade plus light result."""
    validate_grading_config(selected_features, pass_thresholds, standby_thresholds)
    enabled = active_features(selected_features, pass_thresholds, standby_thresholds)
    grade, light, _, _ = grade_details(chicken, enabled, pass_thresholds, standby_thresholds)
    return grade, light


def summarize_results(results):
    """Summarize three-level grading results."""
    total = len(results)
    pass_count = sum(1 for item in results if item["grade_result"] == GRADE_PASS)
    standby_count = sum(1 for item in results if item["grade_result"] == GRADE_STANDBY)
    fail_count = sum(1 for item in results if item["grade_result"] == GRADE_FAIL)

    return {
        "total": total,
        "pass_count": pass_count,
        "standby_count": standby_count,
        "fail_count": fail_count,
        "pass_rate": (pass_count / total * 100) if total else 0,
        "standby_rate": (standby_count / total * 100) if total else 0,
        "fail_rate": (fail_count / total * 100) if total else 0,
    }
