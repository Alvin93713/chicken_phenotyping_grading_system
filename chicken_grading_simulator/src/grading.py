"""Chicken phenotype grading logic."""

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
        raise ValueError("請至少選擇一個分級特徵。")

    enabled = active_features(selected_features, pass_thresholds, standby_thresholds)
    if not enabled:
        raise ValueError("請至少設定一個大於 0 的通過門檻或備用門檻。")

    for feature in selected_features:
        if feature not in FEATURE_COLUMNS:
            raise ValueError(f"不支援的分級欄位：{feature}")
        pass_value = pass_thresholds.get(feature)
        standby_value = standby_thresholds.get(feature)
        if pass_value is None:
            raise ValueError(f"請設定 {FEATURE_COLUMNS[feature]} 的通過門檻。")
        if standby_value is None:
            raise ValueError(f"請設定 {FEATURE_COLUMNS[feature]} 的備用門檻。")
        if pass_value < 0 or standby_value < 0:
            raise ValueError(f"{FEATURE_COLUMNS[feature]} 門檻不可小於 0。")
        if standby_value > pass_value:
            raise ValueError(f"{FEATURE_COLUMNS[feature]} 的備用門檻不可高於通過門檻。")


def grade_chicken(chicken, selected_features, pass_thresholds, standby_thresholds):
    """Grade one chicken and return grade plus light result."""
    validate_grading_config(selected_features, pass_thresholds, standby_thresholds)

    enabled = active_features(selected_features, pass_thresholds, standby_thresholds)
    all_pass = True
    all_standby = True
    for feature in enabled:
        value = chicken.get(feature)
        if value is None or value == "":
            return GRADE_FAIL, LIGHT_RED
        number = float(value)
        all_pass = all_pass and number >= float(pass_thresholds[feature])
        all_standby = all_standby and number >= float(standby_thresholds[feature])

    if all_pass:
        return GRADE_PASS, LIGHT_GREEN
    if all_standby:
        return GRADE_STANDBY, LIGHT_YELLOW
    return GRADE_FAIL, LIGHT_RED


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
