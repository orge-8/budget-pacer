"""纯函数与节奏控制律的单元测试。

不依赖宿主运行时，只覆盖可独立验证的逻辑：月窗口、标签解析、序列提取、控制律分支。
"""

from datetime import datetime

import pytest

SEP15 = datetime(2026, 9, 15, 12, 0, 0)
SEP_START = datetime(2026, 9, 1)
OCT_START = datetime(2026, 10, 1)


# ---------- 基础工具 ----------


def test_clamp_basic(bp):
    assert bp.clamp(5.0, 0.0, 1.0) == 1.0
    assert bp.clamp(-5.0, 0.0, 1.0) == 0.0
    assert bp.clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_tolerates_reversed_bounds(bp):
    """传入颠倒的上下界不应抛错，应交换后夹取。"""
    assert bp.clamp(0.5, 1.0, 0.0) == 0.5


# ---------- 月窗口 ----------


def test_month_bounds_normal(bp):
    start, end = bp.month_bounds(SEP15)
    assert start == SEP_START
    assert end == OCT_START


def test_month_bounds_december_wraps_year(bp):
    """12 月要跨年，不能变成 13 月。"""
    start, end = bp.month_bounds(datetime(2026, 12, 20))
    assert start == datetime(2026, 12, 1)
    assert end == datetime(2027, 1, 1)


def test_month_bounds_february(bp):
    start, end = bp.month_bounds(datetime(2026, 2, 10))
    assert start == datetime(2026, 2, 1)
    assert end == datetime(2026, 3, 1)


def test_month_key(bp):
    assert bp.month_key(SEP15) == "2026-09"
    assert bp.month_key(datetime(2026, 12, 31)) == "2026-12"


def test_month_progress_midpoint(bp):
    progress, raw, floor = bp.month_progress(SEP15, 1.0)
    assert raw == pytest.approx(14.5 / 30.0, abs=1e-6)
    assert progress == pytest.approx(raw, abs=1e-6)
    assert floor == pytest.approx(1.0 / 30.0, abs=1e-6)


def test_month_progress_floor_protects_month_start(bp):
    """月初原始进度极小，必须被下限抬起来，否则节奏比会被放大成噪声。"""
    early = datetime(2026, 9, 1, 2, 0, 0)
    progress, raw, floor = bp.month_progress(early, 1.0)
    assert raw < floor
    assert progress == pytest.approx(floor, abs=1e-9)


def test_month_progress_zero_floor_days(bp):
    """下限为 0 时退回原始进度。"""
    early = datetime(2026, 9, 1, 2, 0, 0)
    progress, raw, _floor = bp.month_progress(early, 0.0)
    assert progress == pytest.approx(raw, abs=1e-9)


# ---------- 时间标签解析 ----------


def test_parse_label_primary_format(bp):
    """Host 的天/小时桶主力格式。"""
    assert bp._parse_label("2026-09-15 10:00:00") == datetime(2026, 9, 15, 10)


def test_parse_label_iso_variants(bp):
    assert bp._parse_label("2026-09-15T10:00:00") == datetime(2026, 9, 15, 10)
    assert bp._parse_label("2026-09-15") == datetime(2026, 9, 15)


def test_parse_label_with_microseconds(bp):
    """llm_usage.timestamp 可能带微秒。"""
    parsed = bp._parse_label("2026-09-15 10:00:00.123456")
    assert parsed is not None
    assert parsed.replace(microsecond=0) == datetime(2026, 9, 15, 10)


def test_parse_label_epoch_seconds_and_millis(bp):
    seconds = bp._parse_label("1757908800")
    millis = bp._parse_label("1757908800000")
    assert seconds is not None and millis is not None
    assert abs((seconds - millis).total_seconds()) < 2


def test_parse_label_without_year_fills_current_year(bp):
    parsed = bp._parse_label("09-15 10:00")
    assert parsed is not None
    assert (parsed.month, parsed.day) == (9, 15)
    assert parsed.year == datetime.now().year


def test_parse_label_rejects_garbage(bp):
    assert bp._parse_label("") is None
    assert bp._parse_label("   ") is None
    assert bp._parse_label("not-a-date") is None


# ---------- 窗口内求和 ----------


def test_label_in_window_excludes_boundaries(bp):
    assert bp._label_in_window("2026-09-01 00:00:00", SEP_START, OCT_START)
    assert bp._label_in_window("2026-09-30 23:00:00", SEP_START, OCT_START)
    assert not bp._label_in_window("2026-10-01 00:00:00", SEP_START, OCT_START)
    assert not bp._label_in_window("2026-08-31 23:00:00", SEP_START, OCT_START)


def test_window_bucket_filters_cross_month(bp):
    """上月与下月的数据都不能算进本月。"""
    timestamps = [
        "2026-08-31 23:00:00",
        "2026-09-01 00:00:00",
        "2026-09-30 23:00:00",
        "2026-10-01 00:00:00",
    ]
    values = [9.9, 1.0, 2.0, 3.0]
    assert bp.window_bucket(timestamps, values, SEP_START, OCT_START) == pytest.approx(3.0)


def test_window_bucket_ignores_unparsable_labels(bp):
    """宁可少报也不误报：无法解析的桶直接不计入。"""
    timestamps = ["garbage", "2026-09-10 00:00:00"]
    values = [100.0, 1.5]
    assert bp.window_bucket(timestamps, values, SEP_START, OCT_START) == pytest.approx(1.5)


def test_window_bucket_handles_bad_values(bp):
    timestamps = ["2026-09-10 00:00:00", "2026-09-11 00:00:00"]
    values = ["oops", 2.0]
    assert bp.window_bucket(timestamps, values, SEP_START, OCT_START) == pytest.approx(2.0)


def test_window_bucket_empty_inputs(bp):
    assert bp.window_bucket([], [], SEP_START, OCT_START) == 0.0
    assert bp.window_bucket(None, None, SEP_START, OCT_START) == 0.0
    # values 是字典说明结构不对，应安全返回 0
    assert bp.window_bucket(["2026-09-10"], {"a": 1}, SEP_START, OCT_START) == 0.0


def test_window_bucket_short_labels(bp):
    """标签少于数值时，缺失标签的项不计入。"""
    assert bp.window_bucket(["2026-09-10 00:00:00"], [1.0, 2.0], SEP_START, OCT_START) == pytest.approx(1.0)


# ---------- 序列提取 ----------


def test_unwrap_series_both_shapes(bp):
    inner = {"timestamps": [], "values_by_key": {}}
    assert bp.unwrap_series(inner) == inner
    assert bp.unwrap_series({"success": True, "series": inner}) == inner
    assert bp.unwrap_series(None) == {}


def test_series_total_sums_all_models(bp):
    raw = {
        "timestamps": ["2026-09-10 00:00:00", "2026-09-11 00:00:00", "2026-08-01 00:00:00"],
        "values_by_key": {"a": [1.0, 2.0, 99.0], "b": [0.5, 0.5, 99.0]},
    }
    assert bp.series_total(raw, SEP_START, OCT_START) == pytest.approx(4.0)


def test_series_total_accepts_wrapped_shape(bp):
    """兼容外层仍包着 success/series 的返回形态。"""
    raw = {
        "success": True,
        "series": {
            "timestamps": ["2026-09-10 00:00:00"],
            "values_by_key": {"a": [2.5]},
        },
    }
    assert bp.series_total(raw, SEP_START, OCT_START) == pytest.approx(2.5)


def test_series_total_falls_back_to_flat_values(bp):
    raw = {"timestamps": ["2026-09-10 00:00:00"], "values": [7.0]}
    assert bp.series_total(raw, SEP_START, OCT_START) == pytest.approx(7.0)


def test_series_total_handles_empty(bp):
    assert bp.series_total({}, SEP_START, OCT_START) == 0.0
    assert bp.series_total(None, SEP_START, OCT_START) == 0.0
    assert bp.series_total({"values_by_key": {}}, SEP_START, OCT_START) == 0.0


def test_series_by_model_maps_labels(bp):
    raw = {
        "timestamps": ["2026-09-10 00:00:00", "2026-09-11 00:00:00"],
        "values_by_key": {"m1": [1.0, 2.0], "m2": [3.0, 4.0]},
        "labels_by_key": {"m1": "gpt-4", "m2": "glm-4"},
    }
    result = bp.series_by_model(raw, SEP_START, OCT_START)
    assert result == {"gpt-4": pytest.approx(3.0), "glm-4": pytest.approx(7.0)}


def test_series_by_model_falls_back_to_key(bp):
    raw = {"timestamps": ["2026-09-10 00:00:00"], "values_by_key": {"m1": [2.0]}}
    assert bp.series_by_model(raw, SEP_START, OCT_START) == {"m1": pytest.approx(2.0)}


def test_series_by_model_invalid_shape(bp):
    assert bp.series_by_model(None, SEP_START, OCT_START) == {}
    assert bp.series_by_model({"values_by_key": "nope"}, SEP_START, OCT_START) == {}


def test_coerce_datetime_accepts_both_types(bp):
    assert bp.coerce_datetime(datetime(2026, 9, 15, 10)) == datetime(2026, 9, 15, 10)
    assert bp.coerce_datetime("2026-09-15 10:00:00") == datetime(2026, 9, 15, 10)
    assert bp.coerce_datetime(12345) is None
    assert bp.coerce_datetime("garbage") is None


# ---------- 控制律 ----------


def test_compute_adjust_disabled_budget(bp):
    """预算为 0 表示不调节。"""
    decision = bp.compute_adjust(10.0, 0.0, SEP15)
    assert decision.reason == "budget_disabled"
    assert decision.target == 1.0


def test_compute_adjust_on_track_equal_progress(bp):
    """花费进度与时间进度一致时不动手。"""
    decision = bp.compute_adjust(14.5, 30.0, SEP15)
    assert decision.reason == "on_track"
    assert decision.target == 1.0


def test_compute_adjust_deadband_absorbs_small_deviation(bp):
    """死区内不做调整，避免抖动。"""
    decision = bp.compute_adjust(15.5, 30.0, SEP15)
    assert abs(decision.pacing - 1.0) <= 0.10
    assert decision.reason == "on_track"


def test_compute_adjust_brakes_just_past_deadband(bp):
    """刚越过死区就应开始压低——这是死区的边界。"""
    decision = bp.compute_adjust(16.0, 30.0, SEP15)
    assert decision.pacing > 1.10
    assert decision.reason == "brake"


def test_compute_adjust_brakes_when_overspending(bp):
    """花费超前应压低倍率。"""
    decision = bp.compute_adjust(24.0, 30.0, SEP15)
    assert decision.reason == "brake"
    assert 0.0 < decision.target < 1.0
    assert decision.target == pytest.approx(1.0 / decision.pacing, abs=1e-6)


def test_compute_adjust_coasts_without_boost(bp):
    """未启用加频时，花费滞后也只保持 1.0。"""
    decision = bp.compute_adjust(3.0, 30.0, SEP15)
    assert decision.reason == "coast"
    assert decision.target == 1.0


def test_compute_adjust_boosts_when_enabled(bp):
    """启用加频且放宽上限时，花费滞后会提高倍率。"""
    decision = bp.compute_adjust(3.0, 30.0, SEP15, allow_boost=True, max_adjust=5.0)
    assert decision.reason == "boost"
    assert decision.target > 1.0


def test_compute_adjust_boost_respects_max_cap(bp):
    decision = bp.compute_adjust(0.1, 30.0, SEP15, allow_boost=True, max_adjust=1.5)
    assert decision.target == pytest.approx(1.5)


def test_compute_adjust_over_budget_gate(bp):
    """达到预算立即走闸门，不看进度。"""
    decision = bp.compute_adjust(30.0, 30.0, SEP15, over_budget_adjust=0.2)
    assert decision.reason == "over_budget"
    assert decision.target == pytest.approx(0.2)


def test_compute_adjust_over_budget_gate_overrides_min_adjust(bp):
    """超支闸门优先于 min_adjust：设了闸门就等于允许比底线更安静。"""
    decision = bp.compute_adjust(60.0, 30.0, SEP15, min_adjust=0.3, over_budget_adjust=0.0)
    assert decision.target == pytest.approx(0.0)


def test_compute_adjust_min_adjust_holds_when_not_over_budget(bp):
    """未超支时 min_adjust 才是真正的底线，倍率不会被压到它之下。"""
    # 月初进度小、花费已很高但还没超预算 -> 节奏比极大，倍率会被夹到 min_adjust
    decision = bp.compute_adjust(29.0, 30.0, datetime(2026, 9, 2, 12, 0, 0), min_adjust=0.25)
    assert decision.reason == "brake"
    assert decision.target == pytest.approx(0.25)


def test_compute_adjust_over_budget_can_silence(bp):
    """min_adjust 与闸门都为 0 时允许完全静音。"""
    decision = bp.compute_adjust(45.0, 30.0, SEP15, min_adjust=0.0, over_budget_adjust=0.0)
    assert decision.target == 0.0


def test_compute_adjust_brake_hits_min_floor(bp):
    """极端超前的倍率应被 min_adjust 夹住，不会压到负数或零。"""
    decision = bp.compute_adjust(29.0, 30.0, datetime(2026, 9, 1, 1, 0, 0), min_adjust=0.25)
    if decision.reason == "brake":
        assert decision.target == pytest.approx(0.25)


def test_compute_adjust_month_start_floor_prevents_alarm(bp):
    """月初花了少量钱不该被判成严重超前——这正是 p_floor 的作用。"""
    early = datetime(2026, 9, 1, 2, 0, 0)
    decision = bp.compute_adjust(1.0, 30.0, early, p_floor_days=1.0)
    assert decision.reason == "on_track"

    # 关掉下限保护后，同样的花费会被误判为严重超前
    unprotected = bp.compute_adjust(1.0, 30.0, early, p_floor_days=0.0)
    assert unprotected.pacing > 10.0


def test_compute_adjust_records_progress_detail(bp):
    decision = bp.compute_adjust(15.0, 30.0, SEP15)
    assert decision.spend == pytest.approx(15.0)
    assert decision.budget == pytest.approx(30.0)
    assert decision.spend_ratio == pytest.approx(0.5)
    assert 0.0 < decision.progress <= 1.0
    assert decision.raw_progress <= decision.progress


# ---------- 平滑 ----------


def test_smooth_adjust_endpoints(bp):
    assert bp.smooth_adjust(1.0, 1.0, 0.35) == pytest.approx(1.0)
    assert bp.smooth_adjust(1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert bp.smooth_adjust(1.0, 0.0, 1.0) == pytest.approx(0.0)


def test_smooth_adjust_clamps_alpha(bp):
    assert bp.smooth_adjust(1.0, 0.0, 5.0) == pytest.approx(0.0)
    assert bp.smooth_adjust(1.0, 0.0, -3.0) == pytest.approx(1.0)


def test_smooth_adjust_converges_monotonically(bp):
    """连续平滑应逐步逼近目标，不越过。"""
    value = 1.0
    for _ in range(50):
        value = bp.smooth_adjust(value, 0.2, 0.35)
    assert value == pytest.approx(0.2, abs=0.01)


# ---------- 配置解析 ----------


def test_coerce_id_list_variants(bp):
    assert bp.coerce_id_list(None) == []
    assert bp.coerce_id_list(["1", "2"]) == ["1", "2"]
    assert bp.coerce_id_list("1,2") == ["1", "2"]
    assert bp.coerce_id_list("1，2；3") == ["1", "2", "3"]
    assert bp.coerce_id_list(123456789) == ["123456789"]
    assert bp.coerce_id_list("   ") == []


def test_overrides_normalized_from_loose_string(bp):
    """例外额度允许写成宽松字符串，避免用户配置被判非法。"""
    model = bp.BudgetPacerConfig(overrides="123456:5.0，789012:3.0")
    assert len(model.overrides) == 2
    assert model.overrides[0].group_id == "123456"
    assert model.overrides[0].monthly_limit == pytest.approx(5.0)


def test_overrides_defaults_empty(bp):
    model = bp.BudgetPacerConfig()
    assert model.overrides == []
    assert model.budget.monthly_limit == pytest.approx(30.0)
    assert model.budget.allow_boost is False


def test_config_admin_ids_normalized(bp):
    model = bp.BudgetPacerConfig(permission={"admin_ids": "123456789"})
    assert model.permission.admin_ids == ["123456789"]


# ---------- 状态文案 ----------


def test_format_status_contains_key_metrics(bp):
    decision = bp.compute_adjust(24.0, 30.0, SEP15)
    text = bp.format_status(decision, 0.6)
    assert "月度预算" in text
    assert "本月已花" in text
    assert "节奏比" in text
    assert "当前倍率" in text


def test_format_status_includes_override_note(bp):
    decision = bp.compute_adjust(15.0, 30.0, SEP15)
    text = bp.format_status(decision, 1.0, "例外额度：群 123 → 5.0 元")
    assert "例外额度" in text
