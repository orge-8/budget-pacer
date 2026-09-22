"""上线前审计用例：数值健壮性、并发安全、降级可见性、配置容错。

这些用例断言「应该有防护的地方」。断言失败 = 一项真实缺陷，不是测试写错——
先看失败项的实际值，再决定改代码还是改断言。

与 test_pacing.py 的分工：那里测「正常路径算得对」，这里测「异常输入不会静默出错」。
"""

import asyncio
import functools
import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

SEP15 = datetime(2026, 9, 15, 12, 0, 0)


def async_test(fn):
    """把 async 用例包成同步函数。

    不引入 pytest-asyncio 依赖：本项目的测试环境里没有这个插件，
    缺了它 async 用例会被 pytest 直接跳过/报错，看起来像产品缺陷。

    必须用 functools.wraps：pytest 靠 inspect.signature 决定注入哪些 fixture，
    手写 wrapper(*args, **kwargs) 会让它识别不出 bp 参数。
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


# ---------- 可观测性：收集日志，用于断言「降级必须留痕」 ----------


class RecordingLogger:
    def __init__(self):
        self.records = []

    def _record(self, level, msg, *args, **kwargs):
        try:
            text = msg % args if args else str(msg)
        except Exception:
            text = f"{msg} {args}"
        self.records.append((level, text))

    def info(self, msg, *args, **kwargs):
        self._record("info", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._record("warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._record("error", msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._record("debug", msg, *args, **kwargs)

    def text(self, level=None):
        return "\n".join(t for lv, t in self.records if level is None or lv == level)

    def has(self, keyword, level=None):
        return keyword in self.text(level)


# ---------- 假宿主 ----------


class FakeFrequency:
    def __init__(self, fail_ids=(), silent_ids=()):
        self.calls = []
        self.fail_ids = set(fail_ids)
        self.silent_ids = set(silent_ids)

    async def set_adjust(self, chat_id, value):
        self.calls.append((chat_id, value))
        if chat_id in self.fail_ids:
            raise RuntimeError("session not found")
        if chat_id in silent_ids:
            return False
        return True


class FakeLocalStatistics:
    def __init__(self, payload):
        self.payload = payload
        self.raise_error = None

    async def model_trend(self, **kwargs):
        if self.raise_error is not None:
            raise self.raise_error
        return self.payload


class FakeChat:
    def __init__(self, streams):
        self.streams = streams

    async def get_all_streams(self, platform="qq"):
        return self.streams


class FakeDB:
    def __init__(self, rows=None):
        self.rows = rows or []

    async def query(self, **kwargs):
        return self.rows


class FakeSend:
    def __init__(self):
        self.sent = []

    async def text(self, content, stream_id):
        self.sent.append((stream_id, content))
        return True


class FakePaths:
    def __init__(self, data_dir):
        self.data_dir = data_dir


class FakeContext:
    def __init__(self, payload, streams=None, data_dir=None, db_rows=None):
        self.logger = RecordingLogger()
        self.frequency = FakeFrequency()
        self.chat = FakeChat(streams if streams is not None else basic_streams())
        self.send = FakeSend()
        self.db = FakeDB(db_rows)
        self.statistics = type("S", (), {"local": FakeLocalStatistics(payload)})()
        self.paths = FakePaths(data_dir or tempfile.mkdtemp(prefix="bp-audit-"))


def basic_streams():
    return [
        {"session_id": "g1", "group_id": "10001", "is_group_session": True},
        {"session_id": "g2", "group_id": "10002", "is_group_session": True},
    ]


def empty_series():
    return {"timestamps": [], "values_by_key": {}, "labels_by_key": {}}


def series_with(cost, now=None):
    now = now or datetime.now()
    label = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    return {"timestamps": [label], "values_by_key": {"m": [cost]}, "labels_by_key": {"m": "m"}}


def make(bp, payload=None, config=None, streams=None, data_dir=None, db_rows=None):
    plugin = bp.BudgetPacerPlugin()
    ctx = FakeContext(if_none(payload, series_with(0.0)), streams, data_dir, db_rows)
    plugin._set_context(ctx)
    body = dict(config or {})
    section = dict(body.get("plugin") or {})
    section.setdefault("config_version", "1")
    body["plugin"] = section
    plugin.set_plugin_config(body)
    return plugin, ctx


def if_none(value, fallback):
    return fallback if value is None else value


# ================= A. 数值健壮性 =================


def test_adjust_is_finite_for_nan_budget(bp):
    """预算被填成 NaN 时不能把 NaN 透传给宿主。"""
    decision = bp.compute_adjust(15.0, float("nan"), SEP15)
    assert math.isfinite(decision.target), f"target 非有限：{decision.target}"


def test_adjust_is_finite_for_inf_budget(bp):
    decision = bp.compute_adjust(15.0, float("inf"), SEP15)
    assert math.isfinite(decision.target), f"target 非有限：{decision.target}"


def test_adjust_is_finite_for_nan_spend(bp):
    """花费统计异常可能是 NaN（0/0 之类），不能污染倍率。"""
    decision = bp.compute_adjust(float("nan"), 30.0, SEP15)
    assert math.isfinite(decision.target), f"target 非有限：{decision.target}"


def test_adjust_is_finite_for_inf_spend(bp):
    decision = bp.compute_adjust(float("inf"), 30.0, SEP15)
    assert math.isfinite(decision.target), f"target 非有限：{decision.target}"


def test_adjust_is_finite_for_negative_spend(bp):
    decision = bp.compute_adjust(-5.0, 30.0, SEP15)
    assert math.isfinite(decision.target)
    assert 0.0 <= decision.target <= 1.0


def test_adjust_is_finite_for_nan_tuning_params(bp):
    """调参项被填成 NaN 时也不能产出 NaN 倍率。"""
    for field in ("deadband", "min_adjust", "max_adjust", "p_floor_days"):
        kwargs = {field: float("nan")}
        decision = bp.compute_adjust(15.0, 30.0, SEP15, **kwargs)
        assert math.isfinite(decision.target), f"{field}=NaN 导致 target={decision.target}"


def test_adjust_is_finite_for_nan_over_budget_adjust(bp):
    decision = bp.compute_adjust(60.0, 30.0, SEP15, over_budget_adjust=float("nan"))
    assert math.isfinite(decision.target), f"target 非有限：{decision.target}"


def test_target_always_within_bounds(bp):
    """无论输入多离谱，倍率都必须落在 [min_adjust, max_adjust]。"""
    lo, hi = 0.1, 1.0
    cases = [
        (0.0, 30.0), (15.0, 30.0), (299.0, 30.0), (-1.0, 30.0),
        (1e9, 30.0), (0.0, 1e-9), (float("nan"), float("nan")),
    ]
    for spend, budget in cases:
        decision = bp.compute_adjust(spend, budget, SEP15, min_adjust=lo, max_adjust=hi)
        assert math.isfinite(decision.target), f"spend={spend} budget={budget} → {decision.target}"
        assert lo - 1e-9 <= decision.target <= hi + 1e-9, f"越界：{decision.target}"


def test_clamp_handles_nan(bp):
    """clamp 收到 NaN 不应把 NaN 当合法值传出去。"""
    result = bp.clamp(float("nan"), 0.1, 1.0)
    assert math.isfinite(result), f"clamp 透传了 NaN：{result}"


def test_smooth_adjust_handles_nan_previous(bp):
    result = bp.smooth_adjust(float("nan"), 0.5, 0.35)
    assert math.isfinite(result), f"smooth_adjust 透传了 NaN：{result}"


# ================= B. 降级可见性（静默降级 = 故障隐形） =================


@async_test
async def test_empty_statistics_payload_is_logged(bp):
    """取数返回空结构时会被当成「零花费」，必须留痕，否则真机上查无线索。"""
    plugin, ctx = make(bp, payload=empty_series(), config={"budget": {"monthly_limit": 30.0}})
    await plugin._run_once(force=True)

    assert ctx.logger.has("统计", "warning") or ctx.logger.has("花费", "warning") or ctx.logger.has(
        "空", "warning"
    ), f"空统计负载被静默当成零花费，日志无告警。日志：\n{ctx.logger.text()}"


@async_test
async def test_unexpected_statistics_shape_is_logged(bp):
    """结构不符（缺 values_by_key）同样是静默的错误来源。"""
    plugin, ctx = make(bp, payload={"foo": "bar"}, config={"budget": {"monthly_limit": 30.0}})
    await plugin._run_once(force=True)
    assert ctx.logger.has("结构异常", "warning"), f"异常结构未留痕。日志：\n{ctx.logger.text()}"


@async_test
async def test_set_adjust_returning_false_is_counted(bp):
    """宿主返回 False（会话不存在）时应体现在日志里，不能只报成功数。"""
    plugin, ctx = make(
        bp, payload=series_with(150.0), config={"budget": {"monthly_limit": 30.0}}
    )
    ctx.frequency = FakeFrequency(silent_ids={"g1"})
    ctx.frequency.__class__ = FakeFrequency  # 保持类型清晰
    await plugin._run_once(force=True)

    joined = ctx.logger.text()
    assert "1" in joined, "日志应体现下发统计"
    assert ctx.logger.has("失败") or ctx.logger.has("跳过"), (
        f"有会话下发失败但日志未区分。日志：\n{joined}"
    )


@async_test
async def test_all_outbound_failures_are_visible(bp):
    """全部会话都下发失败时，日志必须明确异常，而不是安静收场。"""
    plugin, ctx = make(bp, payload=series_with(150.0), config={"budget": {"monthly_limit": 30.0}})
    ctx.frequency = FakeFrequency(fail_ids={"g1", "g2"})
    await plugin._run_once(force=True)

    assert ctx.logger.has("warning", "warning") or ctx.logger.has("失败"), (
        f"全部下发失败却无告警。日志：\n{ctx.logger.text()}"
    )


# ================= C. 并发安全 =================


@async_test
async def test_concurrent_run_once_does_not_corrupt_state(bp):
    """命令与巡检循环可能同时触发 _run_once，倍率状态不能被交叉污染。"""
    plugin, ctx = make(bp, payload=series_with(150.0), config={"budget": {"monthly_limit": 30.0}})

    await asyncio.gather(
        plugin._run_once(force=True),
        plugin._run_once(force=True),
        plugin._run_once(force=True),
    )

    assert plugin._applied_adjust == pytest.approx(0.1), (
        f"并发调用导致倍率错乱：{plugin._applied_adjust}"
    )


@async_test
async def test_duplicate_on_load_does_not_start_two_loops(bp):
    plugin, _ctx = make(bp, payload=series_with(0.0))
    await plugin.on_load()
    first = plugin._loop_task
    await plugin.on_load()
    second = plugin._loop_task
    assert first is second, "重复 on_load 启动了多个巡检任务"
    await plugin.on_unload()


# ================= D. 配置容错 =================


def test_plugin_loads_with_extreme_config(bp):
    """极端配置不应让插件加载失败——宁可退化到安全值。"""
    model = bp.BudgetPacerConfig(
        budget={
            "monthly_limit": -1.0,
            "check_interval_seconds": 0,
            "deadband": -5.0,
            "smoothing": 99.0,
            "min_adjust": 5.0,
            "max_adjust": -3.0,
            "over_budget_adjust": 42.0,
            "p_floor_days": -10.0,
            "override_scan_limit": -1,
        }
    )
    assert model.budget.monthly_limit == pytest.approx(-1.0)


def test_override_with_bad_values_is_ignored(bp):
    """例外额度里的脏数据应被忽略而不是崩溃。"""
    model = bp.BudgetPacerConfig(overrides="abc:xyz，10001:5.0，:3，，")
    limits = [o.group_id for o in model.overrides]
    assert "10001" in limits


def test_override_missing_limit_is_dropped(bp):
    model = bp.BudgetPacerConfig(overrides="10001")
    assert model.overrides == []


# ================= E. 命令解析健壮性 =================


def test_command_argument_handles_garbage(bp):
    for text in ("", "   ", "/", "／", "/预算", "/预算   ", "///预算 a b c", "x" * 5000):
        result = bp.BudgetPacerPlugin._command_argument(text)
        assert isinstance(result, str)


def test_message_text_does_not_recurse_forever(bp):
    """自引用字典不得导致无限递归。"""
    payload = {"message": {}}
    payload["message"]["message"] = payload  # 构造环
    result = bp.BudgetPacerPlugin._message_text(payload)
    assert isinstance(result, str)


def test_stream_id_extraction_is_safe(bp):
    assert bp.BudgetPacerPlugin._stream_id(None) == ""
    assert bp.BudgetPacerPlugin._stream_id({}) == ""
    assert bp.BudgetPacerPlugin._stream_id({"stream_id": "x"}) == "x"
    assert bp.BudgetPacerPlugin._stream_id({"message": {"session_id": "y"}}) == "y"


# ================= F. 副作用守卫 =================


@async_test
async def test_zero_budget_never_touches_sessions(bp):
    """未配置预算就绝不下发倍率——避免「默认值把 bot 静音」。"""
    plugin, ctx = make(bp, payload=series_with(999.0), config={"budget": {"monthly_limit": 0.0}})
    await plugin._run_once(force=True)
    assert ctx.frequency.calls == [], f"预算为 0 时不该下发：{ctx.frequency.calls}"


@async_test
async def test_paused_plugin_does_not_adjust(bp):
    plugin, ctx = make(bp, payload=series_with(999.0), config={"budget": {"monthly_limit": 30.0}})
    plugin._paused = True
    await plugin._run_once(force=True)
    assert ctx.frequency.calls == [], f"暂停时仍在下发：{ctx.frequency.calls}"


@async_test
async def test_disabled_plugin_hook_does_nothing(bp):
    plugin, ctx = make(bp, payload=series_with(999.0), config={"plugin": {"enabled": False}})
    plugin._applied_adjust = 0.2
    await plugin.handle_new_session(message={"session_id": "new"})
    assert ctx.frequency.calls == [], "插件被禁用时 hook 仍在下发"


@async_test
async def test_statistics_failure_keeps_previous_spend(bp):
    """取数抛错时应沿用上次值，而不是当成零花费（否则会静默解除压制）。"""
    plugin, ctx = make(bp, payload=series_with(150.0), config={"budget": {"monthly_limit": 30.0}})
    await plugin._run_once(force=True)
    assert plugin._applied_adjust == pytest.approx(0.1)

    ctx.statistics.local.raise_error = RuntimeError("rpc timeout")
    await plugin._run_once(force=True)
    assert plugin._applied_adjust == pytest.approx(0.1), (
        f"取数失败后被当成零花费，压制被解除：{plugin._applied_adjust}"
    )


# ================= G. 生命周期 =================


@async_test
async def test_unload_is_idempotent(bp):
    plugin, _ctx = make(bp, payload=series_with(0.0))
    await plugin.on_load()
    await plugin.on_unload()
    await plugin.on_unload()
    assert plugin._loop_task is None


@async_test
async def test_config_update_restarts_loop(bp):
    plugin, _ctx = make(bp, payload=series_with(0.0))
    await plugin.on_load()
    await plugin.on_config_update("self", {}, "2")
    assert plugin._loop_task is not None
    await plugin.on_unload()
