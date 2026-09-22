"""冒烟测试：用假上下文跑完整生命周期，验证控制链路连通。

不依赖真实宿主，覆盖：加载/卸载、单轮巡检、超支闸门、单群例外双闸门、
命令响应、跨月重置。直接运行：python tests/smoke_test.py
"""

import asyncio
import importlib.util
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.CRITICAL)
LOGGER = logging.getLogger("smoke")

FAILURES: list = []
_SMOKE_DATA_DIR: str | None = None


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


def load_plugin_module():
    """按路径加载插件，规避扁平 sys.path 下的同名模块冲突。"""
    spec = importlib.util.spec_from_file_location("budget_pacer_smoke", ROOT / "plugin.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------- 假宿主组件 ----------


class FakeFrequency:
    def __init__(self) -> None:
        self.calls: list = []

    async def set_adjust(self, chat_id: str, value: float) -> bool:
        self.calls.append((chat_id, value))
        return True

    async def get_adjust(self, chat_id: str) -> float:
        for cid, value in reversed(self.calls):
            if cid == chat_id:
                return value
        return 1.0

    async def get_current_talk_value(self, chat_id: str) -> float:
        return 0.5

    def latest_for(self, chat_id: str):
        for cid, value in reversed(self.calls):
            if cid == chat_id:
                return value
        return None


class FakeLocalStatistics:
    def __init__(self, series: dict) -> None:
        self.series = series
        self.calls: list = []

    async def model_trend(self, **kwargs):
        self.calls.append(kwargs)
        return self.series


class FakeStatistics:
    def __init__(self, series: dict) -> None:
        self.local = FakeLocalStatistics(series)


class FakeChat:
    def __init__(self, streams: list) -> None:
        self.streams = streams

    async def get_all_streams(self, platform: str = "qq"):
        return self.streams


class FakeDB:
    def __init__(self, rows: list | None = None) -> None:
        self.rows = rows or []
        self.queries: list = []

    async def query(self, **kwargs):
        self.queries.append(kwargs)
        return self.rows


class FakeSend:
    def __init__(self) -> None:
        self.sent: list = []

    async def text(self, content: str, stream_id: str) -> bool:
        self.sent.append((stream_id, content))
        return True


class FakePaths:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir


def _default_data_dir() -> str:
    """冒烟测试默认的数据目录。

    必须落在临时目录：插件在没有 ctx.paths 时会降级写到插件目录，
    而带着测试写入的倍率部署会让真机一启动就压制发言。
    """
    global _SMOKE_DATA_DIR
    if _SMOKE_DATA_DIR is None:
        _SMOKE_DATA_DIR = tempfile.mkdtemp(prefix="budget-pacer-smoke-")
    return _SMOKE_DATA_DIR


class FakeContext:
    def __init__(self, series: dict, streams: list, db_rows: list | None = None, data_dir: str | None = None) -> None:
        self.logger = LOGGER
        self.statistics = FakeStatistics(series)
        self.frequency = FakeFrequency()
        self.chat = FakeChat(streams)
        self.db = FakeDB(db_rows)
        self.send = FakeSend()
        self.paths = FakePaths(data_dir if data_dir is not None else _default_data_dir())


# ---------- 构造数据 ----------


def month_label(now: datetime) -> str:
    """本月 1 日 0 点的桶标签——必定落在本月的统计窗口内。"""
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start.strftime("%Y-%m-%d %H:%M:%S")


def build_series(now: datetime, cost: float) -> dict:
    """构造 model_trend 已解包后的 series 结构（含上月噪声以验证裁剪）。"""
    first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    prev_month = (first_of_month - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamps": [prev_month, month_label(now)],
        "values_by_key": {"test-model": [999.0, cost]},
        "labels_by_key": {"test-model": "test-model"},
    }


def build_streams() -> list:
    return [
        {"session_id": "group-A", "stream_id": "group-A", "group_id": "10001", "is_group_session": True},
        {"session_id": "group-B", "stream_id": "group-B", "group_id": "10002", "is_group_session": True},
        {"session_id": "private-C", "stream_id": "private-C", "user_id": "20001", "is_group_session": False},
    ]


def new_plugin(
    bp,
    series: dict,
    streams: list,
    config: dict | None = None,
    db_rows: list | None = None,
    data_dir: str | None = None,
):
    plugin = bp.BudgetPacerPlugin()
    ctx = FakeContext(series, streams, db_rows, data_dir)
    plugin._set_context(ctx)

    # SDK 要求非空配置必须携带 plugin.config_version，真机上由 Runner 生成完整配置
    payload = {key: value for key, value in (config or {}).items()}
    plugin_section = dict(payload.get("plugin") or {})
    plugin_section.setdefault("config_version", "1")
    payload["plugin"] = plugin_section

    plugin.set_plugin_config(payload)
    return plugin, ctx


# ---------- 场景 ----------


async def scenario_load_unload(bp) -> None:
    print("\n[1] 生命周期：加载 -> 卸载")
    now = datetime.now()
    # 用临时目录承接状态文件，避免测试把 budget_state.json 留在插件目录里
    # （带着测试写入的倍率部署会导致真机一启动就压制发言）
    with tempfile.TemporaryDirectory() as tmp:
        plugin, ctx = new_plugin(bp, build_series(now, 0.0), build_streams(), data_dir=tmp)

        await plugin.on_load()
        check(plugin._loop_task is not None, "on_load 启动了巡检任务")
        check(plugin._decision.reason == "disabled", "加载后尚未判定，reason 为初始值")

        await plugin.on_unload()
        check(plugin._loop_task is None, "on_unload 停止了巡检任务")
        check((Path(tmp) / bp.STATE_FILE_NAME).is_file(), "状态文件已落盘到数据目录")

        await plugin.on_unload()  # 幂等，重复卸载不应抛错
        check(True, "重复卸载不抛错")

    check(not (ROOT / bp.STATE_FILE_NAME).exists(), "插件目录未残留状态文件")


async def scenario_on_track(bp) -> None:
    print("\n[2] 零花费：不应触发任何压制")
    now = datetime.now()
    plugin, ctx = new_plugin(bp, build_series(now, 0.0), build_streams())

    await plugin._run_once(force=True)

    check(plugin._decision.reason in ("coast", "on_track"), f"未超支（reason={plugin._decision.reason}）")
    check(plugin._applied_adjust == 1.0, f"倍率保持 1.0（实际 {plugin._applied_adjust}）")
    check(len(ctx.frequency.calls) == 3, f"覆盖全部 3 个会话（实际 {len(ctx.frequency.calls)}）")
    check(ctx.statistics.local.calls[0]["days"] == 32, "取数窗口用 32 天")
    check(ctx.statistics.local.calls[0]["top_models"] == 50, "top_models 取满 50 防截断")
    check(ctx.statistics.local.calls[0]["metric"] == "cost", "按 cost 指标取数")


async def scenario_over_budget_gate(bp) -> None:
    print("\n[3] 花费远超预算：硬闸门必须生效")
    now = datetime.now()
    plugin, ctx = new_plugin(
        bp,
        build_series(now, 150.0),  # 预算 30，花 150
        build_streams(),
    )

    await plugin._run_once(force=True)

    check(plugin._decision.reason == "over_budget", "判定为已超预算")
    check(plugin._applied_adjust == 0.1, f"闸门压到 0.1（实际 {plugin._applied_adjust}）")
    values = {value for _cid, value in ctx.frequency.calls}
    check(values == {0.1}, f"所有会话都拿到闸门值（实际 {values}）")


async def scenario_month_trim(bp) -> None:
    print("\n[4] 跨月裁剪：上月数据不能计入本月")
    now = datetime.now()
    # 上月噪声 999 元，本月仅 3 元：若裁剪失效，花费会被判成 1002
    plugin, ctx = new_plugin(bp, build_series(now, 3.0), build_streams())

    spend = await plugin._month_spend()
    check(spend == 3.0, f"只统计本月花费（实际 {spend}）")


async def scenario_override_dual_gate(bp) -> None:
    print("\n[5] 单群例外：双闸门取更严者")
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    db_rows = [
        {"id": 2, "session_id": "group-B", "timestamp": stamp, "cost": 9.0},
        # 上月的记录不应计入
        {"id": 1, "session_id": "group-B", "timestamp": "2000-01-01 00:00:00", "cost": 500.0},
    ]
    config = {
        "budget": {"monthly_limit": 30.0},
        "overrides": [{"group_id": "10002", "monthly_limit": 5.0}],
    }
    plugin, ctx = new_plugin(bp, build_series(now, 0.0), build_streams(), config, db_rows)

    await plugin._run_once(force=True)

    global_value = ctx.frequency.latest_for("group-A")
    override_value = ctx.frequency.latest_for("group-B")
    check(global_value == 1.0, f"全局未超支，保持 1.0（实际 {global_value}）")
    check(override_value is not None and override_value < global_value,
          f"例外群被额外压制（{override_value} < {global_value}）")
    check(override_value == 0.1, f"例外群按其自身预算走闸门（实际 {override_value}）")

    session_spend = await plugin._session_spend("group-B", now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
    check(session_spend == 9.0, f"会话花费按时间裁剪（实际 {session_spend}）")


async def scenario_missing_session(bp) -> None:
    print("\n[6] 宿主 set_adjust 失败：不应中断整轮")
    now = datetime.now()

    class FlakyFrequency(FakeFrequency):
        async def set_adjust(self, chat_id: str, value: float) -> bool:
            if chat_id == "group-B":
                raise RuntimeError("session not found")
            return await super().set_adjust(chat_id, value)

    plugin, ctx = new_plugin(bp, build_series(now, 150.0), build_streams())
    ctx.frequency = FlakyFrequency()

    await plugin._run_once(force=True)

    check(plugin._applied_adjust == 0.1, "单会话失败不影响本轮判定")
    check(len(ctx.frequency.calls) == 2, f"其余会话仍被下发（实际 {len(ctx.frequency.calls)}）")


async def scenario_command(bp) -> None:
    print("\n[7] 命令：查询 / 设预算 / 暂停 / 恢复 / 重置")
    now = datetime.now()
    plugin, ctx = new_plugin(bp, build_series(now, 0.0), build_streams())

    await plugin._run_once(force=True)
    text = plugin._status_text()
    check("月度预算" in text and "节奏比" in text, "状态文本含关键指标")

    reply = await plugin._handle_command("设置 12.5")
    check("12.5" in reply, f"可临时设置预算（{reply.splitlines()[0]}）")
    check(plugin._effective_budget() == 12.5, "临时预算已生效")

    reply = await plugin._handle_command("暂停")
    check(plugin._paused, "暂停生效")
    check(ctx.frequency.latest_for("group-A") == 1.0, "暂停时倍率复位为 1.0")

    reply = await plugin._handle_command("恢复")
    check(not plugin._paused, "恢复生效")

    reply = await plugin._handle_command("重置")
    check(plugin._temporary_budget == 0.0, "重置清除了临时预算")

    reply = await plugin._handle_command("abc")
    check("无法识别" in reply, "非法输入给出提示而不抛错")


async def scenario_admin_guard(bp) -> None:
    print("\n[8] 权限：未配置 admin_ids 时放行，配置后拒绝非管理员")
    now = datetime.now()

    plugin, _ctx = new_plugin(bp, build_series(now, 0.0), build_streams())
    check(plugin._is_admin({"user_id": "999"}) is True, "未配置管理员时放行")

    plugin, _ctx = new_plugin(
        bp,
        build_series(now, 0.0),
        build_streams(),
        {"permission": {"admin_ids": ["123456789", "qq:987654321"]}},
    )
    check(plugin._is_admin({"user_id": "123456789"}) is True, "裸号匹配裸号")
    check(plugin._is_admin({"user_id": "qq:123456789"}) is True, "user_id 带前缀也能匹配")
    check(plugin._is_admin({"user_id": "987654321"}) is True, "配置带前缀、user_id 裸号可匹配")
    check(plugin._is_admin({"user_id": "999"}) is False, "非管理员拒绝")


async def scenario_month_reset(bp) -> None:
    print("\n[9] 跨月重置")
    plugin, _ctx = new_plugin(bp, build_series(datetime.now(), 0.0), build_streams())
    plugin._state_month = "2000-01"
    plugin._applied_adjust = 0.3
    plugin._temporary_budget = 99.0
    plugin._known_sessions.add("stale-session")

    plugin._reset_if_new_month(datetime.now())

    check(plugin._applied_adjust == 1.0, "倍率复位")
    check(plugin._temporary_budget == 0.0, "临时预算清空")
    check(plugin._known_sessions == set(), "会话缓存清空")
    check(plugin._state_month == datetime.now().strftime("%Y-%m"), "月份标记更新")


async def scenario_new_session_hook(bp) -> None:
    print("\n[10] 新会话 hook：观察模式下补设倍率（ON_MESSAGE 在 1.2.5 不派发）")
    now = datetime.now()
    plugin, ctx = new_plugin(bp, build_series(now, 150.0), build_streams())
    await plugin._run_once(force=True)  # 让倍率落到闸门值 0.1

    before = len(ctx.frequency.calls)
    await plugin.handle_new_session(message={"session_id": "fresh-group"})
    check(ctx.frequency.latest_for("fresh-group") == 0.1, "新会话被补设为当前倍率")
    check(len(ctx.frequency.calls) == before + 1, "恰好新增一次下发")

    await plugin.handle_new_session(message={"session_id": "fresh-group"})
    check(len(ctx.frequency.calls) == before + 1, "同一会话再次来消息不再下发（命中缓存）")

    await plugin.handle_new_session(message={})
    check(len(ctx.frequency.calls) == before + 1, "缺 session_id 时安全跳过")

    await plugin.handle_new_session()
    check(len(ctx.frequency.calls) == before + 1, "无载荷时不抛错")

    await plugin.handle_new_session(message={"session_id": "flat-session"})
    check(ctx.frequency.latest_for("flat-session") == 0.1, "另一新会话也被补设")


async def scenario_state_dir(bp) -> None:
    print("\n[11] 状态落盘：优先 ctx.paths.data_dir，坏文件安全降级")
    now = datetime.now()
    with tempfile.TemporaryDirectory() as tmp:
        plugin, _ctx = new_plugin(bp, build_series(now, 0.0), build_streams(), data_dir=tmp)
        plugin._applied_adjust = 0.42
        plugin._save_state()

        state_file = Path(tmp) / bp.STATE_FILE_NAME
        check(state_file.is_file(), "状态写入 data_dir")
        check(not list(Path(tmp).glob("*.tmp")), "临时文件已被原子替换，无残留")

        reloaded, _ctx = new_plugin(bp, build_series(now, 0.0), build_streams(), data_dir=tmp)
        reloaded._load_state()
        check(reloaded._applied_adjust == 0.42, "重启后恢复倍率")

        state_file.write_text("{ 这不是合法 JSON", encoding="utf-8")
        broken, _ctx = new_plugin(bp, build_series(now, 0.0), build_streams(), data_dir=tmp)
        broken._load_state()
        check(broken._applied_adjust == 1.0, "坏 JSON 按空状态继续，不抛错")

        state_file.write_text('{"month": "2000-01", "applied_adjust": 0.05}', encoding="utf-8")
        stale, _ctx = new_plugin(bp, build_series(now, 0.0), build_streams(), data_dir=tmp)
        stale._load_state()
        check(stale._applied_adjust == 1.0, "上月状态被丢弃并复位")


async def scenario_self_check(bp) -> None:
    print("\n[12] 插件内置自检")
    plugin, _ctx = new_plugin(bp, build_series(datetime.now(), 0.0), build_streams())
    plugin._self_check()
    check(True, "自检未抛错")


async def main() -> int:
    print("=" * 60)
    print("budget-pacer 冒烟测试")
    print("=" * 60)
    bp = load_plugin_module()

    for scenario in (
        scenario_load_unload,
        scenario_on_track,
        scenario_over_budget_gate,
        scenario_month_trim,
        scenario_override_dual_gate,
        scenario_missing_session,
        scenario_command,
        scenario_admin_guard,
        scenario_month_reset,
        scenario_new_session_hook,
        scenario_state_dir,
        scenario_self_check,
    ):
        try:
            await scenario(bp)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {scenario.__name__} 抛出异常：{exc!r}")
            FAILURES.append(scenario.__name__)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"冒烟失败：{len(FAILURES)} 项")
        for item in FAILURES:
            print(f"  - {item}")
    else:
        print("冒烟通过：全部场景 OK")

    if _SMOKE_DATA_DIR is not None:
        shutil.rmtree(_SMOKE_DATA_DIR, ignore_errors=True)

    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
