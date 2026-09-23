"""月度预算驱动的发言频率调节插件

工作方式:
1. 每 N 秒巡检一次本机月累计花费:
   - statistics.local.model_trend(days=32, bucket="hour", top_models=50, metric="cost")
     取最近 32 天的小时桶, 按「本月 1 日 0 点」过滤后求和, 得到自然月累计花费;
   - days 上限 365、top_models 上限 50(Host 侧硬限制), 均取满以避免截断少算。
2. 用「花费进度 ÷ 时间进度」得到节奏比 pacing, 做比例控制:
   - pacing > 1 花费超前 -> 压低倍率; pacing < 1 花费滞后 -> 放松(可选加频);
   - 死区 + 指数平滑避免抖动; 花费达到预算后走硬闸门。
3. 把倍率通过 frequency.set_adjust 下发给所有活跃会话:
   - 宿主侧倍率是内存态, 重启即失效, 因此启动时从 budget_state.json 恢复并重新下发;
   - 新建会话不继承倍率, 由 on_message 事件兜底补设。

统计口径: 月窗口按本地日期截断, 是「本月 1 日 0 点至当前」的准确口径。
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import field_validator

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import HookMode, ToolParameterInfo, ToolParamType

# ---------- 纯函数区（不依赖 maibot_sdk 的运行时, 可独立测试） ----------

STATE_FILE_NAME = "budget_state.json"
_SECONDS_PER_DAY = 86400.0

# 时间标签解析格式, 覆盖 Host 可能返回的多种形态
_FULL_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
)
# 不含年份的标签, 解析前需补当前年（直接 strptime 会触发 3.13 的弃用警告）
_PARTIAL_FORMATS = (
    "%m-%d %H:%M:%S",
    "%m-%d %H:%M",
    "%m-%d",
    "%m/%d %H:%M:%S",
    "%m/%d %H:%M",
    "%m/%d",
)
# 例外额度配置里可能出现的分隔符（用户习惯差异很大，统一归一化）
_OVERRIDE_SEPARATORS = ("，", ",", "；", ";", " ", "\t", "\n", "|")


def finite_or(value: Any, fallback: float) -> float:
    """把非有限值（NaN / ±Inf）或无法转换的值替换为兜底值。

    配置文件里可以写 `nan` / `inf`，统计链路也可能因 0/0 产出 NaN；
    这类值一旦透传到 frequency.set_adjust，宿主侧的行为是未定义的。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def clamp(value: float, low: float, high: float) -> float:
    """把数值夹到 [low, high] 区间；NaN 视为越界并夹到下限。"""
    low = finite_or(low, 0.0)
    high = finite_or(high, 1.0)
    value = finite_or(value, low)
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def month_bounds(now: datetime) -> tuple[datetime, datetime]:
    """返回当前自然月的 [起, 止) 边界（本地时间）。"""
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def month_key(now: datetime) -> str:
    """当前月的字符串标记, 用于跨月重置判定。"""
    return now.strftime("%Y-%m")


def month_progress(now: datetime, p_floor_days: float) -> tuple[float, float, float]:
    """计算月度时间进度。

    返回 (采用进度, 原始进度, 下限)。原始进度在月初极小, 会把节奏比放大成噪声,
    因此用 p_floor_days(默认 1 天)折算出的下限兜底。
    """
    start, end = month_bounds(now)
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return 1.0, 1.0, 0.0

    elapsed = (now - start).total_seconds()
    raw = clamp(elapsed / total_seconds, 0.0, 1.0)
    floor_seconds = max(0.0, float(p_floor_days)) * _SECONDS_PER_DAY
    floor = clamp(floor_seconds / total_seconds, 0.0, 1.0)
    return max(raw, floor), raw, floor


def _parse_label(label: str) -> Optional[datetime]:
    """把统计桶的时间标签解析成本地 datetime, 失败返回 None。

    支持 epoch 秒/毫秒、ISO 与常见日期格式; 无年份的标签补当前年后解析。
    """
    text = (label or "").strip()
    if not text:
        return None

    # 纯数字: epoch 秒或毫秒
    if text.replace(".", "", 1).isdigit():
        try:
            ts = float(text)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts)
        except (ValueError, OSError, OverflowError):
            return None

    for fmt in _FULL_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    year = datetime.now().year
    for fmt in _PARTIAL_FORMATS:
        try:
            return datetime.strptime(f"{year}-{text}", f"%Y-{fmt}")
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _label_in_window(label: str, start: datetime, end: datetime) -> bool:
    """判断时间标签是否落在 [start, end) 窗口内。"""
    parsed = _parse_label(label)
    if parsed is None:
        return False
    return start <= parsed < end


def window_bucket(timestamps: Any, values: Any, start: datetime, end: datetime) -> float:
    """把 (时间标签序列, 数值序列) 中落在窗口内的数值求和。

    无法解析的桶不计入——宁可少报也不把上月数据算进本月。
    """
    if not isinstance(values, (list, tuple)):
        return 0.0
    labels = timestamps if isinstance(timestamps, (list, tuple)) else []
    total = 0.0
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        label = str(labels[index]) if index < len(labels) else ""
        if _label_in_window(label, start, end):
            total += number
    return total


def unwrap_series(raw: Any) -> dict:
    """兼容两种返回形态: 已解包的 series, 或外层仍包着 series 键。"""
    if not isinstance(raw, dict):
        return {}
    inner = raw.get("series")
    if isinstance(inner, dict):
        return inner
    return raw


def series_total(raw: Any, start: datetime, end: datetime) -> float:
    """从 model_trend 返回结构里提取窗口内总和。"""
    series = unwrap_series(raw)
    if not series:
        return 0.0

    timestamps = series.get("timestamps") or series.get("time_labels") or series.get("labels")
    by_key = series.get("values_by_key") or series.get("data_by_key")
    if isinstance(by_key, dict):
        return sum(window_bucket(timestamps, values, start, end) for values in by_key.values())

    values = series.get("values") or series.get("data")
    if values is not None:
        return window_bucket(timestamps, values, start, end)
    return 0.0


def series_by_model(raw: Any, start: datetime, end: datetime) -> dict:
    """从 model_trend 提取 {模型名: 窗口内总和}。"""
    series = unwrap_series(raw)
    if not series:
        return {}

    timestamps = series.get("timestamps") or series.get("time_labels") or series.get("labels")
    by_key = series.get("values_by_key") or series.get("data_by_key")
    if not isinstance(by_key, dict):
        return {}

    labels_by_key = series.get("labels_by_key") or series.get("label_by_key") or {}
    out: dict = {}
    for key, values in by_key.items():
        label = labels_by_key.get(key) if isinstance(labels_by_key, dict) and labels_by_key.get(key) else key
        out[str(label)] = window_bucket(timestamps, values, start, end)
    return out


def coerce_datetime(value: Any) -> Optional[datetime]:
    """把 datetime / 字符串时间统一成本地 naive datetime。"""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        return _parse_label(value)
    return None


@dataclass
class PacingDecision:
    """一次节奏判定的完整结果, 用于日志与命令展示。"""

    target: float = 1.0
    spend: float = 0.0
    budget: float = 0.0
    progress: float = 0.0
    raw_progress: float = 0.0
    progress_floor: float = 0.0
    spend_ratio: float = 0.0
    pacing: float = 0.0
    reason: str = "disabled"


def compute_adjust(
    spend: float,
    budget: float,
    now: datetime,
    *,
    p_floor_days: float = 1.0,
    deadband: float = 0.10,
    min_adjust: float = 0.10,
    max_adjust: float = 1.00,
    allow_boost: bool = False,
    over_budget_adjust: float = 0.10,
) -> PacingDecision:
    """按「花费进度 ÷ 时间进度」计算目标倍率。

    这是闭环的比例控制: 只依赖当前花费与预算, 不需要知道「降频能省多少」。
    超前消费就压低倍率, 花费随之下降, 节奏比会自行回落, 因而天然收敛。
    """
    decision = PacingDecision()
    # 先净化输入：配置可写 nan / inf，统计链路也可能因 0/0 产出 NaN
    spend = max(0.0, finite_or(spend, 0.0))
    budget = finite_or(budget, 0.0)
    p_floor_days = max(0.0, finite_or(p_floor_days, 1.0))
    deadband = max(0.0, finite_or(deadband, 0.10))
    min_adjust = finite_or(min_adjust, 0.10)
    max_adjust = finite_or(max_adjust, 1.00)
    over_budget_adjust = finite_or(over_budget_adjust, 0.10)

    decision.spend = spend
    decision.budget = budget

    if budget <= 0:
        decision.reason = "budget_disabled"
        decision.target = 1.0
        return decision

    progress, raw_progress, floor = month_progress(now, p_floor_days)
    decision.progress = progress
    decision.raw_progress = raw_progress
    decision.progress_floor = floor

    spend_ratio = spend / budget
    decision.spend_ratio = spend_ratio
    pacing = spend_ratio / progress if progress > 0 else 0.0
    decision.pacing = pacing

    if spend_ratio >= 1.0:
        # 硬闸门: 已达/超出预算, 不问进度直接压到安全值
        decision.reason = "over_budget"
        target = min(1.0, over_budget_adjust)
    elif abs(pacing - 1.0) <= deadband:
        decision.reason = "on_track"
        target = 1.0
    elif pacing > 1.0:
        decision.reason = "brake"
        target = 1.0 / pacing
    elif allow_boost:
        decision.reason = "boost"
        target = 1.0 / pacing
    else:
        decision.reason = "coast"
        target = 1.0

    target = clamp(target, min_adjust, max_adjust)
    if decision.reason == "over_budget":
        target = min(target, over_budget_adjust)
    decision.target = target
    return decision


def smooth_adjust(previous: float, target: float, alpha: float) -> float:
    """指数平滑, 避免倍率在两次巡检之间跳变。

    previous 异常（NaN 或类型错误）时退回 1.0，即「不做调节」——
    这比把 NaN 继续传下去安全得多。
    """
    previous = finite_or(previous, 1.0)
    target = finite_or(target, 1.0)
    alpha = clamp(finite_or(alpha, 0.35), 0.0, 1.0)
    return previous * (1.0 - alpha) + target * alpha


def coerce_id_list(value: Any) -> Any:
    """宽容解析 ID 列表字段。

    用户在 config.toml 里常把列表写成单个字符串、逗号/分号分隔字符串甚至整数,
    这里统一归一化为字符串列表, 避免 pydantic 直接拒载整份配置。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if not text:
            return []
        for sep in ("，", ",", "；", ";", " ", "\t"):
            text = text.replace(sep, ";")
        return [part.strip() for part in text.split(";") if part.strip()]
    return value


def format_status(decision: PacingDecision, adjust: float, override_note: str = "") -> str:
    """把判定结果格式化成一行可读文本。"""
    reason_text = {
        "on_track": "节奏正常",
        "brake": "花费超前，正在压低",
        "coast": "花费滞后（未启用加频）",
        "boost": "花费滞后，正在放松",
        "over_budget": "已达预算，进入闸门",
        "budget_disabled": "未设置预算",
        "disabled": "调节已暂停",
    }.get(decision.reason, decision.reason)

    lines = [
        f"月度预算：{decision.budget:.4f} 元",
        f"本月已花：{decision.spend:.4f} 元（{decision.spend_ratio * 100:.1f}%）",
        f"时间进度：{decision.progress * 100:.1f}%",
        f"节奏比：{decision.pacing:.2f}（{reason_text}）",
        f"当前倍率：{adjust:.3f}",
    ]
    if override_note:
        lines.append(override_note)
    return "\n".join(lines)


# ---------- 配置模型 ----------


class PluginSection(PluginConfigBase):
    __ui_label__ = "插件设置"

    config_version: str = Field(default="1", description="配置版本号（热更新迁移用，勿手动修改）")
    enabled: bool = Field(default=True, description="是否启用插件")


class BudgetSection(PluginConfigBase):
    __ui_label__ = "预算与调节"

    monthly_limit: float = Field(default=30.0, description="月度预算，单位元；<=0 表示不调节")
    check_interval_seconds: int = Field(default=300, description="巡检间隔（秒），建议不小于 300")
    apply_private: bool = Field(default=True, description="是否同时调节私聊会话")
    p_floor_days: float = Field(default=1.0, description="月初进度下限（折算为天数），避免月初噪声放大")
    deadband: float = Field(default=0.10, description="死区带宽：节奏比在此范围内不调整，防抖动")
    smoothing: float = Field(default=0.35, description="指数平滑系数（0~1），越大响应越快")
    min_adjust: float = Field(default=0.10, description="最低倍率；设 0 表示允许完全静音")
    max_adjust: float = Field(default=1.00, description="最高倍率；大于 1 才可能加频")
    allow_boost: bool = Field(default=False, description="花费滞后时是否提高倍率以贴近预算")
    over_budget_adjust: float = Field(default=0.10, description="达到预算后的闸门倍率")
    override_scan_limit: int = Field(default=2000, description="例外会话花费明细的扫描条数上限")


class OverrideSection(PluginConfigBase):
    __ui_label__ = "例外额度"

    group_id: str = Field(default="", description="群号")
    monthly_limit: float = Field(default=0.0, description="该群的独立月度预算，单位元；<=0 表示不启用")
    enabled: bool = Field(default=True, description="是否启用该例外")

    @field_validator("group_id", mode="before")
    @classmethod
    def _normalize_group_id(cls, value: Any) -> Any:
        if value is None:
            return ""
        return str(value).strip()


class PermissionSection(PluginConfigBase):
    __ui_label__ = "权限"

    admin_ids: list = Field(default_factory=list, description="管理员 QQ 号列表；留空则所有人可操作")

    @field_validator("admin_ids", mode="before")
    @classmethod
    def _normalize_admins(cls, value: Any) -> Any:
        return coerce_id_list(value)


class BudgetPacerConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    budget: BudgetSection = Field(default_factory=BudgetSection)
    overrides: list[OverrideSection] = Field(default_factory=list, description="单群例外额度列表")
    permission: PermissionSection = Field(default_factory=PermissionSection)

    @staticmethod
    def _parse_override_item(item: Any) -> Any:
        """把例外项规范成 {group_id, monthly_limit}；无法识别时返回 None。

        这里主动丢弃脏数据，而不是留给 pydantic 抛错：一项手误不该让
        整份配置校验失败、插件静默退回默认值（用户完全看不出原因）。
        """
        if isinstance(item, dict):
            if item.get("enabled", True) is False:
                return None
            group_id = str(item.get("group_id") or "").strip()
            raw_limit = item.get("monthly_limit")
        else:
            text = str(item or "").strip()
            if ":" not in text:
                return None
            group_id, _, raw_limit = text.partition(":")
            group_id = group_id.strip()

        if not group_id:
            return None
        try:
            limit = float(raw_limit)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(limit) or limit <= 0:
            return None
        return {"group_id": group_id, "monthly_limit": limit}

    @field_validator("overrides", mode="before")
    @classmethod
    def _normalize_overrides(cls, value: Any) -> Any:
        """允许把例外写成宽松形式，例如 "123456:5.0，789012:3.0"。

        用户常把列表写成单个字符串或逗号分隔串，这里统一归一化，
        避免 pydantic 直接拒载整份配置。
        """
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]

        if isinstance(value, str):
            text = value
            for sep in _OVERRIDE_SEPARATORS:
                text = text.replace(sep, "|")
            chunks: list = [chunk.strip() for chunk in text.split("|") if chunk.strip()]
        elif isinstance(value, (list, tuple, set)):
            chunks = list(value)
        else:
            return value

        out: list = []
        for chunk in chunks:
            parsed = cls._parse_override_item(chunk)
            if parsed is not None:
                out.append(parsed)
        return out


# ---------- 插件主体 ----------


class BudgetPacerPlugin(MaiBotPlugin):
    """按月度预算动态调节发言频率。"""

    config_model = BudgetPacerConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 基类 __init__ 不接受参数，这里显式丢弃以免 Runner 传参时崩溃
        super().__init__()
        self._loop_task: Optional[asyncio.Task] = None
        self._applied_adjust: float = 1.0
        self._session_adjust: dict = {}  # 每会话最后成功下发的倍率，用于逐会话跳过无变更下发
        self._known_sessions: set = set()
        self._decision: PacingDecision = PacingDecision()
        self._temporary_budget: float = 0.0
        self._paused: bool = False
        self._state_month: str = ""
        self._last_reason: str = ""
        self._was_enabled: bool = False

    # ---- 生命周期 ----

    async def on_load(self) -> None:
        self._self_check()
        self._load_state()
        if self.config.plugin.enabled:
            self._start_loop()
        self._was_enabled = bool(self.config.plugin.enabled)
        self.ctx.logger.info(
            "月度预算插件已加载：预算=%.4f 元，巡检=%d 秒，加频=%s",
            self._effective_budget(),
            self.config.budget.check_interval_seconds,
            "开" if self.config.budget.allow_boost else "关",
        )

    async def on_unload(self) -> None:
        await self._stop_loop()
        # 宿主倍率是内存态：插件卸载后没人再管它，不复位就会一直压着发言频率
        await self._reset_adjust_to_default("插件卸载")
        self._save_state()
        self.ctx.logger.info("月度预算插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        if scope != "self":
            return
        self.ctx.logger.info("配置已更新：version=%s", version)
        # self.config 在钩子触发时可能已被宿主换成新配置，
        # 不能用它判断「从开到关」，用自己记录的上次状态
        was_enabled = self._was_enabled
        await self._stop_loop()
        now_enabled = bool(self.config.plugin.enabled)
        self._was_enabled = now_enabled
        if now_enabled:
            self._start_loop()
        elif was_enabled:
            # 从开到关：立即复位宿主倍率，否则禁用后频率仍停留在上次压制的值
            await self._reset_adjust_to_default("插件禁用")

    async def _reset_adjust_to_default(self, reason: str) -> None:
        """把宿主倍率复位为 1.0 并清空本地跟踪状态。失败只记日志不抛出。"""
        try:
            await self._apply_adjust(1.0)
        except Exception as exc:
            self.ctx.logger.warning("复位宿主倍率失败（%s）：%s", reason, exc)
            return
        self._applied_adjust = 1.0
        self._session_adjust.clear()
        self.ctx.logger.info("%s：宿主倍率已复位为 1.0", reason)

    # ---- 启动时自检纯函数 ----

    def _self_check(self) -> None:
        try:
            base = datetime(2026, 9, 15, 12, 0, 0)
            start, end = month_bounds(base)
            assert start == datetime(2026, 9, 1) and end == datetime(2026, 10, 1), "month_bounds 错误"

            ts = ["2026-08-31 23:00:00", "2026-09-01 00:00:00", "2026-09-15 10:00:00"]
            vals = [9.9, 1.0, 2.0]
            assert window_bucket(ts, vals, start, end) == 3.0, "window_bucket 过滤错误"

            on_track = compute_adjust(15.0, 30.0, base)
            assert on_track.reason == "on_track", "节奏正常判定错误"

            braking = compute_adjust(30.0, 30.0, base)
            assert braking.reason == "over_budget", "超支闸门判定错误"

            assert smooth_adjust(1.0, 0.0, 0.5) == 0.5, "smooth_adjust 错误"
            self.ctx.logger.info("[自检] 预算节奏纯函数 OK")
        except AssertionError as exc:
            self.ctx.logger.error("[自检] 行为异常：%s", exc)

    # ---- 事件 ----

    @HookHandler(
        "chat.receive.after_process",
        name="handle_new_session",
        description="为首次出现的会话补设发言频率倍率",
        mode=HookMode.OBSERVE,
    )
    async def handle_new_session(self, message=None, **kwargs) -> None:
        """新会话兜底：补设频率倍率。

        不要改用 EventHandler(ON_MESSAGE)：该事件的派发在 MaiBot 1.2.5 中是被注释掉的
        （src/chat/message_receive/bot.py:846），注册了也永远不会触发。
        这里用 observe 模式的入站 hook，旁路观察且不阻塞消息主链。
        """
        if self._paused or not self.config.plugin.enabled:
            return

        session_id = ""
        if isinstance(message, dict):
            session_id = str(message.get("session_id") or "").strip()
        if not session_id:
            session_id = self._stream_id(kwargs)
        if not session_id or session_id in self._known_sessions:
            return

        if len(self._known_sessions) > 2000:
            self._known_sessions.clear()
        self._known_sessions.add(session_id)

        if abs(self._applied_adjust - 1.0) < 1e-9:
            return
        try:
            await self.ctx.frequency.set_adjust(session_id, self._applied_adjust)
            self._session_adjust[session_id] = self._applied_adjust
            self.ctx.logger.info("新会话 %s 已补设倍率 %.3f", session_id, self._applied_adjust)
        except Exception as exc:
            self.ctx.logger.debug("新会话补设倍率失败(%s)：%s", session_id, exc)

    # ---- 命令 ----

    @Command(
        "budget",
        description="查看或调整月度预算与发言频率调节",
        pattern=r"^\s*[/／]\s*(?:预算|budget)(?:\s+.*)?$",
    )
    async def cmd_budget(self, **kwargs) -> tuple:
        stream_id = self._stream_id(kwargs)
        if not self._is_admin(kwargs):
            # fail-close：未配置 admin_ids 或不在名单内一律拒绝；静默，只留日志
            self.ctx.logger.info("预算命令被拒绝：非管理员或未配置 admin_ids（fail-close）")
            return True, "", 0

        text = self._message_text(kwargs)
        argument = self._command_argument(text)
        reply = await self._handle_command(argument)
        if stream_id and reply:
            await self._send_text(reply, stream_id)
        return True, reply, 2 if stream_id else 0

    async def _handle_command(self, argument: str) -> str:
        argument = (argument or "").strip()
        parts = argument.split()
        action = parts[0].lower() if parts else ""

        if action in ("暂停", "pause"):
            self._paused = True
            await self._apply_adjust(1.0)
            self._applied_adjust = 1.0
            self._save_state()
            return "预算调节已暂停，倍率已复位为 1.0。"

        if action in ("恢复", "resume", "开启"):
            self._paused = False
            await self._run_once(force=True)
            return f"预算调节已恢复。\n{self._status_text()}"

        if action in ("重置", "reset"):
            self._temporary_budget = 0.0
            self._applied_adjust = 1.0
            self._known_sessions.clear()
            await self._apply_adjust(1.0)
            self._save_state()
            return "已重置：临时预算清除，倍率复位为 1.0。"

        # 「设置 30」或直接「30」
        value_text = ""
        if action in ("设置", "set") and len(parts) >= 2:
            value_text = parts[1]
        elif action:
            value_text = parts[0]

        if value_text:
            try:
                value = float(value_text)
            except ValueError:
                return f"无法识别预算数值：{value_text}\n用法：/预算 30 或 /预算 设置 30"
            if value < 0:
                return "预算不能为负数。"
            self._temporary_budget = value
            self._save_state()
            await self._run_once(force=True)
            return f"本月预算已临时设为 {value:.4f} 元。\n{self._status_text()}"

        return self._status_text()

    def _status_text(self) -> str:
        if self._paused:
            return "预算调节处于暂停状态。发送「/预算 恢复」继续。"
        return format_status(self._decision, self._applied_adjust, self._override_summary())

    def _override_summary(self) -> str:
        rows = self._active_overrides()
        if not rows:
            return ""
        return "例外额度：" + "，".join(f"群 {item['group_id']} → {item['monthly_limit']} 元" for item in rows)

    # ---- Tool: 让 LLM 能回答预算问题 ----

    @Tool(
        "budget_status",
        brief_description="查询本月预算的使用进度与当前发言频率倍率",
        detailed_description="返回月度预算、本月已花费、时间进度、节奏比与当前发言频率倍率。"
        "适用于「这个月花了多少」「预算还剩多少」「为什么最近话少」等场景。无需参数。",
        parameters=[
            ToolParameterInfo(
                name="stream_id",
                param_type=ToolParamType.STRING,
                description="当前聊天流 ID",
                required=False,
            ),
        ],
    )
    async def tool_budget_status(self, stream_id: str = "", **kwargs):
        await self._run_once(force=True)
        return {
            "success": True,
            "status": self._status_text(),
            "message": "已获取预算状态，请据此回答，不要杜撰数字。",
        }

    # ---- 控制循环 ----

    def _start_loop(self) -> None:
        if self._loop_task and not self._loop_task.done():
            return
        self._loop_task = asyncio.create_task(self._control_loop())

    async def _stop_loop(self) -> None:
        task = self._loop_task
        self._loop_task = None
        if not task or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _control_loop(self) -> None:
        # 首轮稍作延迟，避开宿主启动瞬间统计能力未就绪
        await asyncio.sleep(min(30, max(5, self.config.budget.check_interval_seconds // 10)))
        while True:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.ctx.logger.error("预算巡检失败：%s", exc, exc_info=True)
            await asyncio.sleep(max(30, self.config.budget.check_interval_seconds))

    async def _run_once(self, force: bool = False) -> None:
        """执行一轮判定并按需下发。

        force 仅表示「由命令/工具触发的立即判定」，**不影响下发策略**：
        下发始终以「倍率是否真的变化」为准。宿主侧 set_adjust 会连带唤醒 Planner
        （maisaka/runtime.py: adjust_talk_frequency → _schedule_message_turn），
        所以无变更的调用等于平白增加花费，与预算插件的目的完全相悖。
        """
        if self._paused:
            return

        now = datetime.now()
        self._reset_if_new_month(now)

        budget = self._effective_budget()
        spend = await self._month_spend()
        decision = compute_adjust(
            spend,
            budget,
            now,
            p_floor_days=self.config.budget.p_floor_days,
            deadband=self.config.budget.deadband,
            min_adjust=self.config.budget.min_adjust,
            max_adjust=self.config.budget.max_adjust,
            allow_boost=self.config.budget.allow_boost,
            over_budget_adjust=self.config.budget.over_budget_adjust,
        )
        self._decision = decision

        target = smooth_adjust(self._applied_adjust, decision.target, self.config.budget.smoothing)
        target = clamp(target, self.config.budget.min_adjust, self.config.budget.max_adjust)
        if decision.reason == "over_budget":
            target = min(target, self.config.budget.over_budget_adjust)

        changed = abs(target - self._applied_adjust) > 1e-6
        if changed:
            self._applied_adjust = target
            self._save_state()
            self.ctx.logger.info(
                "预算调节：花费=%.4f/%.4f 进度=%.1f%% 节奏比=%.2f 倍率=%.3f（%s）",
                decision.spend,
                decision.budget,
                decision.progress * 100,
                decision.pacing,
                target,
                decision.reason,
            )
        # 无论全局倍率是否变化都要走一遍下发：
        # 例外群走自己的闸门（min(全局, 自身)），即使全局不变，
        # 例外群自身花费变化也会改变它应得的倍率。
        # 跳过无变更下发在 _apply_adjust 内逐会话判断。
        if decision.reason == "budget_disabled":
            # 预算禁用 = 插件不干预：绝不调用宿主（调用即唤醒 Planner）。
            # 唯一例外：之前已下发过非 1.0 的倍率，需一次性复位，避免永远卡在低值。
            if abs(self._applied_adjust - 1.0) > 1e-6:
                self._applied_adjust = 1.0
                self._session_adjust.clear()
                await self._apply_adjust(1.0)
                self.ctx.logger.info("预算已禁用（额度为 0），倍率复位为 1.0")
        else:
            applied = await self._apply_adjust(target)
            if applied:
                self.ctx.logger.info("预算下发 %d 个会话，倍率 %.3f（%s）", applied, target, decision.reason)
        if decision.reason != self._last_reason:
            self._last_reason = decision.reason

    # ---- 数据采集 ----

    def _effective_budget(self) -> float:
        if self._temporary_budget > 0:
            return self._temporary_budget
        return float(self.config.budget.monthly_limit or 0.0)

    def _active_overrides(self) -> list:
        out: list = []
        for item in self.config.overrides or []:
            group_id = str(getattr(item, "group_id", "") or "").strip()
            limit = float(getattr(item, "monthly_limit", 0.0) or 0.0)
            enabled = bool(getattr(item, "enabled", True))
            if enabled and group_id and limit > 0:
                out.append({"group_id": group_id, "monthly_limit": limit})
        return out

    async def _month_spend(self) -> float:
        """取本月累计花费。

        Host 的统计窗口是「最近 N 天」而非自然月，因此取 32 天的小时桶后，
        按本月 1 日 0 点裁剪求和。days 上限 365、top_models 上限 50 均已被 Host 归一化，
        这里显式取满以避免活跃模型被截断而少算花费。
        """
        start, end = month_bounds(datetime.now())
        try:
            raw = await self.ctx.statistics.local.model_trend(
                days=32,
                bucket="hour",
                top_models=50,
                metric="cost",
            )
        except Exception as exc:
            self.ctx.logger.warning(
                "读取花费统计失败，沿用上次值 %.4f 元：%s", self._decision.spend, exc
            )
            return self._decision.spend

        series = unwrap_series(raw)
        if not isinstance(series.get("values_by_key"), dict) or not series.get("timestamps"):
            # 结构不符时下面会算出 0，等价于「本月没花钱」→ 频率不会被收紧。
            # 这是静默降级，必须留痕：否则真机上只看到「插件没生效」，拿不到任何线索。
            self.ctx.logger.warning(
                "花费统计结构异常，本月花费按 0 处理（频率可能不会收紧）：keys=%s",
                sorted(series)[:6] if isinstance(series, dict) else type(raw).__name__,
            )

        total = series_total(raw, start, end)
        by_model = series_by_model(raw, start, end)
        top = sorted(by_model.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if top:
            self.ctx.logger.debug("本月花费前三位：%s", "，".join(f"{k}={v:.4f}" for k, v in top))
        return total

    async def _session_spend(self, session_id: str, start: datetime) -> float:
        """取单个会话的本月花费。

        Host 的 filters 只支持等值匹配（无范围查询），因此按 id 倒序拉最近若干条
        明细后自行按时间裁剪。达到扫描上限时会在日志提示可能偏低。
        """
        limit = max(100, int(self.config.budget.override_scan_limit or 2000))
        try:
            rows = await self.ctx.db.query(
                model_name="ModelUsage",
                query_type="get",
                filters={"session_id": session_id},
                order_by=["-id"],
                limit=limit,
            )
        except Exception as exc:
            self.ctx.logger.warning("查询会话花费失败(%s)：%s", session_id, exc)
            return 0.0

        if not isinstance(rows, list):
            return 0.0

        total = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            stamp = coerce_datetime(row.get("timestamp"))
            if stamp is None or stamp < start:
                continue
            try:
                total += float(row.get("cost") or 0.0)
            except (TypeError, ValueError):
                continue

        if len(rows) >= limit:
            self.ctx.logger.warning(
                "会话 %s 的花费明细已达扫描上限 %d 条，本月花费可能偏低", session_id, limit
            )
        return total

    # ---- 频率下发 ----

    async def _active_streams(self) -> list:
        try:
            streams = await self.ctx.chat.get_all_streams("qq")
        except Exception as exc:
            self.ctx.logger.warning("获取会话列表失败：%s", exc)
            return []
        if not isinstance(streams, list):
            return []
        return [item for item in streams if isinstance(item, dict)]

    def _target_streams(self, streams: list) -> list:
        out: list = []
        for item in streams:
            if not self.config.budget.apply_private and not item.get("is_group_session"):
                continue
            session_id = str(item.get("session_id") or item.get("stream_id") or "").strip()
            if session_id:
                out.append((session_id, str(item.get("group_id") or "").strip()))
        return out

    async def _apply_adjust(self, adjust: float) -> int:
        streams = await self._active_streams()
        if not streams:
            return 0

        overrides = {item["group_id"]: item["monthly_limit"] for item in self._active_overrides()}
        start, _ = month_bounds(datetime.now())
        now = datetime.now()
        applied = 0

        for session_id, group_id in self._target_streams(streams):
            value = adjust
            if group_id and group_id in overrides:
                spend = await self._session_spend(session_id, start)
                decision = compute_adjust(
                    spend,
                    overrides[group_id],
                    now,
                    p_floor_days=self.config.budget.p_floor_days,
                    deadband=self.config.budget.deadband,
                    min_adjust=self.config.budget.min_adjust,
                    max_adjust=self.config.budget.max_adjust,
                    allow_boost=self.config.budget.allow_boost,
                    over_budget_adjust=self.config.budget.over_budget_adjust,
                )
                # 双闸门取更严者：总账超了大家省，某群自己超了它额外再省
                value = min(value, decision.target)
            value = clamp(value, self.config.budget.min_adjust, self.config.budget.max_adjust)

            # 逐会话跳过：仅在「已记录且等于目标」时跳过宿主调用。
            # 无记录视为未知状态，必须同步一次（插件重载后宿主倍率未必一致）。
            # 宿主 set_adjust 会唤醒 Planner（烧钱），故已知无变更时绝不重复调用。
            if session_id in self._session_adjust and abs(self._session_adjust[session_id] - value) < 1e-6:
                continue

            try:
                result = await self.ctx.frequency.set_adjust(session_id, value)
            except Exception as exc:
                self.ctx.logger.debug("设置 %s 的倍率失败：%s", session_id, exc)
                continue
            if result:
                applied += 1
                self._session_adjust[session_id] = value
                self._known_sessions.add(session_id)
        return applied

    # ---- 状态持久化（宿主倍率是内存态，重启需恢复） ----

    def _state_path(self) -> Optional[Path]:
        """状态文件路径。

        仅使用 ctx.paths.data_dir（SDK 2.6.0+ 提供，插件自己的数据目录）；
        取不到时返回 None，状态仅存内存，重启后自动重算——
        不回退到插件源码目录：整目录更新/重装会覆盖源码目录，
        且与 SDK 推荐的用户数据路径不一致（审核意见，2026-09-22）。
        """
        paths = getattr(self.ctx, "paths", None)
        data_dir = getattr(paths, "data_dir", None)
        if data_dir in (None, ""):
            return None
        try:
            target = Path(str(data_dir))
            target.mkdir(parents=True, exist_ok=True)
            return target / STATE_FILE_NAME
        except Exception as exc:
            self.ctx.logger.debug("无法使用 ctx.paths.data_dir：%s", exc)
            return None

    def _reset_if_new_month(self, now: datetime) -> None:
        current = month_key(now)
        if not self._state_month:
            self._state_month = current
            return
        if self._state_month == current:
            return
        self.ctx.logger.info("跨月重置：%s -> %s，倍率复位", self._state_month, current)
        self._state_month = current
        self._applied_adjust = 1.0
        self._temporary_budget = 0.0
        self._known_sessions.clear()

    def _load_state(self) -> None:
        """读取持久化状态。文件缺失、坏 JSON、结构不符一律按空状态继续。"""
        try:
            path = self._state_path()
            if path is None or not path.is_file():
                self._state_month = month_key(datetime.now())
                return
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.ctx.logger.warning("读取预算状态失败：%s", exc)
            self._state_month = month_key(datetime.now())
            return

        if not isinstance(data, dict):
            self._state_month = month_key(datetime.now())
            return

        try:
            self._applied_adjust = float(data.get("applied_adjust") or 1.0)
        except (TypeError, ValueError):
            self._applied_adjust = 1.0
        try:
            self._temporary_budget = float(data.get("temporary_budget") or 0.0)
        except (TypeError, ValueError):
            self._temporary_budget = 0.0
        self._state_month = str(data.get("month") or "")

        current = month_key(datetime.now())
        if self._state_month != current:
            # 状态是上个月的，丢弃并复位
            self._applied_adjust = 1.0
            self._temporary_budget = 0.0
            self._state_month = current

    def _save_state(self) -> None:
        """落盘状态，用「临时文件 + 原子替换」避免写出半截 JSON。"""
        path = None
        try:
            path = self._state_path()
            if path is None:
                return
            payload = {
                "month": self._state_month or month_key(datetime.now()),
                "applied_adjust": round(self._applied_adjust, 6),
                "temporary_budget": round(self._temporary_budget, 6),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            temp = path.with_name(f"{path.name}.{time.time_ns()}.tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(path)
        except Exception as exc:
            self.ctx.logger.debug("保存预算状态失败：%s", exc)

    # ---- 工具方法 ----

    @staticmethod
    def _command_argument(text: str) -> str:
        """从完整消息里取出命令后面的参数部分。"""
        raw = (text or "").strip()
        if not raw:
            return ""
        import re

        matched = re.match(r"^\s*[/／]\s*(?:预算|budget)\s*(.*)$", raw, flags=re.IGNORECASE | re.DOTALL)
        if matched:
            return matched.group(1).strip()
        return raw

    @staticmethod
    def _message_text(obj: Any, _depth: int = 0) -> str:
        """从命令 kwargs 里尽力提取原始消息文本。

        限制递归深度：kwargs 是宿主传来的嵌套字典，一旦出现自引用（或某层包裹变深），
        无限递归会直接把插件进程打挂，而这类畸形输入静态审查看不出来。
        """
        if _depth > 4 or not isinstance(obj, dict):
            return ""
        for key in ("plain_text", "raw_message", "processed_plain_text", "text", "content"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for key in ("message", "event", "context"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                found = BudgetPacerPlugin._message_text(inner, _depth + 1)
                if found:
                    return found
            elif isinstance(inner, str) and inner.strip():
                return inner
        return ""

    @staticmethod
    def _stream_id(obj: Any) -> str:
        """从命令 kwargs / 会话对象里尽力提取 stream_id（字段名随版本有差异）。"""
        candidates: list = []
        if isinstance(obj, dict):
            candidates.append(obj)
            for key in ("message", "chat_stream", "event", "context"):
                if isinstance(obj.get(key), dict):
                    candidates.append(obj[key])
        for candidate in candidates:
            for key in ("stream_id", "chat_id", "session_id", "stream"):
                value = candidate.get(key)
                if isinstance(value, dict):
                    value = value.get("stream_id")
                if value:
                    return str(value)
        return ""

    @staticmethod
    def _normalize_admin_id(value: Any) -> str:
        """归一化账号 ID，兼容 "123456789" 与 "qq:123456789" 两种写法。"""
        text = str(value or "").strip()
        if not text:
            return ""
        return text.split(":")[-1].strip()

    def _is_admin(self, kwargs: dict) -> bool:
        """管理员校验：fail-close——未配置 admin_ids 时拒绝所有命令。

        本插件影响全 bot 的发言频率，一旦被误操作就是整月节奏被改；
        与只读类插件不同，这里默认拒绝比默认放行安全。
        拒绝时静默，只留日志。
        """
        admins = {
            self._normalize_admin_id(item) for item in (self.config.permission.admin_ids or [])
        }
        admins.discard("")
        if not admins:
            return False

        user_id = ""
        sources = [kwargs]
        if isinstance(kwargs.get("message"), dict):
            sources.append(kwargs["message"])
        for source in sources:
            for key in ("user_id", "sender_id", "from_user_id", "qq"):
                if source.get(key):
                    user_id = str(source[key])
                    break
            if user_id:
                break

        return self._normalize_admin_id(user_id) in admins

    async def _send_text(self, text: str, stream_id: str) -> bool:
        if not stream_id or not text:
            return False
        try:
            return bool(await self.ctx.send.text(text, stream_id))
        except Exception as exc:
            self.ctx.logger.error("发送预算状态失败：%s", exc)
            return False


def create_plugin():
    return BudgetPacerPlugin()
