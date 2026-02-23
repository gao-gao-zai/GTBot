from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque
from collections import deque

from nonebot import logger
from pydantic import BaseModel, ConfigDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from pydantic import SecretStr

from plugins.GTBot.model import Message
from . import LongMemoryContainer
from .tool import (
    add_event_log_info,
    add_group_profile_info,
    add_public_knowledge,
    add_user_profile_info,
    delete_event_log_info,
    delete_group_profile_info,
    delete_public_knowledge,
    delete_user_profile_info,
    get_event_log_info,
    get_group_profile_info,
    get_public_knowledge,
    get_user_profile_info,
    search_event_log_info,
    search_group_profile_info,
    search_public_knowledge,
    search_user_profile_info,
    update_event_log_info,
    update_group_profile_info,
    update_public_knowledge,
    update_user_profile_info,
)


def _extract_similarity_score(obj: Any) -> float | None:
    """尝试从命中项中提取相似度分数。

    说明：
        LongMemory 的 Qdrant 检索命中通常包含 `similarity` 字段（或 `score`）。
        本函数用于在入库预取阶段做“阀值过滤”。

    Args:
        obj: 命中对象。

    Returns:
        float | None: 相似度分数；若无法提取则返回 None。
    """

    if obj is None:
        return None

    for name in ("similarity", "score"):
        value = getattr(obj, name, None)
        if isinstance(value, (int, float)):
            return float(value)

    if isinstance(obj, dict):
        for key in ("similarity", "score"):
            value = obj.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def _take_hits_above_threshold(
    hits: Any,
    *,
    similarity_threshold: float,
    max_items: int,
) -> list[Any]:
    """按“相似度阀值 + 最大容量”筛选命中列表。

    规则：
        - 仅保留 `similarity >= similarity_threshold` 的条目。
        - 最多返回 `max_items` 条（命中通常已按相似度从高到低排序）。

    Args:
        hits: 命中项列表。
        similarity_threshold: 相似度阀值。
        max_items: 最大返回条数。

    Returns:
        list[Any]: 过滤后的命中列表。
    """

    if max_items <= 0:
        return []

    items = hits if isinstance(hits, list) else []

    out: list[Any] = []
    for item in items:
        score = _extract_similarity_score(item)
        if score is None:
            continue
        if score < float(similarity_threshold):
            continue
        out.append(item)
        if len(out) >= int(max_items):
            break

    return out


def _to_jsonable(obj: Any) -> Any:
    """将对象转换为可 JSON 序列化的结构。

    Args:
        obj: 任意对象。

    Returns:
        Any: 可被 json.dumps 处理的对象。
    """

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _format_ingest_input(
    *,
    runtime_ctx: "LongMemoryIngestRuntimeContext",
    messages: list[Message],
    prefetch: dict[str, Any],
    max_message_chars: int = 800,
) -> str:
    """将入库输入格式化为更易读的文本。

    说明：
        目标是让 LLM 更容易理解输入内容：
        - 消息列表（按序号）
        - 预取上下文（仍保留 JSON 块，便于精确引用）

    Args:
        runtime_ctx: ToolRuntime 上下文。
        messages: 本次要整理的消息列表。
        prefetch: 预取上下文。
        max_message_chars: 单条消息 content 最大展示字符数。

    Returns:
        str: 格式化后的文本。
    """

    def _clip(text: str) -> str:
        s = str(text or "")
        if len(s) <= int(max_message_chars):
            return s
        return s[: int(max_message_chars)] + "…(truncated)"

    lines: list[str] = []
    lines.append("# 长期记忆入库输入")
    lines.append("")
    lines.append("## 最近聊天消息（按时间顺序）")
    if not messages:
        lines.append("(empty)")
    else:
        for idx, m in enumerate(messages, start=1):
            mid = getattr(m, "message_id", None)
            uid = getattr(m, "user_id", None)
            uname = getattr(m, "user_name", None)
            ts = getattr(m, "send_time", None)
            content = _clip(str(getattr(m, "content", "") or ""))
            lines.append(f"{idx}. message_id={mid} user_id={uid} user_name={uname} send_time={ts}")
            lines.append(f"   content: {content}")

    lines.append("")
    lines.append("## 预取上下文（JSON）")
    lines.append(
        "以下是与本次消息可能相关的长期记忆命中项（已做阀值/容量过滤）。\n"
        "请优先依据这些信息做去重与更新（update），必要时再新增（add）。"
    )
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            _to_jsonable(prefetch),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    lines.append("```")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LongMemoryIngestConfig:
    """长期记忆入库管理器配置。

    说明：
        A/B/C/D/E 来自你的需求描述。

        - `processed_capacity` (A)：已处理消息最多保留条数。
        - `pending_capacity` (B)：未处理消息最多保留条数（超出会丢弃并告警）。
        - `flush_pending_threshold` (C)：当未处理消息数达到该阈值时立即触发一次整理。
        - `idle_flush_seconds` (D)：当 D 秒内无新消息加入时触发一次整理。
        - `flush_max_messages` (E)：每次整理最多交给入库 LLM 的消息条数（取最近的 E 条）。

    Attributes:
        processed_capacity: 已处理队列容量 A。
        pending_capacity: 未处理队列容量 B。
        flush_pending_threshold: 阈值触发 C。
        idle_flush_seconds: 空闲触发 D（秒）。
        flush_max_messages: 每次最多处理 E 条。
        max_concurrent_flushes: 全局并发入库数限制。
        model_id: 入库 LLM 的 model id。
        base_url: 入库 LLM 的 base_url。
        api_key: 入库 LLM 的 api_key。
        model_parameters: OpenAI 兼容参数透传（会剔除 streaming）。
        public_knowledge_similarity_threshold: 公共知识相似度阀值（仅保留 similarity>=阀值的命中）。
        public_knowledge_max_items: 公共知识命中最大返回条数（容量上限）。
        event_log_similarity_threshold: 事件日志相似度阀值（仅保留 similarity>=阀值的命中）。
        event_log_max_items: 事件日志命中最大返回条数（容量上限）。
        user_profile_similarity_threshold: 用户画像相似度阀值（仅保留 similarity>=阀值的命中）。
        user_profile_max_items: 用户画像命中最大返回条数（容量上限）。
        group_profile_min_items_threshold: 群画像条目数阀值（SQLite 无相似度分数，使用“条目数”作为阀值）。
        group_profile_max_items: 群画像最大返回条数（SQLite 按时间排序，容量上限）。
    """

    processed_capacity: int = 200
    pending_capacity: int = 200
    flush_pending_threshold: int = 20
    idle_flush_seconds: float = 60.0
    flush_max_messages: int = 20

    max_concurrent_flushes: int = 2

    model_id: str = ""
    base_url: str = ""
    api_key: str = ""
    model_parameters: dict[str, Any] = field(default_factory=dict)

    public_knowledge_similarity_threshold: float = 0.0
    public_knowledge_max_items: int = 5

    event_log_similarity_threshold: float = 0.0
    event_log_max_items: int = 5

    user_profile_similarity_threshold: float = 0.0
    user_profile_max_items: int = 10

    group_profile_min_items_threshold: int = 0
    group_profile_max_items: int = 20

    def validate(self) -> "LongMemoryIngestConfig":
        """校验并归一化配置。

        Returns:
            LongMemoryIngestConfig: 校验后的配置（同一实例）。

        Raises:
            ValueError: 当配置不满足基本约束时。
        """

        if self.processed_capacity <= 0:
            raise ValueError("processed_capacity(A) 必须 > 0")
        if self.pending_capacity <= 0:
            raise ValueError("pending_capacity(B) 必须 > 0")
        if self.flush_pending_threshold <= 0:
            raise ValueError("flush_pending_threshold(C) 必须 > 0")
        if self.flush_max_messages <= 0:
            raise ValueError("flush_max_messages(E) 必须 > 0")
        if self.idle_flush_seconds <= 0:
            raise ValueError("idle_flush_seconds(D) 必须 > 0")
        if self.flush_max_messages > self.pending_capacity:
            raise ValueError("E 不能大于 B（flush_max_messages > pending_capacity）")
        if self.flush_pending_threshold > self.pending_capacity:
            raise ValueError("C 不能大于 B（flush_pending_threshold > pending_capacity）")
        if self.max_concurrent_flushes <= 0:
            raise ValueError("max_concurrent_flushes 必须 > 0")

        if not (0.0 <= float(self.public_knowledge_similarity_threshold) <= 1.0):
            raise ValueError("public_knowledge_similarity_threshold 必须在 [0, 1] 范围内")
        if self.public_knowledge_max_items <= 0:
            raise ValueError("public_knowledge_max_items 必须 > 0")

        if not (0.0 <= float(self.event_log_similarity_threshold) <= 1.0):
            raise ValueError("event_log_similarity_threshold 必须在 [0, 1] 范围内")
        if self.event_log_max_items <= 0:
            raise ValueError("event_log_max_items 必须 > 0")

        if not (0.0 <= float(self.user_profile_similarity_threshold) <= 1.0):
            raise ValueError("user_profile_similarity_threshold 必须在 [0, 1] 范围内")
        if self.user_profile_max_items <= 0:
            raise ValueError("user_profile_max_items 必须 > 0")

        if self.group_profile_max_items <= 0:
            raise ValueError("group_profile_max_items 必须 > 0")
        if self.group_profile_min_items_threshold < 0:
            raise ValueError("group_profile_min_items_threshold 必须 >= 0")
        if self.group_profile_min_items_threshold > self.group_profile_max_items:
            raise ValueError("group_profile_min_items_threshold 不能大于 group_profile_max_items")
        return self


class LongMemoryIngestRuntimeContext(BaseModel):
    """入库 LLM 的最小 ToolRuntime 上下文。

    说明：
        - LongMemory 工具只依赖 `runtime.context.long_memory/session_id/group_id/user_id`。
        - 为避免依赖完整的 `GroupChatContext`，这里定义最小字段集合。

    Attributes:
        long_memory: LongMemory 服务容器。
        session_id: 会话 ID（建议直接提供，格式：group_<群号>/private_<QQ号>）。
        group_id: 群号（可选）。
        user_id: 用户 ID（可选）。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    long_memory: Any
    session_id: str
    group_id: int | None = None
    user_id: int | None = None


@dataclass(slots=True)
class _SessionState:
    """单个会话的入库状态。"""

    processed: Deque[Message]
    pending: Deque[Message]
    seq: int = 0
    last_msg_at: float = 0.0
    last_flush_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    idle_task: asyncio.Task[None] | None = None


def _default_ingest_prompt() -> str:
    """默认入库提示词。

    Returns:
        str: system prompt。
    """

    return """
# 角色
你是一个客观的记忆记录员。你的任务是如实记录聊天中发生的所有事实与变化，尽量做到“应记尽记，简洁明了”，但必须放到正确的记忆类型里。

# 输入格式
你将收到以下格式的输入：

## 1. 最近聊天消息（按时间顺序）
每条消息包含：
- message_id: 消息 ID
- user_id: 发送者 QQ 号
- user_name: 发送者昵称（群聊时有）
- send_time: 发送时间（Unix 时间戳）
- content: 消息内容

## 2. 预取上下文（JSON 格式）
包含与本次消息可能相关的记忆命中：
- event_log_hits: 事件日志命中（同 session 的历史事件）
- user_profile_hits: 用户画像命中（相关用户的个人信息）
- group_profile: 群画像（当前群组特征）
- group_profile_hits: 群画像语义命中（同群的相关条目）
- public_knowledge_hits: 公共知识命中（通用知识/事实）

# 记录原则
1. **客观记录**：只记录事实，不做价值判断
2. **应记尽记**：尽量让每条输入消息都“在记忆里有落点”（事件日志/用户画像/群画像/公共知识至少其一）
3. **放对地方**：不同类型的内容必须写入对应的记忆工具；不要把“群内即时对话/诉求”误写为公共知识
4. **优先更新**：**写入前必须检查预取上下文**。如果已有相似记录，必须优先选择更新（update/upsert）而不是新增（add）。
5. **不编造**：只根据聊天内容记录，不添加未提及的信息
6. **排除通用常识**：**严禁将 AI 预训练知识中已存在的通用事实写入公共知识**（如：MC 合成配方、科学定义、公开菜谱、通用代码教程）。
   - 此类内容若被讨论，仅记录为【事件日志】。
   - 除非聊天中产生了**特定变体、群内共识、独特见解或上下文相关的结论**，否则不允许写入公共知识。
7. **简洁明了**：记录内容需**精炼**。去除冗余对话，保留核心事实；使用结构化要点，避免长篇大论。

# 分流决策树（非常重要）
请按以下顺序决定写入位置（从上到下匹配，命中即用对应工具；必要时可多条工具组合）：

## A. 事件日志（event_log）——优先级最高
凡是“本次会话/近期发生了什么、谁说了什么、提出了什么、讨论了什么、决定了什么、进行了什么活动”，都属于事件日志。
典型关键词：提议、想做/想要、请求、讨论、问答、吐槽、争论、计划、进度、约定、提醒、投票、链接分享、发图。
*注意：即使讨论的是通用知识（如“怎么做菜”），讨论过程本身也应记录为事件日志，而非公共知识。*

## B. 用户画像（user_profile）
凡是“某个用户相对稳定的属性/偏好/习惯/长期计划/与他人关系”，才写用户画像。
判断要点：离开当前场景过一周/一个月仍可能成立。
*操作指引：若预取上下文中有该用户相似画像，请合并内容后更新。*

## C. 群画像（group_profile）
凡是“整个群的长期特征/群规/常见话题/群文化/群公告”，才写群画像。
注意：群里某次临时讨论不算群画像，临时讨论写事件日志。
    
也包括“群内工具/机器人/插件说明”：
- 本群内某个机器人（如“某某 Bot"）的功能、命令、使用步骤、注意事项、FAQ。
- 这类信息依赖群内环境/配置，换一个群可能不成立，因此不得写入公共知识。
*操作指引：若预取上下文中有相似群画像，请合并内容后更新。*

## D. 公共知识（public_knowledge）——最严格
**前置过滤（一票否决）：**
- **通用常识过滤**：如果内容是 AI 已知的通用事实/教程/定义（如游戏攻略、科学定义），**禁止写入**。
- 包含群内专有实体（机器人名/内部系统名/群内代号/特定群配置）。
- 依赖群权限或群配置才能成立的流程/命令说明。
- 无法保证“换一个群/换一个人也成立”。
- 记录的是“某人想要/需要/提出的需求/计划/请求”本身。

只有满足以下【全部】条件，才允许写 public_knowledge：
1) **独特性**：不是 AI 预训练知识中随处可见的内容，而是聊天中产生的**独特结论、特定配置、 niche 知识或经过讨论确认的特殊规则**。
2) 去标识化：不得包含任何 QQ 号、昵称、群号、群内指代（“群友/我们群”也尽量不要）
3) 可复用：换一个群/换一个人也成立（通用事实、常识、教程、方法论、明确规则）
4) 非诉求：不得记录“某人想要/需要/提出的需求/计划/请求”本身

*操作指引：调用此工具时，若发现预取上下文中有相似知识，请务必使用 mode: "upsert" 进行更新。*

# 公共知识写入规范（当且仅当满足 D）
1. 标题必须是通用标题（不出现人名/群名/QQ 号）
2. 内容必须是可复用的结构化知识（要点/步骤/定义/规则），不要复述聊天原话
3. 严禁把对话上下文、个人诉求、具体成员信息写入 public_knowledge

# 正反例（用于防止误判）

## 反例 1（禁止写入 public_knowledge - 通用常识）
输入："群友 A 问 MC 里铁镐怎么合成，群友 B 回答 3 个铁锭 2 根木棍"
正确做法：
- 写事件日志：记录"A 询问合成配方，B 进行了回答"
- **不写公共知识**：因为这是游戏通用配方，AI 已知，无需存储
错误做法：
- 把"MC 铁镐合成配方”写入公共知识

## 反例 2（禁止写入 public_knowledge - 个人诉求）
输入："不想取名了 (3067684279) 提出想玩一款小游戏，并询问怎么制作"
正确做法：
- 写事件日志：记录“谁提出了什么需求/讨论了什么/是否有人回复"
- 视情况写用户画像：如果该用户反复表现出“喜欢小游戏/想学习制作游戏”的稳定倾向，可补充到用户画像
错误做法：
- 把上述原话（含 QQ 号/诉求）写成公共知识

## 正例（允许写入 public_knowledge - 独特结论）
输入：群内讨论后得出一个非通用的结论，例如“本群约定将所有项目版本号统一采用 YYYY.MM.DD 格式”
正确做法：
- 标题："项目版本号命名规范约定"
- 内容："统一采用 YYYY.MM.DD 格式（年。月。日），适用于群内所有协作项目"
- 工具参数：`mode: "upsert"`

# 工具使用说明

## add_event_log_info - 记录事件日志
记录聊天中发生的具体事件、对话、活动。
参数：
- details: 事件详情（必填）
- event_name: 事件名称（可选，如"闲聊"、"讨论"、"提问"等）
- relevant_members: 相关成员列表 [user_id, ...]
- time_slots: 时间段 [{"start_time": 时间戳，"end_time": 时间戳}]

适用场景：
- 有人发言/提问/回答
- 群内活动/讨论
- 任何聊天中发生的事件

成员指代规范：
- 在 details 中提到成员时，使用 "昵称 (QQ 号)" 格式
- 示例："小明 (12345678) 问了一个问题，小红 (87654321) 回答了"

## add_user_profile_info - 记录用户画像
记录用户的个人特征、偏好、习惯、关系等持久性信息。
参数：
- user_id: 用户 QQ 号
- info: 信息内容（字符串或字符串列表）

适用场景：
- 用户表达偏好（如"我喜欢喝奶茶"）
- 用户提及个人信息（年龄、职业、爱好等）
- 用户提及与其他人的关系

## add_group_profile_info - 记录群画像
记录群组整体特征、群规、活跃话题、群体文化等。
参数：
- group_id: 群号
- info: 信息内容（字符串或字符串列表）
- category: 分类（可选）

适用场景：
- 群名称/群介绍/群规
- 群内常见话题/活动类型
- 群体特征/文化

## add_public_knowledge - 记录公共知识
记录对所有人有用的通用知识、事实。
参数：
- title: 标题
- content: 内容
- mode: "add" 或 "upsert"（默认 upsert，会按相似度去重）
- threshold: 相似度阈值（默认 0.85）

适用场景：
- 客观事实/常识
- 游戏/动漫/技术等相关信息
- 对群体有参考价值的知识
- **注意**：请勿记录 AI 已知的通用常识（如基础游戏攻略、科学定义），仅记录聊天中产生的独特结论或特定规则。

# 输出要求
在执行完所有工具调用后，用简短的一句话总结本次记录的内容。
"""


class LongMemoryIngestManager:
    """长期记忆入库管理器。

    该管理器按 session_id 隔离维护两段队列：
        - processed: 已经整理过的消息（最多 A 条）。
        - pending: 尚未整理的消息（最多 B 条，溢出会丢弃并告警）。

    当触发条件满足时（C 阈值或 D 空闲），会把最近最多 E 条消息交给入库 LLM，
    并附带预取的公共知识/事件/画像上下文，驱动 LLM 使用 LongMemory 工具完成整理。

    Attributes:
        config: 入库配置。
        long_memory: LongMemory 服务容器。
    """

    def __init__(
        self,
        *,
        config: LongMemoryIngestConfig,
        long_memory: LongMemoryContainer,
        ingest_prompt: str | None = None,
        ingest_runner: Callable[[LongMemoryIngestRuntimeContext, list[Message], dict[str, Any]], Awaitable[str]]
        | None = None,
    ) -> None:
        """初始化入库管理器。

        Args:
            config: 入库配置。
            long_memory: LongMemory 服务容器。
            ingest_prompt: 可选自定义 system prompt。
            ingest_runner: 可选自定义执行器（用于测试/替换 LLM）。
        """

        self.config: LongMemoryIngestConfig = config.validate()
        self.long_memory: LongMemoryContainer = long_memory
        self._sessions: dict[str, _SessionState] = {}
        self._global_sem = asyncio.Semaphore(int(self.config.max_concurrent_flushes))
        self._ingest_prompt: str = ingest_prompt or _default_ingest_prompt()
        self._ingest_runner = ingest_runner or self._default_ingest_runner

    def _get_or_create_session(self, session_id: str) -> _SessionState:
        """获取或创建会话状态。"""

        sid = str(session_id).strip()
        if not sid:
            sid = "test_session"

        state = self._sessions.get(sid)
        if state is not None:
            return state

        state = _SessionState(
            processed=deque(maxlen=int(self.config.processed_capacity)),
            pending=deque(maxlen=int(self.config.pending_capacity)),
        )
        self._sessions[sid] = state
        return state

    async def add_message(
        self,
        *,
        session_id: str,
        message: Message,
        group_id: int | None = None,
        user_id: int | None = None,
    ) -> None:
        """向指定会话追加一条消息，并触发必要的入库。

        Args:
            session_id: 会话 ID。
            message: 消息对象（Pydantic Message）。
            group_id: 可选群号。
            user_id: 可选用户 ID。
        """

        sid = str(session_id).strip() or "test_session"
        state = self._get_or_create_session(sid)

        async with state.lock:
            now = float(time.time())
            state.last_msg_at = now

            # pending deque 本身有 maxlen，但我们要在丢弃时告警；因此手动控制。
            if len(state.pending) >= int(self.config.pending_capacity):
                dropped = state.pending.popleft()
                logger.warning(
                    f"LongMemory pending 队列溢出：已丢弃最旧未处理消息（session={sid} "
                    f"message_id={getattr(dropped, 'message_id', None)} "
                    f"user_id={getattr(dropped, 'user_id', None)}）"
                )

            state.pending.append(message)

            # 重置 idle flush 任务
            if state.idle_task is not None and not state.idle_task.done():
                state.idle_task.cancel()
            state.idle_task = asyncio.create_task(
                self._idle_flush_after(
                    session_id=sid,
                    group_id=group_id,
                    user_id=user_id,
                    delay_seconds=float(self.config.idle_flush_seconds),
                )
            )

            if len(state.pending) >= int(self.config.flush_pending_threshold):
                # 阈值触发：不在锁内执行重活
                asyncio.create_task(
                    self.flush_session(session_id=sid, group_id=group_id, user_id=user_id, reason="threshold")
                )

    async def _idle_flush_after(
        self,
        *,
        session_id: str,
        group_id: int | None,
        user_id: int | None,
        delay_seconds: float,
    ) -> None:
        """空闲触发：在 D 秒无新消息后 flush。"""

        try:
            await asyncio.sleep(float(delay_seconds))
            await self.flush_session(session_id=session_id, group_id=group_id, user_id=user_id, reason="idle")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(
                f"LongMemory idle flush 任务异常（session={session_id}）: {type(exc).__name__}: {exc!r}"
            )
            logger.error(traceback.format_exc())

    async def flush_session(
        self,
        *,
        session_id: str,
        group_id: int | None,
        user_id: int | None,
        reason: str,
    ) -> None:
        """对某个会话执行一次入库整理。"""

        sid = str(session_id).strip() or "test_session"
        state = self._get_or_create_session(sid)

        # 单 session 互斥：避免 threshold/idle 重复触发
        async with state.lock:
            if not state.pending:
                return

            # 取最近 E 条（保持原顺序）
            e = int(self.config.flush_max_messages)
            batch = list(state.pending)[-e:]

            # 从 pending 中移除这段尾部
            for _ in range(min(len(state.pending), len(batch))):
                state.pending.pop()

        # 全局并发限制
        async with self._global_sem:
            try:
                extra = await self._prefetch_context(sid, group_id=group_id, user_id=user_id, messages=batch)
                runtime_ctx = LongMemoryIngestRuntimeContext(
                    long_memory=self.long_memory,
                    session_id=sid,
                    group_id=group_id,
                    user_id=user_id,
                )
                summary = await self._ingest_runner(runtime_ctx, batch, extra)
                logger.info(
                    f"LongMemory 入库完成（session={sid} reason={reason}）: {str(summary)[:500]}"
                )

                async with state.lock:
                    for m in batch:
                        state.processed.append(m)
                    state.last_flush_at = float(time.time())
            except Exception as exc:
                logger.error(
                    f"LongMemory 入库失败（session={sid} reason={reason} batch={len(batch)}）: "
                    f"{type(exc).__name__}: {exc!r}"
                )
                logger.error(traceback.format_exc())
                # 失败：把 batch 重新塞回 pending 的尾部，尽量不丢
                async with state.lock:
                    for m in batch:
                        if len(state.pending) >= int(self.config.pending_capacity):
                            dropped = state.pending.popleft()
                            logger.warning(
                                f"LongMemory pending 回滚时溢出：丢弃最旧未处理消息（session={sid} "
                                f"message_id={getattr(dropped, 'message_id', None)}）"
                            )
                        state.pending.append(m)

    async def _prefetch_context(
        self,
        session_id: str,
        *,
        group_id: int | None,
        user_id: int | None,
        messages: list[Message],
    ) -> dict[str, Any]:
        """预取与本次消息可能相关的长期记忆上下文。

        Args:
            session_id: 会话 ID。
            group_id: 群号。
            user_id: 用户 ID。
            messages: 本次要整理的消息列表。

        Returns:
            dict[str, Any]: 预取上下文，供入库 LLM 参考。
        """

        query_text = "\n".join([str(getattr(m, "content", "") or "") for m in messages]).strip()
        query_text = query_text[:800]

        out: dict[str, Any] = {
            "session_id": session_id,
            "group_id": group_id,
            "user_id": user_id,
            "query": query_text,
        }

        # 公共知识
        try:
            hits = await self.long_memory.public_knowledge.search_public_knowledge(
                query_text,
                n_results=int(self.config.public_knowledge_max_items),
                min_similarity=float(self.config.public_knowledge_similarity_threshold),
            )
            filtered = _take_hits_above_threshold(
                hits,
                similarity_threshold=float(self.config.public_knowledge_similarity_threshold),
                max_items=int(self.config.public_knowledge_max_items),
            )
            out["public_knowledge_hits"] = [_to_jsonable(h) for h in filtered]
        except Exception as exc:
            out["public_knowledge_hits"] = []
            out["public_knowledge_error"] = str(exc)

        # 事件日志（限定 session）
        try:
            hits2 = await self.long_memory.event_log_manager.search_events(
                query_text,
                n_results=int(self.config.event_log_max_items),
                session_id=session_id,
                min_similarity=float(self.config.event_log_similarity_threshold),
                order_by="similarity",
                order="desc",
            )
            filtered2 = _take_hits_above_threshold(
                hits2,
                similarity_threshold=float(self.config.event_log_similarity_threshold),
                max_items=int(self.config.event_log_max_items),
            )
            out["event_log_hits"] = [_to_jsonable(h) for h in filtered2]
        except Exception as exc:
            out["event_log_hits"] = []
            out["event_log_error"] = str(exc)

        # 用户画像（全库）
        try:
            hits3 = await self.long_memory.user_profile_manager.search_user_profiles(
                query_text,
                n_results=int(self.config.user_profile_max_items),
                order_by="similarity",
                order="desc",
            )
            filtered3 = _take_hits_above_threshold(
                hits3,
                similarity_threshold=float(self.config.user_profile_similarity_threshold),
                max_items=int(self.config.user_profile_max_items),
            )
            out["user_profile_hits"] = [_to_jsonable(h) for h in filtered3]
        except Exception as exc:
            out["user_profile_hits"] = []
            out["user_profile_error"] = str(exc)

        # 群画像（仅群聊）
        if group_id and int(group_id) > 0:
            try:
                # 语义命中（用于提示 LLM 做去重与 update）
                hits4 = await self.long_memory.group_profile_manager.search_group_profiles(
                    query_text,
                    group_id=int(group_id),
                    n_results=int(self.config.group_profile_max_items),
                    min_similarity=None,
                    order_by="similarity",
                    order="desc",
                    touch_last_called=True,
                )
                out["group_profile_hits"] = [_to_jsonable(h) for h in hits4]

                profile = await self.long_memory.group_profile_manager.get_group_profiles(
                    int(group_id),
                    limit=int(self.config.group_profile_max_items),
                    sort_by="last_updated",
                    sort_order="desc",
                )
                desc = getattr(profile, "description", None)
                if isinstance(desc, list) and len(desc) < int(self.config.group_profile_min_items_threshold):
                    out["group_profile"] = None
                else:
                    out["group_profile"] = _to_jsonable(profile)
            except Exception as exc:
                out["group_profile_hits"] = []
                out["group_profile"] = None
                out["group_profile_error"] = str(exc)
        else:
            out["group_profile_hits"] = []
            out["group_profile"] = None

        return out

    def _resolve_llm_settings(self) -> tuple[str, str, str, float | None, dict[str, Any]]:
        """解析入库 LLM 的连接参数。

        Returns:
            tuple[str, str, str, float | None, dict[str, Any]]: (model_id, base_url, api_key, temperature, model_kwargs)
        """

        model_id = (self.config.model_id or "").strip()
        base_url = (self.config.base_url or "").strip()
        api_key = (self.config.api_key or "").strip()

        model_kwargs: dict[str, Any] = dict(self.config.model_parameters or {})

        temperature: float | None = None
        if "temperature" in model_kwargs:
            raw_temp = model_kwargs.pop("temperature", None)
            if isinstance(raw_temp, (int, float)):
                temperature = float(raw_temp)

        # 移除 streaming 相关参数（避免透传未知字段）
        for k in ["stream", "streaming", "stream_chunk_chars", "stream_flush_interval_sec"]:
            model_kwargs.pop(k, None)

        return model_id, base_url, api_key, temperature, model_kwargs

    def _build_long_memory_tools(self) -> list[BaseTool]:
        """构建入库可用的 LongMemory 工具列表。

        Returns:
            list[BaseTool]: 工具列表。
        """

        return [
            add_user_profile_info,
            delete_user_profile_info,
            update_user_profile_info,
            get_user_profile_info,
            search_user_profile_info,
            add_group_profile_info,
            delete_group_profile_info,
            update_group_profile_info,
            get_group_profile_info,
            search_group_profile_info,
            add_event_log_info,
            get_event_log_info,
            search_event_log_info,
            update_event_log_info,
            delete_event_log_info,
            add_public_knowledge,
            get_public_knowledge,
            search_public_knowledge,
            update_public_knowledge,
            delete_public_knowledge,
        ]

    async def _default_ingest_runner(
        self,
        runtime_ctx: LongMemoryIngestRuntimeContext,
        messages: list[Message],
        extra_context: dict[str, Any],
    ) -> str:
        """默认入库执行器：调用 LLM + LongMemory 工具完成整理。

        Args:
            runtime_ctx: ToolRuntime 上下文。
            messages: 本次消息列表。
            extra_context: 预取上下文。

        Returns:
            str: LLM 的摘要输出。
        """

        model_id, base_url, api_key, temperature, model_kwargs = self._resolve_llm_settings()
        if not model_id or not base_url:
            raise ValueError("缺少入库 LLM 配置：model_id/base_url 不能为空")

        model = ChatOpenAI(
            model=model_id,
            base_url=base_url,
            api_key=SecretStr(api_key or ""),
            temperature=temperature,
            streaming=False,
            model_kwargs=model_kwargs,
        )

        tools = self._build_long_memory_tools()
        agent = create_agent(
            model=model,
            tools=tools,
            context_schema=LongMemoryIngestRuntimeContext,
            middleware=[],
        )

        chat_context = [
            SystemMessage(content=self._ingest_prompt),
            HumanMessage(
                content=(
                    "请根据以下输入整理长期记忆，并使用工具进行写入/更新/删除。\n\n"
                    + _format_ingest_input(
                        runtime_ctx=runtime_ctx,
                        messages=messages,
                        prefetch=extra_context,
                    )
                )
            ),
        ]

        result = await agent.ainvoke(
            input={"messages": chat_context},
            context=runtime_ctx,
        )

        msgs = result.get("messages", []) if isinstance(result, dict) else []
        if not msgs:
            return "(no messages)"

        last = msgs[-1]
        content = getattr(last, "content", "")
        return str(content or "").strip() or "(empty)"
