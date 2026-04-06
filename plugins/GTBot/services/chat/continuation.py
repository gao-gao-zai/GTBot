from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from time import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from nonebot import logger
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.message import Message

from ...ConfigManager import total_config
from ...llm_provider import build_chat_model
from ...model import GroupMessage, GroupMessageRecord
from ..message import GroupMessageManager


config = total_config.processed_configuration.current_config_group


@dataclass(slots=True)
class ContinuationWindowState:
    """描述单个群聊续聊窗口的运行时状态。

    `opened_at` 和 `expires_at` 只表示窗口真正开始观察后续消息的时间范围，
    也就是主链路响应完成后的关窗计时基准。`history_started_at` 单独记录
    “可纳入续聊判断的聊天记录起点”，用于把机器人响应期间新到达的用户消息
    一并纳入首轮分析，但不会改变窗口超时的起算点。
    """

    session_id: str
    group_id: int
    opened_at: float
    expires_at: float
    source_trigger_mode: str
    last_response_id: str
    last_bot_message_id: int | None = None
    history_started_at: float | None = None
    buffered_message_ids: list[int] = field(default_factory=list)
    pending_message_ids: list[int] = field(default_factory=list)
    accumulated_message_ids: list[int] = field(default_factory=list)
    debounce_task: asyncio.Task[None] | None = None
    expiry_task: asyncio.Task[None] | None = None
    closed: bool = False


@dataclass(slots=True)
class ContinuationAnalysisResult:
    should_reply: bool
    confidence: float | None
    reason: str
    focus_message_ids: list[int]
    raw_output: str


def _normalize_ai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    chunks.append(str(text))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content or "")


def _truncate_text(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 3)]}..."


class ContinuationManager:
    """管理群聊续聊窗口及其小模型判定流程。

    该管理器只维护内存态，不主动生成回复内容。它负责在主链路成功结束后开窗、
    聚合窗口内的新增消息、按防抖时机调用小模型做二分类判断，并在需要时重新回到
    现有 `run_chat_turn` 主链路。窗口计时始终从机器人本轮响应结束时开始，但消息
    历史的纳入范围可以从响应开始时刻向后覆盖。
    """

    def __init__(self) -> None:
        self._windows: dict[str, ContinuationWindowState] = {}
        self._lock = asyncio.Lock()

    def is_enabled(self) -> bool:
        cfg = config.chat_model.continuation
        return bool(cfg.enabled and cfg.analyzer_model_id and cfg.analyzer_provider)

    async def has_active_window(self, session_id: str) -> bool:
        async with self._lock:
            state = self._windows.get(str(session_id))
            if state is None:
                return False
            if state.closed or state.expires_at <= time():
                return False
            return True

    async def open_window(
        self,
        *,
        session_id: str,
        group_id: int,
        source_trigger_mode: str,
        last_response_id: str,
        last_bot_message_id: int | None = None,
        trigger_started_at: float | None = None,
        bot: Bot | None = None,
        msg_mg: GroupMessageManager | None = None,
        cache: Any | None = None,
    ) -> None:
        """为指定群会话开启新的续聊窗口。

        窗口超时始终从当前调用时刻开始计算，不受 `trigger_started_at` 影响。
        `trigger_started_at` 仅用作聊天记录起点：若机器人处理本轮请求期间已经有
        新的群消息落库，则会在开窗后立即预加载这些消息，使它们成为续聊分析的
        首批候选输入，但分析动作本身仍然只会在响应结束之后发生。

        Args:
            session_id: 续聊窗口所属的会话 ID。
            group_id: 群号。
            source_trigger_mode: 本轮机器人回复最初的触发方式。
            last_response_id: 本轮主链路生成的响应 ID。
            last_bot_message_id: 本轮最近一条机器人出站消息 ID。
            trigger_started_at: 本轮开始响应前最后一条非机器人消息的时间戳。
            bot: 当前机器人实例，用于过滤机器人自身消息。
            msg_mg: 消息管理器，用于预加载响应期间落库的新消息。
            cache: 续聊真正触发主链路时要继续透传的缓存对象。
        """
        if not self.is_enabled():
            return

        cfg = config.chat_model.continuation
        now = time()
        new_state = ContinuationWindowState(
            session_id=str(session_id),
            group_id=int(group_id),
            opened_at=now,
            expires_at=now + float(cfg.window_seconds),
            history_started_at=float(trigger_started_at) if trigger_started_at is not None else None,
            source_trigger_mode=str(source_trigger_mode or "unknown"),
            last_response_id=str(last_response_id or ""),
            last_bot_message_id=int(last_bot_message_id) if last_bot_message_id else None,
        )

        old_state: ContinuationWindowState | None = None
        async with self._lock:
            old_state = self._windows.get(new_state.session_id)
            self._windows[new_state.session_id] = new_state
            new_state.expiry_task = asyncio.create_task(
                self._expire_window_after_timeout(new_state.session_id, new_state.expires_at)
            )

        await self._cancel_state_tasks(old_state)
        logger.debug(
            "opened continuation window: session={} group_id={} trigger={} last_response_id={} opened_at={} expires_at={} history_started_at={} pre_history_messages={}",
            new_state.session_id,
            new_state.group_id,
            new_state.source_trigger_mode,
            new_state.last_response_id,
            new_state.opened_at,
            new_state.expires_at,
            new_state.history_started_at,
            int(getattr(cfg, "pre_history_messages", 0) or 0),
        )

        if bot is None or msg_mg is None:
            return
        await self._prime_window_with_buffered_messages(
            state=new_state,
            bot=bot,
            msg_mg=msg_mg,
            cache=cache,
        )

    async def close_window(self, session_id: str, *, reason: str) -> None:
        state = await self._pop_window(session_id, reason=reason)
        await self._cancel_state_tasks(state)

    async def register_incoming_message(
        self,
        *,
        session_id: str,
        group_id: int,
        message_id: int,
        bot: Bot,
        msg_mg: GroupMessageManager,
        cache: Any,
    ) -> None:
        cfg = config.chat_model.continuation
        forced_state: ContinuationWindowState | None = None
        scheduled_state: ContinuationWindowState | None = None
        debounce_delay = float(cfg.debounce_seconds)

        async with self._lock:
            state = self._windows.get(str(session_id))
            if state is None or state.closed or state.expires_at <= time():
                return

            msg_id = int(message_id)
            if msg_id in state.accumulated_message_ids or msg_id in state.buffered_message_ids:
                return

            state.pending_message_ids.append(msg_id)
            state.accumulated_message_ids.append(msg_id)

            if len(state.accumulated_message_ids) >= int(cfg.max_accumulated_messages):
                forced_state = self._windows.pop(state.session_id, None)
                if forced_state is not None:
                    forced_state.closed = True
            else:
                if state.debounce_task is not None and not state.debounce_task.done():
                    state.debounce_task.cancel()
                state.debounce_task = asyncio.create_task(
                    self._debounce_then_flush(
                        session_id=state.session_id,
                        delay=debounce_delay,
                        bot=bot,
                        msg_mg=msg_mg,
                        cache=cache,
                    )
                )
                scheduled_state = state

        if forced_state is not None:
            await self._cancel_state_tasks(forced_state)
            await self._run_analysis_and_maybe_reply(
                state=forced_state,
                bot=bot,
                msg_mg=msg_mg,
                cache=cache,
                final_analysis=True,
            )
            return

        if scheduled_state is not None:
            logger.debug(
                "registered continuation message: session={} pending={} accumulated={}",
                scheduled_state.session_id,
                len(scheduled_state.pending_message_ids),
                len(scheduled_state.accumulated_message_ids),
            )

    async def _debounce_then_flush(
        self,
        *,
        session_id: str,
        delay: float,
        bot: Bot,
        msg_mg: GroupMessageManager,
        cache: Any,
    ) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            await self._flush_pending_batch(
                session_id=session_id,
                bot=bot,
                msg_mg=msg_mg,
                cache=cache,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("continuation debounce flush failed: session={}", session_id, exc_info=True)

    async def _flush_pending_batch(
        self,
        *,
        session_id: str,
        bot: Bot,
        msg_mg: GroupMessageManager,
        cache: Any,
    ) -> None:
        async with self._lock:
            state = self._windows.get(str(session_id))
            if state is None or state.closed:
                return
            if not state.pending_message_ids:
                return
            pending_snapshot = list(state.pending_message_ids)
            state.pending_message_ids.clear()

        await self._run_analysis_and_maybe_reply(
            state=state,
            bot=bot,
            msg_mg=msg_mg,
            cache=cache,
            final_analysis=False,
            pending_message_ids=pending_snapshot,
        )

    async def _prime_window_with_buffered_messages(
        self,
        *,
        state: ContinuationWindowState,
        bot: Bot,
        msg_mg: GroupMessageManager,
        cache: Any | None,
    ) -> None:
        """将机器人响应期间已到达的用户消息并入新开的续聊窗口。

        这里仅在开窗后补齐“历史上已经存在但尚未参与续聊判断”的消息，
        不会提前在响应进行中触发分析。若补齐后的累计消息已达到上限，会立刻
        执行最后一次分析并关窗；否则按普通窗口消息一样进入防抖流程。

        Args:
            state: 已写入管理器的窗口状态。
            bot: 当前机器人实例，用于排除机器人自身发言。
            msg_mg: 消息管理器。
            cache: 传给主链路的缓存对象，仅在后续触发回复时使用。
        """
        preload_message_ids = await self._collect_buffered_message_ids(
            state=state,
            bot=bot,
            msg_mg=msg_mg,
        )
        if not preload_message_ids:
            return

        cfg = config.chat_model.continuation
        debounce_delay = float(cfg.debounce_seconds)

        async with self._lock:
            current_state = self._windows.get(state.session_id)
            if current_state is None or current_state.closed:
                return

            for message_id in preload_message_ids:
                if message_id in current_state.buffered_message_ids or message_id in current_state.accumulated_message_ids:
                    continue
                current_state.buffered_message_ids.append(message_id)
                current_state.pending_message_ids.append(message_id)

            if not current_state.pending_message_ids:
                return

            if current_state.debounce_task is not None and not current_state.debounce_task.done():
                current_state.debounce_task.cancel()
            current_state.debounce_task = asyncio.create_task(
                self._debounce_then_flush(
                    session_id=current_state.session_id,
                    delay=debounce_delay,
                    bot=bot,
                    msg_mg=msg_mg,
                    cache=cache,
                )
            )

    async def _collect_buffered_message_ids(
        self,
        *,
        state: ContinuationWindowState,
        bot: Bot,
        msg_mg: GroupMessageManager,
    ) -> list[int]:
        """收集响应开始后、开窗前已落库的非机器人消息 ID。

        该方法用于实现“窗口超时从响应结束后算，但聊天记录从响应时开始算”的
        语义。它只按时间边界筛选消息，不负责改动窗口状态，以便调用方统一处理
        防抖与达到上限后的最终分析逻辑。

        Args:
            state: 当前窗口状态，其中 `history_started_at` 为筛选下界。
            bot: 当前机器人实例。
            msg_mg: 消息管理器。

        Returns:
            按时间顺序排列、可并入续聊窗口的消息 ID 列表。
        """
        if state.history_started_at is None:
            return []

        cfg = config.chat_model.continuation
        try:
            recent_messages = await msg_mg.get_recent_messages(
                limit=max(50, int(cfg.max_accumulated_messages) + 20),
                session_id=state.session_id,
                asc_order=True,
            )
        except Exception:
            logger.error(
                "failed to preload continuation messages: session={}",
                state.session_id,
                exc_info=True,
            )
            return []

        buffered_message_ids: list[int] = []
        history_started_at = float(state.history_started_at)
        bot_self_id = int(bot.self_id)
        for item in recent_messages:
            try:
                message_id = int(item.message_id)
                user_id = int(item.user_id)
                send_time = float(item.send_time)
            except Exception:
                continue

            if user_id == bot_self_id or send_time <= history_started_at:
                continue
            buffered_message_ids.append(message_id)
        return buffered_message_ids

    async def _run_analysis_and_maybe_reply(
        self,
        *,
        state: ContinuationWindowState,
        bot: Bot,
        msg_mg: GroupMessageManager,
        cache: Any,
        final_analysis: bool,
        pending_message_ids: list[int] | None = None,
    ) -> None:
        if not state.accumulated_message_ids and not state.buffered_message_ids:
            return

        cumulative_message_ids = list(state.buffered_message_ids) + list(state.accumulated_message_ids)
        if not cumulative_message_ids:
            return
        focus_message_ids = list(pending_message_ids or [])
        if not focus_message_ids:
            focus_message_ids = [cumulative_message_ids[-1]]

        result = await self._analyze_window(
            state=state,
            analysis_message_ids=cumulative_message_ids,
            focus_message_ids=focus_message_ids,
            bot=bot,
            msg_mg=msg_mg,
            cache=cache,
        )
        if result is None:
            return

        logger.info(
            "continuation analysis result: session={} final={} should_reply={} confidence={} reason={} buffered_count={} accumulated_count={} pending_focus_count={}",
            state.session_id,
            final_analysis,
            result.should_reply,
            result.confidence,
            result.reason,
            len(state.buffered_message_ids),
            len(state.accumulated_message_ids),
            len(focus_message_ids),
        )

        if result.should_reply:
            await self.close_window(state.session_id, reason="analyzer_should_reply")
            await self._continue_with_main_chain(
                state=state,
                anchor_message_id=focus_message_ids[-1],
                bot=bot,
                msg_mg=msg_mg,
                cache=cache,
            )
            return

        if final_analysis:
            await self.close_window(state.session_id, reason="final_analysis_completed")

    async def _continue_with_main_chain(
        self,
        *,
        state: ContinuationWindowState,
        anchor_message_id: int,
        bot: Bot,
        msg_mg: GroupMessageManager,
        cache: Any,
    ) -> None:
        try:
            latest_message = await msg_mg.get_message_by_msg_id(int(anchor_message_id))
        except Exception:
            logger.error(
                "failed to load continuation anchor message: session={} anchor_message_id={}",
                state.session_id,
                anchor_message_id,
                exc_info=True,
            )
            return

        from .runtime import ChatTurn, _build_group_session, _build_transport, run_chat_turn
        from ..message.segments import deserialize_message_segments

        turn = ChatTurn(
            session=_build_group_session(state.group_id),
            sender_user_id=int(latest_message.user_id),
            sender_name=str(latest_message.user_name or ""),
            anchor_message_id=int(latest_message.message_id),
            input_text=str(latest_message.content or ""),
            trigger_mode="group_continuation",
            trigger_meta={
                "reason": "continuation",
                "last_response_id": state.last_response_id,
                "source_trigger_mode": state.source_trigger_mode,
                "final_analysis": False,
                "continuation_history_started_at": state.history_started_at,
            },
            source="passive",
            event=None,
            message=deserialize_message_segments(latest_message.serialized_segments),
        )
        transport = _build_transport(
            bot=bot,
            message_manager=msg_mg,
            cache=cache,
            turn=turn,
        )
        await run_chat_turn(turn=turn, transport=transport, bot=bot, msg_mg=msg_mg, cache=cache)

    async def _analyze_window(
        self,
        *,
        state: ContinuationWindowState,
        analysis_message_ids: list[int],
        focus_message_ids: list[int],
        bot: Bot,
        msg_mg: GroupMessageManager,
        cache: Any,
    ) -> ContinuationAnalysisResult | None:
        """执行一次续聊二分类判定，并构造给小模型的完整上下文。

        判定输入分为两部分：一部分是结构化字段，明确本轮触发批次、窗口计数和
        上一条机器人回复；另一部分是复用主聊天链路格式化与清洗逻辑得到的完整
        聊天记录文本，覆盖“响应开始后直到当前判定时刻”为止的全部续聊轨迹。
        其中，响应结束前到达的消息会保留在轨迹中，但不会计入累计上限。

        Args:
            state: 当前窗口状态。
            analysis_message_ids: 截至当前时刻累计的完整续聊消息 ID 列表。
            focus_message_ids: 本次新触发判定的焦点消息 ID 列表。
            bot: 当前机器人实例。
            msg_mg: 消息管理器。
            cache: 用户缓存管理器，用于复用主链路消息格式化能力。

        Returns:
            解析成功时返回结构化判定结果；模型调用失败或输出非法时返回 `None`。
        """
        cfg = config.chat_model.continuation

        try:
            all_recent = await msg_mg.get_recent_messages(
                limit=max(50, int(cfg.max_accumulated_messages) + 20),
                session_id=state.session_id,
            )
        except Exception:
            logger.error("failed to load continuation messages: session={}", state.session_id, exc_info=True)
            return None

        all_recent = list(reversed(all_recent))
        by_message_id = {int(item.message_id): item for item in all_recent}

        conversation_message_ids = state.buffered_message_ids + state.accumulated_message_ids
        conversation_messages = [
            by_message_id[msg_id]
            for msg_id in conversation_message_ids[-max(1, int(cfg.max_accumulated_messages) + int(cfg.max_pending_messages)) :]
            if msg_id in by_message_id
        ]
        analysis_messages = [
            by_message_id[msg_id]
            for msg_id in analysis_message_ids[-int(cfg.max_pending_messages) :]
            if msg_id in by_message_id
        ]
        focus_messages = [
            by_message_id[msg_id]
            for msg_id in focus_message_ids[-int(cfg.max_pending_messages) :]
            if msg_id in by_message_id
        ]

        if not conversation_messages or not analysis_messages:
            return None

        last_bot_message = self._find_last_bot_message(
            recent_messages=all_recent,
            bot_self_id=int(bot.self_id),
            before_message_id=analysis_messages[0].message_id,
            preferred_message_id=state.last_bot_message_id,
        )

        history_messages = self._select_history_messages_for_analyzer(
            recent_messages=all_recent,
            state=state,
            bot_self_id=int(bot.self_id),
        )

        prompt_payload = self._build_prompt_payload(
            state=state,
            last_bot_message=last_bot_message,
            conversation_messages=history_messages,
            analysis_messages=analysis_messages,
            focus_messages=focus_messages,
            final_analysis=bool(len(state.accumulated_message_ids) >= int(cfg.max_accumulated_messages)),
            formatted_history=await self._format_conversation_for_analyzer(
                history_messages=history_messages,
                bot=bot,
                cache=cache,
            ),
            bot_self_id=int(bot.self_id),
        )

        try:
            model = build_chat_model(
                provider_type=cfg.analyzer_provider_type,
                model_id=cfg.analyzer_model_id,
                base_url=cfg.analyzer_base_url,
                api_key=cfg.analyzer_api_key,
                streaming=False,
                model_parameters=dict(cfg.analyzer_parameters),
            )
            response = await model.ainvoke(
                [
                    SystemMessage(content=self._build_system_prompt()),
                    SystemMessage(content=self._build_developer_prompt()),
                    HumanMessage(content=json.dumps(prompt_payload, ensure_ascii=False, indent=2)),
                ]
            )
        except Exception:
            logger.error("continuation analyzer invoke failed: session={}", state.session_id, exc_info=True)
            return None

        raw_output = _normalize_ai_content(getattr(response, "content", ""))
        try:
            payload = self._parse_analyzer_output(raw_output)
        except Exception:
            logger.warning(
                "continuation analyzer returned invalid JSON: session={} output={}",
                state.session_id,
                _truncate_text(raw_output, 400),
            )
            return None

        return ContinuationAnalysisResult(
            should_reply=bool(payload["should_reply"]),
            confidence=payload.get("confidence"),
            reason=str(payload.get("reason") or ""),
            focus_message_ids=[int(item) for item in payload.get("focus_message_ids", []) if str(item).isdigit()],
            raw_output=raw_output,
        )

    def _find_last_bot_message(
        self,
        *,
        recent_messages: list[GroupMessageRecord],
        bot_self_id: int,
        before_message_id: int,
        preferred_message_id: int | None,
    ) -> GroupMessageRecord | None:
        if preferred_message_id:
            for item in recent_messages:
                if int(item.message_id) == int(preferred_message_id):
                    return item

        candidates = [
            item
            for item in recent_messages
            if int(item.user_id) == int(bot_self_id) and int(item.message_id) < int(before_message_id)
        ]
        if not candidates:
            return None
        return candidates[-1]

    def _build_prompt_payload(
        self,
        *,
        state: ContinuationWindowState,
        last_bot_message: GroupMessageRecord | None,
        conversation_messages: list[GroupMessageRecord],
        analysis_messages: list[GroupMessageRecord],
        focus_messages: list[GroupMessageRecord],
        final_analysis: bool,
        formatted_history: str,
        bot_self_id: int,
    ) -> dict[str, Any]:
        """构造续聊判定模型的结构化输入。

        这里会同时提供两层视角：`conversation_messages` 用于让小模型理解
        “上一条机器人回复之后，群里整体是怎么继续发展的”；`analysis_messages`
        则标记本轮真正触发判定的消息批次。响应进行中收到的补充消息会进入
        `conversation_messages` 和可能的首轮 `analysis_messages`，但不会计入
        `max_accumulated_messages` 的关窗阈值。

        Args:
            state: 当前续聊窗口状态。
            last_bot_message: 上一条机器人回复。
            conversation_messages: 供模型理解上下文走向的完整续聊消息序列。
            analysis_messages: 截至当前时刻累计的完整分析消息序列。
            focus_messages: 本次新触发判定的焦点消息序列。
            final_analysis: 当前是否已进入最后一次判定。
            formatted_history: 复用主链路清洗后的完整聊天记录文本。
            bot_self_id: 机器人自身账号 ID，用于标注消息角色。

        Returns:
            传给小模型的 JSON 负载。
        """
        counted_message_ids = set(state.accumulated_message_ids)
        conversation_timeline = [
            {
                "role": "assistant" if int(item.user_id) == int(bot_self_id) else "user",
                "message_id": int(item.message_id),
                "user_id": int(item.user_id),
                "user_name": str(item.user_name or ""),
                "send_time": float(item.send_time),
                "content": _truncate_text(str(item.content or ""), 600 if int(item.user_id) == int(bot_self_id) else 400),
                "counts_towards_limit": int(item.message_id) in counted_message_ids,
            }
            for item in conversation_messages
        ]

        return {
            "session": {
                "session_id": state.session_id,
                "group_id": state.group_id,
                "source_trigger_mode": state.source_trigger_mode,
                "seconds_since_last_response": round(max(0.0, time() - state.opened_at), 3),
                "final_analysis": final_analysis,
                "accumulated_count": len(state.accumulated_message_ids),
                "buffered_count": len(state.buffered_message_ids),
                "analysis_count": len(analysis_messages),
                "focus_count": len(focus_messages),
            },
            "last_bot_reply": None
            if last_bot_message is None
            else {
                "message_id": int(last_bot_message.message_id),
                "content": _truncate_text(str(last_bot_message.content or ""), 600),
            },
            "conversation_messages": conversation_timeline,
            "formatted_chat_history": formatted_history,
            "analysis_messages": [
                {
                    "message_id": int(item.message_id),
                    "user_id": int(item.user_id),
                    "user_name": str(item.user_name or ""),
                    "send_time": float(item.send_time),
                    "content": _truncate_text(str(item.content or ""), 400),
                }
                for item in analysis_messages
            ],
            "focus_messages": [
                {
                    "message_id": int(item.message_id),
                    "user_id": int(item.user_id),
                    "user_name": str(item.user_name or ""),
                    "send_time": float(item.send_time),
                    "content": _truncate_text(str(item.content or ""), 400),
                }
                for item in focus_messages
            ],
        }

    def _select_history_messages_for_analyzer(
        self,
        *,
        recent_messages: list[GroupMessageRecord],
        state: ContinuationWindowState,
        bot_self_id: int,
    ) -> list[GroupMessageRecord]:
        """从消息库中选出供续聊判定使用的完整历史轨迹。

        当续聊已经跨越多个机器人回合时，不能只保留当前窗口内的用户消息，否则
        第二轮及之后的小模型判定会丢掉前序上下文。这里会优先根据窗口继承下来的
        `history_started_at` 回放从该时刻开始的完整消息序列；若无法确定起点，再退化
        为“上一条机器人回复 + 当前窗口消息”的保守策略。

        Args:
            recent_messages: 当前会话最近消息，需按时间正序排列。
            state: 当前续聊窗口状态。
            bot_self_id: 机器人自身账号 ID。

        Returns:
            供小模型理解多轮续聊上下文的完整消息序列。
        """
        history_started_at = state.history_started_at
        pre_history_messages = max(0, int(getattr(config.chat_model.continuation, "pre_history_messages", 0) or 0))
        max_context_messages = max(1, int(getattr(config.chat_model.continuation, "max_analyzer_context_messages", 40) or 40))
        if history_started_at is not None:
            start_index = 0
            for index, item in enumerate(recent_messages):
                if float(getattr(item, "send_time", 0.0) or 0.0) > float(history_started_at):
                    start_index = max(0, index - pre_history_messages)
                    break
            history_messages = [
                item
                for item in recent_messages[start_index:]
                if pre_history_messages > 0 or float(getattr(item, "send_time", 0.0) or 0.0) > float(history_started_at)
            ]
            if history_messages:
                logger.debug(
                    "selected continuation history: session={} history_started_at={} pre_history_messages={} selected_count={} max_context_messages={}",
                    state.session_id,
                    history_started_at,
                    pre_history_messages,
                    len(history_messages[-max_context_messages:]),
                    max_context_messages,
                )
                return history_messages[-max_context_messages:]

        conversation_message_ids = state.buffered_message_ids + state.accumulated_message_ids
        fallback_messages = [
            item
            for item in recent_messages
            if int(item.message_id) in set(conversation_message_ids)
        ]
        if fallback_messages:
            return fallback_messages[-max_context_messages:]

        return [
            item
            for item in recent_messages
            if int(getattr(item, "user_id", 0) or 0) != int(bot_self_id)
        ][-max_context_messages:]

    async def _format_conversation_for_analyzer(
        self,
        *,
        history_messages: list[GroupMessageRecord],
        bot: Bot,
        cache: Any,
    ) -> str:
        """复用主聊天链路的清洗与格式化逻辑，生成判定模型可读历史文本。

        续聊判定不应该看到一份和主模型完全不同的原始拼接文本，否则容易在 CQ 码、
        用户名展示、消息模板等方面出现理解偏差。因此这里直接调用主链路使用的
        `_format_messages_for_chat_context(...)`，让小模型看到相对完整且经过清洗的
        聊天记录。

        Args:
            history_messages: 从续聊起点累计到当前时刻的完整消息轨迹。
            bot: 当前机器人实例。
            cache: 用户缓存管理器。

        Returns:
            经主链路相同规则清洗后的聊天记录文本；若没有可用消息则返回空字符串。
        """
        timeline_messages = [item.to_domain() for item in history_messages]
        if not timeline_messages:
            return ""

        from .runtime import _format_messages_for_chat_context

        return await _format_messages_for_chat_context(
            messages=timeline_messages,
            self_id=int(bot.self_id),
            bot=bot,
            cache=cache,
        )


    def _build_system_prompt(self) -> str:
        return (
            "你是 GTBot 的群聊续聊判定器。\n"
            "你的唯一任务是：判断在 GTBot 刚刚回复后，面对群内的新消息，GTBot 是否应该继续接话。\n"
            "你的目标是在“不打扰群友闲聊”和“保留自然、有价值的连续对话”之间做出准确判断。\n"
            "【严格格式要求】\n"
            "你必须且只能输出一个合法的 JSON 对象，绝对不允许输出任何 Markdown 标记（如 ```json）、解释说明或额外文本。"
        )



    def _build_developer_prompt(self) -> str:
        return (
            "【核心判定原则】\n"
            "1. 认准对话关系，忽略知识盲区：你的任务是判断“用户是不是在跟机器人继续聊”，而不是判断“你懂不懂用户说的话”。绝不能因为不认识消息中的人名、作品、梗、缩写或黑话而判定不回复。\n"
            "2. 边界保守原则：当明显是群友间的闲聊、与机器人无关的新话题、或纯无意义的起哄时，判定不回复（false）。只要对话关系成立，即使消息很短，也应判定回复（true）。\n\n"
            "【强续聊信号（应判定 true）】\n"
            "- 实体/条件替换：复用上一轮的问题模板，只换了对象。如上一轮问“沃玛”，这轮问“花少北呢”“那国服呢”“iOS呢”。\n"
            "- 纠错与质疑：对机器人的回答提出矛盾或要求核对。如“这不对吧”“你再查查”“现在都2026了”。\n"
            "- 简短追问：如“还有呢”“然后呢”“真的假的”。不要因为字数少就判定为 false。\n\n"
            "【忽略信号（应判定 false）】\n"
            "- 群友互相@或明显在彼此闲聊。\n"
            "- 开启了与 GTBot 刚才参与话题完全无关的新话题。\n"
            "- 纯表情、语气词，且无法推动上一轮话题继续。\n\n"
            "【输入上下文说明】\n"
            "- 优先基于 `formatted_chat_history` 和 `conversation_messages` 的连续性做判断。\n"
            "- `conversation_messages` 中 `role=\"assistant\"` 表示 GTBot 发出的消息，`role=\"user\"` 表示群成员发出的消息。判断前必须先分清是谁说的，再判断是否属于续聊。\n"
            "- 不要把用户的复述、吐槽、纠错、质疑误认为是 GTBot 自己的新回复。消息发送者身份以 `conversation_messages.role` 为准，`formatted_chat_history` 只用于辅助理解完整上下文。\n"
            "- `focus_messages` 是本次判定的焦点。即使它很短，也要结合历史理解。\n"
            "- 如果 `final_analysis=true`，代表这是最后一次观察，标准不变，不要放宽或收紧。\n\n"
            "【输出 JSON 结构】\n"
            "必须严格按照以下字段顺序输出 JSON（先输出 analysis 进行思考，再输出结果）：\n"
            "{\n"
            "  \"analysis\": \"简短的一句话分析：新消息是否指向机器人？是否是上轮话题的延续或纠错？\",\n"
            "  \"reason\": \"归类原因，如：entity_replacement, correction, unrelated_chat, short_followup\",\n"
            "  \"confidence\": 0.85,\n"
            "  \"focus_message_ids\": [123, 124],\n"
            "  \"should_reply\": true\n"
            "}\n"
            "注意：should_reply 必须是布尔值且放在最后。"
        )

    def _parse_analyzer_output(self, raw_output: str) -> dict[str, Any]:
        normalized = str(raw_output or "").strip()
        if not normalized:
            raise ValueError("empty analyzer output")

        start = normalized.find("{")
        end = normalized.rfind("}")
        if start >= 0 and end > start:
            normalized = normalized[start : end + 1]

        payload = json.loads(normalized)
        if not isinstance(payload, dict):
            raise ValueError("analyzer output is not a JSON object")
        if "should_reply" not in payload:
            raise ValueError("missing should_reply")

        confidence = payload.get("confidence")
        if confidence is not None:
            payload["confidence"] = max(0.0, min(1.0, float(confidence)))
        if "focus_message_ids" in payload and not isinstance(payload["focus_message_ids"], list):
            payload["focus_message_ids"] = []
        return payload

    async def _expire_window_after_timeout(self, session_id: str, expected_expires_at: float) -> None:
        try:
            now = time()
            delay = max(0.0, float(expected_expires_at) - now)
            if delay > 0:
                await asyncio.sleep(delay)
            await self.close_window(session_id, reason="timeout")
        except asyncio.CancelledError:
            raise

    async def _pop_window(self, session_id: str, *, reason: str) -> ContinuationWindowState | None:
        async with self._lock:
            state = self._windows.pop(str(session_id), None)
            if state is not None:
                state.closed = True
                logger.debug(
                    "closed continuation window: session={} group_id={} reason={}",
                    state.session_id,
                    state.group_id,
                    reason,
                )
            return state

    async def _cancel_state_tasks(self, state: ContinuationWindowState | None) -> None:
        if state is None:
            return

        current_task = asyncio.current_task()
        for task in (state.debounce_task, state.expiry_task):
            if task is None or task.done() or task is current_task:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("continuation background task raised during cancellation", exc_info=True)


_continuation_manager = ContinuationManager()


def get_continuation_manager() -> ContinuationManager:
    return _continuation_manager
