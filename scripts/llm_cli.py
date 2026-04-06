from __future__ import annotations

import argparse
import asyncio
import ast
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any
from typing import cast

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout

    _HAS_PROMPT_TOOLKIT = True
except Exception:  # noqa: BLE001
    PromptSession: Any = None
    EditingMode: Any = None
    KeyBindings: Any = None
    patch_stdout: Any = None
    _HAS_PROMPT_TOOLKIT = False

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 作为脚本直接运行时，sys.path[0] 会指向 scripts/，导致顶层 `plugins` 无法解析。
# 这里显式把项目根目录加入 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CLI 测试默认不触发 LongMemory 模块的全局自动初始化。
os.environ.setdefault("GTBOT_LONGMEMORY_AUTOINIT", "0")


def _try_init_nonebot() -> None:
    """尽力初始化 NoneBot（仅用于 CLI/脚本导入插件模块）。

    背景：
        本仓库的部分模块同时承担“NoneBot 插件”和“可复用库”的角色，
        在导入阶段可能会执行 `get_driver().on_startup` 之类的注册逻辑。
        当 CLI 以普通脚本方式运行时，NoneBot 往往未初始化，直接导入会抛出
        `ValueError: NoneBot has not been initialized.`。

        为了做到“只改 CLI，不改插件”，这里做 best-effort 初始化：
        - 若 driver 已存在：不做任何事。
        - 若 driver 不存在：尝试调用 `nonebot.init()` 创建 driver。

    Returns:
        None: 无返回值。
    """

    try:
        from nonebot import get_driver, init as nonebot_init

        try:
            get_driver()
            return
        except Exception:
            pass

        try:
            nonebot_init()
        except Exception:
            return
    except Exception:
        return


_try_init_nonebot()

from plugins.GTBot.services.chat.context import GroupChatContext
from plugins.GTBot.tools.long_memory.IngestManager import LongMemoryContainer
from plugins.GTBot.tools.long_memory import tool as long_memory_tools

# CLI 场景下显式触发一次 rebuild，确保 Pydantic schema 就绪。
# GroupChatContext 已做运行时降级（避免依赖 NoneBot 初始化）。
GroupChatContext.model_rebuild()


def _format_exception(err: BaseException) -> str:
    """格式化异常信息（尽量给出可定位的上下文）。

    Args:
        err: 捕获到的异常。

    Returns:
        str: 适合在 CLI 打印的异常信息。
    """

    parts: list[str] = []
    parts.append(f"{err.__class__.__module__}.{err.__class__.__name__}: {err!s}")

    # Qdrant Client: qdrant_client.http.exceptions.UnexpectedResponse
    status_code = getattr(err, "status_code", None)
    if status_code is not None:
        parts.append(f"status_code={status_code}")
    url = getattr(err, "url", None)
    if url is not None:
        parts.append(f"url={url}")
    content = getattr(err, "content", None)
    if content is not None:
        try:
            text = content.decode("utf-8", errors="replace") if isinstance(content, (bytes, bytearray)) else str(content)
        except Exception:
            text = str(content)
        text = text.strip().replace("\r\n", "\n")
        if len(text) > 1200:
            text = text[:1200] + "...<truncated>"
        parts.append(f"content=\n{text}")

    # httpx / requests 风格的 response 对象
    resp = getattr(err, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code is not None:
            parts.append(f"response.status_code={code}")
        try:
            resp_text = getattr(resp, "text", None)
            if resp_text:
                resp_text = str(resp_text).strip().replace("\r\n", "\n")
                if len(resp_text) > 1200:
                    resp_text = resp_text[:1200] + "...<truncated>"
                parts.append(f"response.text=\n{resp_text}")
        except Exception:
            pass

    return "\n".join(parts)


def _parse_model_kwargs(raw: str) -> dict[str, Any]:
    """解析 model_kwargs JSON 字符串。

    Args:
        raw: JSON 字符串；为空时返回空字典。

    Returns:
        dict[str, Any]: 解析得到的参数字典。

    Raises:
        ValueError: JSON 格式非法或根对象不是 dict。
    """

    if not raw.strip():
        return {}

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # 兼容：某些 shell 场景下用户更容易传入 Python 字面量（单引号等）。
        # 例如：{'reasoning_effort': 'low'}
        try:
            obj = ast.literal_eval(raw)
        except Exception as e:
            raise ValueError(f"model-kwargs 不是合法 JSON：{e}") from e

    if not isinstance(obj, dict):
        raise ValueError("model-kwargs 必须是 JSON 对象（dict）")

    return obj


def _build_long_memory(args: argparse.Namespace) -> LongMemoryContainer:
    """构造 LongMemoryManager。

    优先使用命令行参数，其次读取环境变量；若都未提供，则回退为模块内默认值。

    Args:
        args: 命令行参数。

    Returns:
        LongMemoryManager: LongMemory 服务管理器。
    """

    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL")
    embed_url = args.embed_url or os.getenv("EMBED_URL")
    embed_model = args.embed_model or os.getenv("EMBED_MODEL")
    embed_key = args.embed_key or os.getenv("EMBED_API_KEY")
    qdrant_key = args.qdrant_key or os.getenv("QDRANT_API_KEY")
    collection = args.qdrant_collection or os.getenv("QDRANT_COLLECTION") or "long_memory"

    if not (qdrant_url and embed_url and embed_model):
        # 回退：使用 LongMemory 模块内默认创建的配置（用于开发环境快速启动）。
        # 注意：该默认值可能是内网地址；建议在 CLI 中显式传参。
        return LongMemoryContainer.create(
            qdrant_server_url="http://127.0.0.1:6333" if not qdrant_url else qdrant_url,
            embed_service_url="http://127.0.0.1:30020/v1/embeddings" if not embed_url else embed_url,
            embed_model="qwen3-embedding-0.6b" if not embed_model else embed_model,
            embed_api_key=embed_key,
            qdrant_api_key=qdrant_key,
            qdrant_collection_name=collection,
        )

    return LongMemoryContainer.create(
        qdrant_server_url=qdrant_url,
        embed_service_url=embed_url,
        embed_model=embed_model,
        embed_api_key=embed_key,
        qdrant_api_key=qdrant_key,
        qdrant_collection_name=collection,
    )


def _build_context(*, long_memory: LongMemoryContainer, group_id: int, user_id: int, session_id: str | None = None) -> GroupChatContext:
    """构造 GroupChatContext（用于 ToolRuntime 注入）。

    CLI 场景下没有 NoneBot 的 bot/event/message 等对象，这里使用 `model_construct`
    绕过运行时校验，仅保留工具实际需要的 `long_memory` 等字段。

    Args:
        long_memory: LongMemory 服务。
        group_id: 群号（用于某些工具的分组语义；不一定会用到）。
        user_id: 用户 ID（用于用户画像相关工具）。
        session_id: 会话 ID（可选，用于事件日志等工具）。

    Returns:
        GroupChatContext: 构造出的上下文对象。
    """

    return GroupChatContext.model_construct(
        bot=None,
        event=None,
        message=None,
        group_id=int(group_id),
        user_id=int(user_id),
        message_id=0,
        session_id=session_id,
        message_manager=None,
        cache=None,
        long_memory=long_memory,
    )


def _build_tools() -> list[Any]:
    """加载要用于 CLI 测试的工具列表。

    Returns:
        list[Any]: LangChain tools 列表（BaseTool）。
    """

    # 说明：启用“所有工具”时不能仅凭 obj.name 来判定工具；因为模块对象（例如 __loader__）
    # 也可能有 name 字段。这里严格筛选 BaseTool，避免把杂项塞进 create_agent 导致崩溃。
    try:
        from langchain_core.tools import BaseTool as _BaseTool
    except Exception:  # noqa: BLE001
        try:
            from langchain.tools import BaseTool as _BaseTool  # type: ignore
        except Exception:  # noqa: BLE001
            _BaseTool = object  # type: ignore[assignment]

    tools: list[Any] = []
    for attr_name in dir(long_memory_tools):
        if attr_name.startswith("__"):
            continue
        obj = getattr(long_memory_tools, attr_name, None)

        if not isinstance(obj, _BaseTool):
            continue

        tool_name = getattr(obj, "name", None)
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue

        tools.append(obj)

    tools.sort(key=lambda t: str(getattr(t, "name", "")))
    return tools


def _print_new_messages(messages: list[BaseMessage], *, start: int, show_tools: bool) -> None:
    """打印本轮新增消息。

    Args:
        messages: 完整消息列表。
        start: 从该索引开始视为“新增”。
        show_tools: 是否输出 ToolMessage。
    """

    def _dump_obj(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            return str(obj)

    new_msgs = messages[start:]
    for msg in new_msgs:
        # 先打印 LLM 发起的工具调用（tool_calls），再打印工具返回（ToolMessage）。
        if show_tools and isinstance(msg, AIMessage):
            tool_calls: Any = getattr(msg, "tool_calls", None)
            if not tool_calls:
                tool_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls")

            # 兼容旧版 function_call
            function_call: Any = None
            if not tool_calls:
                function_call = getattr(msg, "additional_kwargs", {}).get("function_call")

            if tool_calls:
                for tc in list(tool_calls):
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                    # args 可能是 JSON 字符串
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    print(f"[TOOL_CALL] name={name} id={call_id}\n{_dump_obj(args)}\n")
            elif function_call:
                name = function_call.get("name") if isinstance(function_call, dict) else None
                args = function_call.get("arguments") if isinstance(function_call, dict) else None
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                print(f"[TOOL_CALL] name={name}\n{_dump_obj(args)}\n")

        if isinstance(msg, ToolMessage):
            if not show_tools:
                continue
            name = getattr(msg, "name", None) or getattr(msg, "tool", None) or "tool"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"[TOOL:{name}]\n{content}\n")

    # 最终答复：取最后一条 AIMessage.content
    for msg in reversed(new_msgs):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(content)
            break


async def _chat_loop(args: argparse.Namespace) -> int:
    """运行交互式 CLI 聊天循环。

    Args:
        args: 命令行参数。

    Returns:
        int: 进程退出码。
    """

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[FATAL] 缺少 API Key：请使用 --api-key 或环境变量 OPENAI_API_KEY")
        return 2

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        print("[FATAL] 缺少 base_url：请使用 --base-url 或环境变量 OPENAI_BASE_URL")
        return 2

    model_id = args.model or os.getenv("OPENAI_MODEL")
    if not model_id:
        print("[FATAL] 缺少模型名：请使用 --model 或环境变量 OPENAI_MODEL")
        return 2

    model_kwargs = _parse_model_kwargs(args.model_kwargs)

    long_memory = _build_long_memory(args)
    session_id = args.session_id.strip() if args.session_id else None
    context = _build_context(long_memory=long_memory, group_id=args.group_id, user_id=args.user_id, session_id=session_id)

    # 注入 LongMemory 与会话 ID（供 LongMemory tools 使用）。
    # LongMemory 工具会通过 ToolRuntime.context 获取 long_memory 与会话信息，无需额外注入。

    # 预检：尽早暴露 Qdrant URL/端口错误（例如填成 127.0.0.1 但没写 6333 端口）。
    # 不阻止 CLI 启动，但会输出明确提示，避免用户在工具调用时才看到 "UnexpectedResponse"。
    try:
        await asyncio.wait_for(long_memory.user_profile_manager.client.get_collections(), timeout=3.0)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Qdrant 连接预检失败：{e!r}")
        print(
            "[HINT] 请确认 Qdrant 可访问，例如：--qdrant-url http://127.0.0.1:6333 （默认端口通常是 6333）。"
        )

    tools = _build_tools()

    middleware: list[Any] = []
    if args.max_tool_calls_per_turn > 0:
        middleware.append(
            ToolCallLimitMiddleware(
                run_limit=int(args.max_tool_calls_per_turn),
                exit_behavior="continue",
            )
        )

    # LangChain 提示：reasoning_effort 若是显式参数则不要塞进 model_kwargs。
    # 这里采用“先尝试显式参数，失败则回退”的方式：
    # - 能消除 warning 的环境会走显式参数
    # - 不支持 reasoning_effort 的环境仍可通过 model_kwargs 传递
    chatopenai_kwargs: dict[str, Any] = {
        "model": model_id,
        "base_url": base_url,
        "api_key": SecretStr(api_key),
        "model_kwargs": model_kwargs,
    }

    if args.reasoning_effort:
        model_kwargs.pop("reasoning_effort", None)
        try:
            model = ChatOpenAI(**chatopenai_kwargs, reasoning_effort=args.reasoning_effort)
        except Exception:
            model_kwargs["reasoning_effort"] = args.reasoning_effort
            model = ChatOpenAI(**chatopenai_kwargs)
    else:
        model = ChatOpenAI(**chatopenai_kwargs)

    agent = create_agent(
        model=model,
        tools=tools,
        context_schema=GroupChatContext,
        middleware=middleware,
    )

    history: list[BaseMessage] = []
    if args.system:
        history.append(SystemMessage(content=args.system))

    print("CLI 已启动。输入文本开始对话；输入 /exit 退出；/reset 清空上下文；/tools 查看工具列表。")

    session: Any | None = None
    if args.input_mode != "basic":
        if not _HAS_PROMPT_TOOLKIT:
            print("[WARN] 未安装 prompt_toolkit，无法启用输入编辑模式；已回退到基础输入。")
            print("[HINT] 可执行：pip install prompt_toolkit")
        else:
            # simple 模式：尽量贴近“普通输入框”体验（非 Vim 模态）。
            if args.input_mode == "simple":
                editing_mode = EditingMode.EMACS
            else:
                editing_mode = EditingMode.VI if args.input_mode == "vi" else EditingMode.EMACS

            key_bindings = KeyBindings()

            def _accept(event: Any) -> None:  # noqa: ANN001
                event.app.current_buffer.validate_and_handle()

            # 说明：多数终端无法区分 Ctrl+Enter 与 Enter；prompt_toolkit 也不支持 "c-enter"。
            # 在 VS Code 终端里 Ctrl+J 常被占用（隐藏终端），因此这里使用：
            # - Alt+Enter（等价于 escape+enter）发送
            # - F2 发送（备用）
            @key_bindings.add("escape", "enter")
            def _accept_alt_enter(event: Any) -> None:  # noqa: ANN001
                _accept(event)

            @key_bindings.add("f2")
            def _accept_f2(event: Any) -> None:  # noqa: ANN001
                _accept(event)

            session = PromptSession(
                editing_mode=editing_mode,
                multiline=True,
                key_bindings=key_bindings,
            )

    async def _read_user_input() -> str:
        """读取一行用户输入（支持 prompt_toolkit 的 vi/emacs 模式）。

        Returns:
            str: 去除首尾空白后的用户输入。
        """

        if session is None:
            text = await asyncio.to_thread(input, "> ")
            return str(text).strip()

        # prompt_toolkit 必须使用异步接口，否则会在运行中的事件循环里调用 asyncio.run()。
        with patch_stdout():
            text = await session.prompt_async("> ")
        return str(text).strip()

    exit_code = 0
    try:
        while True:
            try:
                user_input = await _read_user_input()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                exit_code = 0
                break

            if not user_input:
                continue

            if user_input in {"/exit", "/quit", "exit", "quit"}:
                print("退出。")
                exit_code = 0
                break

            if user_input == "/reset":
                history = [SystemMessage(content=args.system)] if args.system else []
                print("已清空上下文。")
                continue

            if user_input == "/tools":
                names = [getattr(t, "name", str(t)) for t in tools]
                print("已加载工具：")
                for n in names:
                    print(f"- {n}")
                continue

            history.append(HumanMessage(content=user_input))
            before_len = len(history)

            try:
                resp = await agent.ainvoke(
                    input=cast(Any, {"messages": history}),
                    context=context,
                )
            except Exception as e:
                print("[ERROR] 调用失败：")
                print(_format_exception(e))
                continue

            out_msgs: list[BaseMessage] = resp.get("messages", [])
            if not out_msgs:
                print("[WARN] 未收到 messages 输出。")
                continue

            # create_agent 返回的 messages 是“全量状态”，直接替换本地 history。
            history = out_msgs

            _print_new_messages(history, start=before_len, show_tools=not args.hide_tools)
    finally:
        # 退出前显式关闭资源：
        # - aiosqlite 连接若不 close，后台线程可能导致进程退出卡住。
        # - Qdrant Async client 也尽量关闭（best-effort）。
        try:
            await long_memory.group_profile_manager.close()
        except Exception:
            pass

        try:
            client = getattr(long_memory.user_profile_manager, "client", None)
            close_fn = getattr(client, "close", None) if client is not None else None
            if callable(close_fn):
                res = close_fn()
                if inspect.isawaitable(res):
                    await res
        except Exception:
            pass

    return int(exit_code)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 解析器对象。
    """

    parser = argparse.ArgumentParser(description="用于测试 LangChain tools 的简单 LLM 对话 CLI")

    parser.add_argument("--model", type=str, default="", help="模型名（也可用环境变量 OPENAI_MODEL）")
    parser.add_argument("--base-url", type=str, default="", help="OpenAI 兼容接口 base_url（也可用 OPENAI_BASE_URL）")
    parser.add_argument("--api-key", type=str, default="", help="API Key（也可用 OPENAI_API_KEY）")
    parser.add_argument("--model-kwargs", type=str, default="{}", help='额外模型参数 JSON，例如: {"temperature": 0.2}')
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="",
        choices=["low", "medium", "high"],
        help="快捷设置模型参数 reasoning_effort（避免 JSON 引号问题）",
    )

    parser.add_argument("--system", type=str, default="", help="system 提示词（可选）")
    parser.add_argument("--group-id", type=int, default=10000, help="注入到上下文的群号")
    parser.add_argument("--user-id", type=int, default=123456, help="注入到上下文的用户 ID")
    parser.add_argument("--session-id", type=str, default="", help="会话 ID（例如 group_123 或 private_456）；不指定则从 group_id/user_id 推断")

    parser.add_argument(
        "--input-mode",
        type=str,
        default="simple",
        choices=["basic", "simple", "vi", "emacs"],
        help=(
            "输入编辑模式：basic=原生 input；simple=方向键编辑+多行（Alt+Enter 或 F2 发送）；"
            "vi=Vim 风格；emacs=Emacs 风格（需要 prompt_toolkit）"
        ),
    )

    parser.add_argument("--max-tool-calls-per-turn", type=int, default=10, help="单回合最大工具调用次数；0 表示不限制")
    parser.add_argument("--hide-tools", action="store_true", help="不打印工具输出（ToolMessage）")

    parser.add_argument("--qdrant-url", type=str, default="", help="Qdrant URL（也可用 QDRANT_URL）")
    parser.add_argument("--qdrant-key", type=str, default="", help="Qdrant API Key（也可用 QDRANT_API_KEY）")
    parser.add_argument("--qdrant-collection", type=str, default="", help="Qdrant collection（也可用 QDRANT_COLLECTION）")

    parser.add_argument("--embed-url", type=str, default="", help="Embedding URL（OpenAI 兼容 embeddings；也可用 EMBED_URL）")
    parser.add_argument("--embed-model", type=str, default="", help="Embedding 模型名（也可用 EMBED_MODEL）")
    parser.add_argument("--embed-key", type=str, default="", help="Embedding API Key（也可用 EMBED_API_KEY）")

    return parser


def main() -> int:
    """CLI 入口。

    Returns:
        int: 退出码。
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        return asyncio.run(_chat_loop(args))
    except ValueError as e:
        print(f"[FATAL] 参数错误：{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
