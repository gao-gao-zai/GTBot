from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_CONFIG_PATH = REPO_ROOT / "plugins" / "GTBot" / "config" / "api_config.json"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers and return the numeric result as plain text."""

    return str(a + b)


@tool
def multiply_numbers(a: float, b: float) -> str:
    """Multiply two numbers and return the numeric result as plain text."""

    return str(a * b)


@tool("tavily_search_like")
def tavily_search_like(query: str) -> str:
    """Search-like demo tool that returns a tiny fake news digest for testing tool chains."""

    return json.dumps(
        {
            "query": query,
            "results": [
                {
                    "title": "Demo headline A",
                    "url": "https://example.com/a",
                    "snippet": "A short fake headline used to verify agent tool-calling behavior.",
                },
                {
                    "title": "Demo headline B",
                    "url": "https://example.com/b",
                    "snippet": "Another fake result so the model has something to summarize.",
                },
            ],
        },
        ensure_ascii=False,
    )


SCENARIO_DEFAULT_PROMPTS: dict[str, str] = {
    "math_chain": (
        "You must use tools exactly twice. "
        "First call add_numbers with a=2 and b=3. "
        "Then call multiply_numbers with a=<result_of_first_tool> and b=4. "
        "Finally answer with exactly one sentence containing the final number."
    ),
    "content_then_tool": (
        "You must follow this exact behavior. "
        "Step 1: first output a short natural-language progress sentence to the user saying you are about to search. "
        "Step 2: after that, in the same agent run, call the tool tavily_search_like exactly once with query='最近热点新闻'. "
        "Step 3: after the tool returns, answer with one short sentence summarizing the result. "
        "Do not stop after the progress sentence."
    ),
    "gtbot_tag_then_tool": (
        "You must follow this exact behavior. "
        "Step 1: first output exactly this tagged content and nothing else before the tool call: "
        "<thinking>我要先接住用户，然后继续调用搜索工具</thinking>"
        "<msg>收到啦，我这就去查最新热点新闻</msg>"
        "<note>我准备开始搜索最近热点新闻</note> "
        "Step 2: after outputting that tagged content, in the same agent run, call the tool tavily_search_like exactly once with query='最近热点新闻'. "
        "Step 3: after the tool returns, output one final tagged answer using <msg> and <note>. "
        "Do not stop after the first tagged content block."
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal repro for LangChain create_agent + ChatOpenAI against "
            "Alibaba DashScope OpenAI-compatible endpoint."
        )
    )
    parser.add_argument("--provider", default="aly", help="Provider alias in plugins/GTBot/config/api_config.json")
    parser.add_argument("--model", default="qwen3.5-plus", help="Model key under the provider alias")
    parser.add_argument("--api-config", default=str(DEFAULT_API_CONFIG_PATH), help="Path to api_config.json")
    parser.add_argument("--base-url", default=None, help="Override OpenAI-compatible base_url")
    parser.add_argument("--api-key", default=None, help="Override API key")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--stream", action="store_true", help="Enable model streaming")
    parser.add_argument("--verbose-messages", action="store_true", help="Print full message content")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_DEFAULT_PROMPTS),
        default="math_chain",
        help="Which minimal repro scenario to run",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt sent to the agent; defaults to the selected scenario prompt",
    )
    return parser.parse_args()


def _load_provider_model_config(
    api_config_path: Path,
    provider_alias: str,
    model_key: str,
) -> tuple[str, str, str, dict[str, Any]]:
    payload = json.loads(api_config_path.read_text(encoding="utf-8"))
    provider = payload[provider_alias]
    model_config = provider["llm_models"][model_key]
    return (
        str(provider.get("api_key", "")),
        str(provider.get("base_url", "")),
        str(model_config.get("model", model_key)),
        dict(model_config.get("parameters", {})),
    )


def _sanitize_model_parameters(model_parameters: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(model_parameters)
    for key in (
        "stream",
        "streaming",
        "process_tool_call_deltas",
        "stream_chunk_chars",
        "stream_flush_interval_sec",
    ):
        cleaned.pop(key, None)
    return cleaned


def _dump_message(message: BaseMessage, *, verbose: bool) -> None:
    print(f"[{message.type}]")

    if isinstance(message, AIMessage):
        print(f"tool_calls={json.dumps(getattr(message, 'tool_calls', []), ensure_ascii=False, default=str)}")
        print(
            "invalid_tool_calls="
            f"{json.dumps(getattr(message, 'invalid_tool_calls', []), ensure_ascii=False, default=str)}"
        )
        print(
            "additional_kwargs="
            f"{json.dumps(getattr(message, 'additional_kwargs', {}), ensure_ascii=False, default=str)}"
        )
        print(
            "response_metadata="
            f"{json.dumps(getattr(message, 'response_metadata', {}), ensure_ascii=False, default=str)}"
        )

    if isinstance(message, ToolMessage):
        print(f"tool_call_id={message.tool_call_id}")
        print(f"name={message.name or message.additional_kwargs.get('name')}")

    content = getattr(message, "content", "")
    if verbose:
        print(f"content={json.dumps(content, ensure_ascii=False, default=str)}")
    else:
        preview = str(content)
        if len(preview) > 300:
            preview = preview[:300] + "...<truncated>"
        print(f"content={preview!r}")
    print()


def _resolve_scenario_payload(scenario: str) -> tuple[list[Any], str]:
    if scenario == "math_chain":
        return [add_numbers, multiply_numbers], SCENARIO_DEFAULT_PROMPTS[scenario]
    if scenario == "content_then_tool":
        return [tavily_search_like], SCENARIO_DEFAULT_PROMPTS[scenario]
    if scenario == "gtbot_tag_then_tool":
        return [tavily_search_like], SCENARIO_DEFAULT_PROMPTS[scenario]
    raise ValueError(f"Unsupported scenario: {scenario}")


def main() -> int:
    args = _parse_args()
    api_config_path = Path(args.api_config).resolve()
    api_key, configured_base_url, model_id, model_parameters = _load_provider_model_config(
        api_config_path=api_config_path,
        provider_alias=args.provider,
        model_key=args.model,
    )
    base_url = str(args.base_url or configured_base_url or DEFAULT_BASE_URL).strip()
    api_key = str(args.api_key or api_key).strip()
    model_parameters = _sanitize_model_parameters(model_parameters)

    if args.temperature is not None:
        model_parameters["temperature"] = args.temperature

    tools, default_prompt = _resolve_scenario_payload(args.scenario)
    prompt = str(args.prompt or default_prompt)

    model = ChatOpenAI(
        model=model_id,
        base_url=base_url,
        api_key=SecretStr(api_key),
        streaming=bool(args.stream),
        extra_body=model_parameters,
    )
    agent = create_agent(model=model, tools=tools)

    print("=== Config ===")
    print(f"provider_alias={args.provider}")
    print(f"model_key={args.model}")
    print(f"model_id={model_id}")
    print(f"scenario={args.scenario}")
    print(f"base_url={base_url}")
    print(f"streaming={bool(args.stream)}")
    print(f"extra_body={json.dumps(model_parameters, ensure_ascii=False, default=str)}")
    print(f"tool_names={[getattr(tool_obj, 'name', str(tool_obj)) for tool_obj in tools]}")
    print(f"prompt={json.dumps(prompt, ensure_ascii=False)}")
    print()

    try:
        response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    except Exception as exc:
        print("=== Exception ===")
        print(repr(exc))
        return 1

    messages = response.get("messages", [])
    print("=== Messages ===")
    for message in messages:
        _dump_message(message, verbose=bool(args.verbose_messages))

    last_ai = next((item for item in reversed(messages) if isinstance(item, AIMessage)), None)
    ai_messages = [item for item in messages if isinstance(item, AIMessage)]
    ai_with_content_and_tools = [
        item
        for item in ai_messages
        if getattr(item, "content", "") and getattr(item, "tool_calls", None)
    ]
    print("=== Summary ===")
    print(f"total_messages={len(messages)}")
    print(f"total_ai_messages={len(ai_messages)}")
    print(f"ai_messages_with_content_and_tool_calls={len(ai_with_content_and_tools)}")
    print(f"last_ai_has_tool_calls={bool(getattr(last_ai, 'tool_calls', None)) if last_ai is not None else False}")
    print(
        "last_ai_invalid_tool_calls="
        f"{len(getattr(last_ai, 'invalid_tool_calls', [])) if last_ai is not None else 0}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
