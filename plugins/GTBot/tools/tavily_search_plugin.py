from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
from langchain.tools import tool
from pydantic import BaseModel, Field

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


class TavilySearchPluginConfig(BaseModel):
    """Tavily 搜索插件配置。"""

    enabled: bool = False
    api_key: str = ""
    search_endpoint: str = "https://api.tavily.com/search"
    timeout_sec: float = Field(default=20.0, ge=1.0, le=120.0)
    include_answer: str = "basic"
    search_depth: str = "advanced"


_config_cache: TavilySearchPluginConfig | None = None


def _config_path() -> Path:
    """返回 Tavily 插件本地配置文件路径。"""

    return Path(__file__).with_name("tavily_search_plugin.config.json")


def _example_path() -> Path:
    """返回 Tavily 插件示例配置文件路径。"""

    return Path(__file__).with_name("tavily_search_plugin.config.json.example")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """安全写入 JSON 配置文件。

    Args:
        path: 目标配置文件路径。
        data: 需要写入的 JSON 数据。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _default_config() -> TavilySearchPluginConfig:
    """构造 Tavily 插件默认配置。"""

    return TavilySearchPluginConfig()


def _ensure_default_files() -> TavilySearchPluginConfig:
    """确保示例配置与本地配置文件存在。"""

    cfg = _default_config()
    payload = cfg.model_dump(mode="json")
    config_path = _config_path()
    example_path = _example_path()

    if not example_path.exists():
        _write_json(example_path, payload)
    if not config_path.exists():
        _write_json(config_path, payload)
    return cfg


def get_tavily_search_plugin_config() -> TavilySearchPluginConfig:
    """读取 Tavily 搜索插件配置。

    Returns:
        TavilySearchPluginConfig: 当前可用的配置对象。
    """

    global _config_cache
    if _config_cache is not None:
        return _config_cache

    default_cfg = _ensure_default_files()
    path = _config_path()
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        if not isinstance(parsed, dict):
            raise TypeError("tavily_search_plugin config must be a JSON object")
        _config_cache = TavilySearchPluginConfig.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"tavily_search_plugin config parse failed, fallback to defaults: {exc!s}")
        _config_cache = default_cfg
        _write_json(path, _config_cache.model_dump(mode="json"))
    return _config_cache


@tool("tavily_search")
async def tavily_search(query: str) -> str:
    """调用 Tavily API 搜索互联网信息。

    Args:
        query: 一个用自然语言描述的问题。

    Returns:
        str: JSON 格式的字符串，包含精简后的搜索结果。
    """

    cfg = get_tavily_search_plugin_config()
    if not cfg.enabled:
        raise RuntimeError("tavily_search 插件未启用，请在 tavily_search_plugin.config.json 中开启 enabled")
    if not str(cfg.api_key).strip():
        raise RuntimeError("tavily_search 插件缺少 api_key，请在 tavily_search_plugin.config.json 中配置")

    payload: dict[str, Any] = {
        "api_key": cfg.api_key,
        "query": query,
        "include_answer": cfg.include_answer,
        "search_depth": cfg.search_depth,
    }

    async with httpx.AsyncClient(timeout=cfg.timeout_sec) as client:
        resp = await client.post(cfg.search_endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()

    data.pop("request_id", None)

    results = data.get("results")
    if isinstance(results, list):
        cleaned_results: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            cleaned = dict(item)
            cleaned.pop("content", None)
            cleaned.pop("score", None)
            cleaned.pop("raw_content", None)
            url = cleaned.get("url")
            if isinstance(url, str) and url:
                cleaned["url"] = unquote(url)
            cleaned_results.append(cleaned)
        data["results"] = cleaned_results

    return json.dumps(data, ensure_ascii=False)


def register(registry) -> None:  # noqa: ANN001
    """向插件注册表注册 Tavily 搜索工具。

    Args:
        registry: 当前插件系统传入的注册表对象。
    """

    registry.add_tool(tavily_search)
