from __future__ import annotations

import json
import os
from urllib.parse import unquote
from typing import Any

import httpx
from langchain.tools import tool


TAVILY_API_KEY = "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"
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"

@tool("tavily_search")
async def tavily_search(
    query: str,
) -> str:
    """一个调用 Tavily API 的工具，用于搜索互联网上的信息。

    Args:
        query: 一个用自然语言描写的问题。

    Returns:
        JSON 格式的数据（字符串），包含搜索结果的详细信息。
    """
    api_key = TAVILY_API_KEY
    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "include_answer": "basic",
        "search_depth": "advanced",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(TAVILY_SEARCH_ENDPOINT, json=payload)
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
    registry.add_tool(tavily_search)
