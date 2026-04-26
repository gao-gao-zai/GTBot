from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GTBOT_LONGMEMORY_AUTOINIT", "0")


def _load_longmemory_package(longmemory_dir: str) -> str:
    """构造临时包名，以便按文件加载 LongMemory 子模块。

    这样可以绕过 `plugins.GTBot.tools.long_memory.__init__` 的插件注册逻辑，
    让单测只聚焦在 `RecallManager.py` 的 rerank 协议适配，而不依赖 NoneBot
    是否已经初始化。

    Args:
        longmemory_dir: LongMemory 目录的绝对路径。

    Returns:
        构造出的临时包名，可用于后续相对导入。
    """
    package_name = "_longmemory_rerank_unit_testpkg"
    pkg = ModuleType(package_name)
    pkg.__path__ = [longmemory_dir]  # type: ignore[attr-defined]
    sys.modules[package_name] = pkg
    return package_name


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块，用于隔离单测依赖。

    Args:
        module_qualname: 注册到 `sys.modules` 的模块全名。
        file_path: 模块文件绝对路径。

    Returns:
        已经执行并注册完成的模块对象。

    Raises:
        RuntimeError: 当无法为目标文件创建模块 spec 时抛出。
    """
    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建 spec: {module_qualname} -> {file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_reranker_cls() -> type[Any]:
    """加载仅包含 RecallManager 依赖的临时测试包。

    Returns:
        `RecallManager.py` 中定义的 `TEIReranker` 类。
    """
    longmemory_dir = str(Path(__file__).resolve().parents[1] / "tools" / "long_memory")
    pkg = _load_longmemory_package(longmemory_dir)
    _load_module_from_path(f"{pkg}.MappingManager", str(Path(longmemory_dir) / "MappingManager.py"))
    recall_mod = _load_module_from_path(f"{pkg}.RecallManager", str(Path(longmemory_dir) / "RecallManager.py"))
    reranker_cls = getattr(recall_mod, "TEIReranker", None)
    if reranker_cls is None:
        raise RuntimeError("无法加载 TEIReranker")
    return cast(type[Any], reranker_cls)


class _FakeResponse:
    def __init__(self, *, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def json(self) -> Any:
        return self._payload


class _FakeClientSession:
    """模拟 aiohttp ClientSession，用于验证 rerank 请求协议。

    该桩对象只覆盖当前测试关注的行为：记录 `post` 调用参数并返回预设响应。
    这样可以在不发出真实网络请求的前提下，断言请求体字段是否已经切换到
    OpenAI 风格的 `model/query/documents` 协议。
    """

    def __init__(self, *, timeout: Any = None) -> None:
        self.timeout = timeout
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_FakeClientSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> _FakeResponse:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        return _FakeResponse(
            status=200,
            payload={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.52},
                ]
            },
        )


def test_reranker_uses_openai_style_request_and_response() -> None:
    captured: dict[str, Any] = {}
    reranker_cls = _load_reranker_cls()

    def _session_factory(*, timeout: Any = None) -> _FakeClientSession:
        session = _FakeClientSession(timeout=timeout)
        captured["session"] = session
        return session

    reranker = reranker_cls(
        api_url="http://localhost:4005/v1/rerank",
        model_name="bge-reranker-v2-m3",
        api_key="test-key",
    )

    with patch.object(sys.modules[reranker_cls.__module__].aiohttp, "ClientSession", new=_session_factory):
        result = asyncio.run(
            reranker.rerank(
                query="数据库连接池怎么优化",
                texts=["文档A", "文档B"],
            )
        )

    assert result == [
        {"index": 1, "score": 0.91},
        {"index": 0, "score": 0.52},
    ]

    session = captured["session"]
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "http://localhost:4005/v1/rerank"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"] == {
        "model": "bge-reranker-v2-m3",
        "query": "数据库连接池怎么优化",
        "documents": ["文档A", "文档B"],
    }
