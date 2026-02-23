#!/usr/bin/env python3
"""Qdrant 用户画像管理器（QdrantUserProfile）功能/边界测试脚本。

该脚本用于在本机真实环境中对 `QdrantUserProfile` 进行端到端验证，包括：

- 写入：`add_user_profile`
- 单用户检索：`get_user_profiles`
- 跨用户检索：`search_user_profiles`
- 反查归属：`get_owner_by_doc_id`
- 更新：`update_by_doc_id`
- 删除：`delete_by_doc_id`、`delete_all_by_user_id`
- 异常与边界：参数缺失、空输入、长度不匹配、doc_id 不存在等

重要说明（避免 NoneBot 初始化副作用）：
    `plugins/GTBot/__init__.py` 在某些运行方式下可能触发 NoneBot 初始化错误。
    因此本脚本不会使用 `import plugins.GTBot...`，而是通过“按文件路径”
    动态加载 LongMemory 目录下的模块，并构造一个临时包以支持相对导入。

使用方式（推荐用文件路径执行）：

    python plugins/GTBot/services/LongMemory/test_qdrant_user_profile_manager.py \
        --qdrant-url http://localhost:6333 \
        --openai-url http://172.26.226.57:30020/v1/embeddings \
        --openai-model qwen3-embedding:0.6b \
        --openai-key <YOUR_KEY>

可选：

- 使用环境变量提供连接信息：

    - `QDRANT_URL`
    - `QDRANT_API_KEY`（若你的 Qdrant 开启了鉴权）
    - `OPENAI_API_KEY`（若你不想在命令行传 embeddings key）

- 如果你暂时没有 Qdrant 服务可用：

    --skip-qdrant

Args:
    --qdrant-url: Qdrant 服务 URL（例如 http://localhost:6333）。
    --qdrant-api-key: API Key（可选，也可用环境变量 `QDRANT_API_KEY`）。
    --openai-url: OpenAI 兼容 embeddings URL。
    --openai-model: Embedding 模型名。
    --openai-key: API Key（也可用环境变量 `OPENAI_API_KEY`）。
    --skip-network: 跳过联网（将使用离线 Dummy 向量生成器）。
    --collection-name: 指定 collection 名称（默认随机生成，避免污染现有数据）。
    --n-results: 检索返回条数。
    --skip-qdrant: 跳过需要连接 Qdrant 的端到端用例。
    --verbose: 打印更多信息。

Returns:
    进程退出码：0 表示全部通过或被跳过；1 表示存在失败。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


VECTOR_DIM = 16


@dataclass(frozen=True)
class TestResult:
    """单个测试用例结果。

    Attributes:
        name: 用例名称。
        ok: 是否通过。
        error: 失败时异常信息。
    """

    name: str
    ok: bool
    error: str | None


def _print_verbose(verbose: bool, message: str) -> None:
    """按需打印调试信息。"""

    if verbose:
        print(message)


def _load_longmemory_package(longmemory_dir: str) -> str:
    """构造一个临时包，支持 LongMemory 内部相对导入。

    Args:
        longmemory_dir: LongMemory 目录绝对路径。

    Returns:
        str: 临时包名。
    """

    package_name = f"_longmemory_qdrant_testpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [longmemory_dir]  # type: ignore[attr-defined]
    sys.modules[package_name] = pkg
    return package_name


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块。

    Args:
        module_qualname: 模块的全限定名（用于相对导入解析）。
        file_path: 目标 .py 文件路径。

    Returns:
        ModuleType: 已加载的模块对象。

    Raises:
        RuntimeError: 加载失败。
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建 spec: {module_qualname} -> {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


def _assert(condition: bool, message: str) -> None:
    """断言辅助。"""

    if not condition:
        raise AssertionError(message)


async def _expect_raises_async(
    exc_type: type[BaseException],
    fn: Callable[[], Any],
) -> None:
    """断言异步函数必须抛出指定异常类型。"""

    try:
        result = fn()
        if asyncio.iscoroutine(result):
            await result
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"期望抛 {exc_type.__name__}，但抛出了 {type(exc).__name__}: {exc!r}") from exc
    raise AssertionError(f"期望抛 {exc_type.__name__}，但未抛异常")


class _DummyVectorGenerator:
    """离线向量生成器（用于测试）。

    该实现不依赖网络，向量维度固定为 VECTOR_DIM，并且对相同文本生成稳定向量。
    """

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """为单条文本生成向量。"""

        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """为多条文本生成向量。"""

        if not texts:
            raise ValueError("texts 列表不能为空")

        out = np.empty((len(texts), VECTOR_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            # 简单但稳定的 hash：按 utf-8 字节累加并做分桶
            b = t.encode("utf-8", errors="ignore")
            acc = np.zeros((VECTOR_DIM,), dtype=np.float32)
            for j, v in enumerate(b):
                acc[j % VECTOR_DIM] += float(v)
            # 归一化，避免向量全零
            norm = float(np.linalg.norm(acc))
            if norm > 0:
                acc /= norm
            out[i] = acc
        return out

    async def embed(self, text: str | list[str]) -> NDArray[np.float32]:
        """兼容协议：支持单条/批量。"""

        if isinstance(text, str):
            return await self.embed_query(text)
        return await self.embed_documents(text)


async def run_suite(
    *,
    qdrant_url: str,
    qdrant_api_key: str | None,
    openai_url: str,
    openai_model: str,
    openai_key: str,
    collection_name: str,
    n_results: int,
    skip_network: bool,
    verbose: bool,
) -> list[TestResult]:
    """运行完整测试集合。

    Args:
        qdrant_url: Qdrant 服务 URL。
        qdrant_api_key: Qdrant API Key（可选）。
        openai_url: OpenAI 兼容 embeddings URL。
        openai_model: Embedding 模型名。
        openai_key: OpenAI API Key。
        collection_name: collection 名称。
        n_results: 检索返回条数。
        skip_network: 是否跳过联网用例。
        verbose: 是否打印更多信息。

    Returns:
        list[TestResult]: 用例结果列表。
    """

    results: list[TestResult] = []

    # 当前文件位于 LongMemory/test/ 下；需要回到上一级 LongMemory 目录加载模块。
    longmemory_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    pkg = _load_longmemory_package(longmemory_dir)

    _load_module_from_path(f"{pkg}.model", os.path.join(longmemory_dir, "model.py"))
    vec_mod = _load_module_from_path(f"{pkg}.VectorGenerator", os.path.join(longmemory_dir, "VectorGenerator.py"))
    qdrant_mod = _load_module_from_path(
        f"{pkg}.qdrant_user_profile",
        os.path.join(longmemory_dir, "UserProfile.py"),
    )

    QdrantUserProfile = getattr(qdrant_mod, "QdrantUserProfile")

    try:
        from qdrant_client.async_qdrant_client import AsyncQdrantClient
    except Exception as exc:  # noqa: BLE001
        return [TestResult(name="import_qdrant_client", ok=False, error=repr(exc))]

    OpenaiVectorGenerator = getattr(vec_mod, "OpenaiVectorGenerator")

    if skip_network:
        vector_generator: Any = _DummyVectorGenerator()
        vector_size = VECTOR_DIM
    else:
        if not openai_key:
            return [TestResult(name="openai_key_missing", ok=False, error="未提供 openai_key（可用 --openai-key 或环境变量 OPENAI_API_KEY）")]
        vector_generator = OpenaiVectorGenerator(
            model_name=openai_model,
            api_key=openai_key,
            api_url=openai_url,
        )
        probe = await vector_generator.embed_query("dimension probe")
        vector_size = int(getattr(probe, "shape", [0])[0])
        if vector_size <= 0:
            return [TestResult(name="vector_size_probe", ok=False, error=f"探测向量维度失败: {vector_size}")]

    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    try:
        service: Any = await QdrantUserProfile.create(
            collection_name=collection_name,
            client=client,
            vector_generator=cast(Any, vector_generator),
            vector_size=vector_size,
        )

        user1 = 10001
        user2 = 10002

        async def case_add_and_get() -> None:
            await service.add_user_profile(user1, ["喜欢打游戏", "讨厌香菜"])  # type: ignore[arg-type]
            await service.add_user_profile(user2, "喜欢听音乐")  # type: ignore[arg-type]

            p1 = await service.get_user_profiles(user_id=user1, limit=10, sort_by="auto")
            _assert(p1.id == user1, "get_user_profiles 返回 user_id 不一致")
            _assert(len(p1.description) == 2, "user1 写入 2 条应返回 2 条")
            _assert(all(x.doc_id for x in p1.description), "doc_id 不能为空")

            ps = await service.get_user_profiles(user_id=[user1, user2], limit=10, sort_by="auto")
            _assert(isinstance(ps, list), "批量 get_user_profiles 返回类型错误")
            _assert(len(ps) == 2, "批量 get_user_profiles 应返回 2 个用户结果")
            _assert(ps[0].id == user1 and ps[1].id == user2, "批量 get_user_profiles 返回顺序应与输入一致")

            c1 = await service.count_user_profile_descriptions(user1)
            _assert(c1 == 2, "count_user_profile_descriptions(user1) 应为 2")

            cs = await service.count_user_profile_descriptions([user1, user2])
            _assert(isinstance(cs, list) and cs == [2, 1], "批量 count_user_profile_descriptions 返回应为 [2, 1]")

        async def case_text_search_single_user() -> None:
            # 按文本检索 user1：应该至少返回 1 条
            p1 = await service.get_user_profiles(user_id=user1, limit=2, text="打游戏", sort_by="text")
            _assert(len(p1.description) >= 1, "text 检索应返回至少 1 条")

        async def case_search_all_users() -> None:
            hits = await service.search_user_profiles("音乐", n_results=n_results)
            _assert(isinstance(hits, list), "search_user_profiles 返回类型错误")
            if hits:
                _assert(hits[0].doc_id, "hit.doc_id 不能为空")
                _assert(hits[0].user_id in (user1, user2), "hit.user_id 不在预期范围")

        async def case_update_and_owner() -> None:
            p1 = await service.get_user_profiles(user_id=user1, limit=10, sort_by="auto")
            target = p1.description[0]

            updated = await service.update_by_doc_id(target.doc_id, descriptions="喜欢玩桌游")
            _assert(updated == 1, "update_by_doc_id 应更新 1 条")

            owner = await service.get_owner_by_doc_id(target.doc_id, return_type="user_id")
            _assert(owner == user1, "get_owner_by_doc_id 返回 user_id 不一致")

            owner_profile = await service.get_owner_by_doc_id(target.doc_id, return_type="profile")
            _assert(owner_profile.id == user1, "反查 profile.id 不一致")

        async def case_touch_and_delete() -> None:
            p2 = await service.get_user_profiles(user_id=user2, limit=10, sort_by="auto")
            _assert(p2.description, "user2 至少应有 1 条")
            doc_id = p2.description[0].doc_id

            touched = await service.touch_read_time_by_doc_id(doc_id)
            _assert(touched == 1, "touch_read_time_by_doc_id 应更新 1 条")

            deleted = await service.delete_by_doc_id(doc_id)
            _assert(deleted == 1, "delete_by_doc_id 应删除 1 条")

            # 删除 user1 全量
            deleted_all = await service.delete_all_by_user_id(user1)
            _assert(deleted_all >= 1, "delete_all_by_user_id 至少删除 1 条")

        async def case_errors() -> None:
            await _expect_raises_async(ValueError, lambda: service.get_user_profiles(user_id=user1, sort_by="text"))
            await _expect_raises_async(ValueError, lambda: service.update_by_doc_id(["a", "b"], user_id=[1]))
            await _expect_raises_async(ValueError, lambda: service.update_by_doc_id("not-exists", user_id=1))

        cases: list[tuple[str, Callable[[], Any]]] = [
            ("add_and_get", case_add_and_get),
            ("text_search_single_user", case_text_search_single_user),
            ("search_all_users", case_search_all_users),
            ("update_and_owner", case_update_and_owner),
            ("touch_and_delete", case_touch_and_delete),
            ("errors", case_errors),
        ]

        for name, fn in cases:
            try:
                _print_verbose(verbose, f"[RUN] {name}")
                await fn()
                results.append(TestResult(name=name, ok=True, error=None))
            except Exception as exc:  # noqa: BLE001
                results.append(TestResult(name=name, ok=False, error=repr(exc)))

    finally:
        # 清理 collection，避免污染。
        try:
            await client.delete_collection(collection_name=collection_name)
        except Exception:
            pass

        try:
            await client.close()
        except Exception:
            pass

    return results


def main() -> int:
    """脚本入口。

    Returns:
        int: 进程退出码。
    """

    parser = argparse.ArgumentParser(description="QdrantUserProfile 功能/边界测试")
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant 服务 URL",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=os.getenv("QDRANT_API_KEY", ""),
        help="Qdrant API Key（可选）",
    )
    parser.add_argument(
        "--openai-url",
        default=os.getenv("OPENAI_URL", "http://172.26.226.57:30020/v1/embeddings"),
        help="OpenAI 兼容 embeddings URL",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "qwen3-embedding:0.6b"),
        help="Embedding 模型名",
    )
    parser.add_argument(
        "--openai-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API Key（也可用环境变量 OPENAI_API_KEY）",
    )
    parser.add_argument(
        "--collection-name",
        default="",
        help="collection 名称（默认随机生成）",
    )
    parser.add_argument("--n-results", type=int, default=10, help="检索返回条数")
    parser.add_argument("--skip-qdrant", action="store_true", help="跳过需要连接 Qdrant 的用例")
    parser.add_argument("--skip-network", action="store_true", help="跳过联网（使用离线向量生成器）")
    parser.add_argument("--verbose", action="store_true", help="打印更多信息")
    args = parser.parse_args()

    if args.skip_qdrant:
        print("[SKIP] --skip-qdrant 已开启，跳过端到端测试")
        return 0

    qdrant_url = str(args.qdrant_url).strip()
    if not qdrant_url:
        print("[SKIP] qdrant_url 为空，跳过端到端测试")
        return 0

    qdrant_api_key = str(args.qdrant_api_key).strip() or None
    openai_url = str(args.openai_url).strip()
    openai_model = str(args.openai_model).strip()
    openai_key = str(args.openai_key).strip()
    collection_name = str(args.collection_name).strip() or f"user_profile_test_{uuid4().hex}"

    try:
        results = asyncio.run(
            run_suite(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                openai_url=openai_url,
                openai_model=openai_model,
                openai_key=openai_key,
                collection_name=collection_name,
                n_results=int(args.n_results),
                skip_network=bool(args.skip_network),
                verbose=bool(args.verbose),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] 测试脚本运行失败: {exc!r}")
        return 1

    ok_count = sum(1 for r in results if r.ok)
    fail_count = sum(1 for r in results if not r.ok)

    print("=== QdrantUserProfile 测试结果 ===")
    for r in results:
        if r.ok:
            print(f"[OK]   {r.name}")
        else:
            print(f"[FAIL] {r.name}: {r.error}")

    print(f"=== 汇总: ok={ok_count} fail={fail_count} ===")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
