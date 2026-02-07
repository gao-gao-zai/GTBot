#!/usr/bin/env python3
"""用户画像管理器（QdrantUserProfile）性能测试脚本。

该脚本用于对 LongMemory 的 Qdrant 后端实现 `QdrantUserProfile` 进行端到端性能基准测试，覆盖：

- 向量生成：`OpenaiVectorGenerator.embed_documents`（可选）
- 写入：`QdrantUserProfile.add_user_profile`
- 单用户检索：`QdrantUserProfile.get_user_profiles`（metadata 排序、text 相似度检索）
- 跨用户检索：`QdrantUserProfile.search_user_profiles`
- 更新：`QdrantUserProfile.update_by_doc_id`
- 删除：`QdrantUserProfile.delete_by_doc_id`、`QdrantUserProfile.delete_all_by_user_id`

与功能测试不同，本脚本关注吞吐与延迟分布（mean/p50/p95 等）。

重要说明（避免 NoneBot 初始化副作用）：
    本脚本不会 `import plugins.GTBot...`，而是通过“按文件路径”动态加载 LongMemory
    目录下的模块，并构造临时包以支持相对导入。

示例（使用真实 embeddings + 本地 Qdrant）：

    python plugins/GTBot/services/LongMemory/perf_qdrant_user_profile_manager.py \
        --qdrant-url http://localhost:6333 \
        --openai-url http://172.26.226.57:30020/v1/embeddings \
        --openai-model qwen3-embedding:0.6b \
        --openai-key dummy_key \
        --users 50 \
        --descs-per-user 5 \
        --concurrency 10

示例（跳过联网：仅压测 Qdrant CRUD + query_points，不调用 embeddings）：

    python plugins/GTBot/services/LongMemory/perf_qdrant_user_profile_manager.py \
        --qdrant-url http://localhost:6333 \
        --skip-network \
        --vector-dim 1024

Args:
    --qdrant-url: Qdrant 服务 URL（例如 http://localhost:6333）。
    --qdrant-api-key: API Key（可选，也可用环境变量 `QDRANT_API_KEY`）。
    --openai-url: OpenAI 兼容 embeddings 接口 URL。
    --openai-model: Embedding 模型名。
    --openai-key: API Key（也可用环境变量 `OPENAI_API_KEY`）。
    --skip-network: 跳过联网（将使用离线 Dummy 向量生成器）。
    --vector-dim: `--skip-network` 时使用的向量维度。
    --collection-name: 指定集合名（默认随机生成）。
    --users: 生成多少个用户。
    --descs-per-user: 每个用户写入多少条描述。
    --embed-batch: 单次向量生成压测的 batch 大小。
    --queries: 压测查询条数（用于 text 检索与跨用户检索）。
    --n-results: 向量检索每次返回条数。
    --concurrency: 并发度（写入/检索/更新/删除会使用 semaphore 控制）。
    --warmup: 每个子测试预热次数。
    --repeats: 每个子测试重复次数（用于统计分布）。
    --skip-delete: 跳过删除阶段（便于复用数据）。
    --skip-cleanup: 跳过 collection 清理（便于复用数据）。

Returns:
    进程退出码：0 表示脚本运行成功；1 表示参数或运行错误。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray


@dataclass
class Metric:
    """单项指标统计。

    Attributes:
        name: 指标名称。
        unit: 单位（例如 ms、ops/s）。
        values: 采样值列表。
    """

    name: str
    unit: str
    values: list[float]


def _now() -> float:
    """返回高精度计时器时间戳。

    Returns:
        高精度时间戳。
    """

    return time.perf_counter()


def _ms(seconds: float) -> float:
    """秒转毫秒。

    Args:
        seconds: 秒。

    Returns:
        毫秒。
    """

    return seconds * 1000.0


def _percentile(sorted_values: list[float], p: float) -> float:
    """计算分位数（0~100），输入必须已排序。

    Args:
        sorted_values: 已排序的值。
        p: 分位数百分比（0~100）。

    Returns:
        分位数值。
    """

    if not sorted_values:
        return float("nan")

    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])

    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _summarize(values: list[float]) -> dict[str, float]:
    """汇总统计信息。

    Args:
        values: 采样值。

    Returns:
        汇总信息。
    """

    if not values:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }

    sorted_values = sorted(values)
    return {
        "n": float(len(values)),
        "mean": float(statistics.mean(values)),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "p50": _percentile(sorted_values, 50.0),
        "p95": _percentile(sorted_values, 95.0),
    }


def _print_table(metrics: list[Metric]) -> None:
    """打印指标表。

    Args:
        metrics: 指标列表。
    """

    headers = ["name", "unit", "n", "mean", "min", "p50", "p95", "max"]
    rows: list[list[str]] = [headers]

    for m in metrics:
        s = _summarize(m.values)
        rows.append(
            [
                m.name,
                m.unit,
                str(int(s["n"])) if s["n"] == s["n"] else "-",
                f"{s['mean']:.3f}" if s["mean"] == s["mean"] else "-",
                f"{s['min']:.3f}" if s["min"] == s["min"] else "-",
                f"{s['p50']:.3f}" if s["p50"] == s["p50"] else "-",
                f"{s['p95']:.3f}" if s["p95"] == s["p95"] else "-",
                f"{s['max']:.3f}" if s["max"] == s["max"] else "-",
            ]
        )

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return "  ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

    print(fmt(rows[0]))
    print("  ".join("-" * w for w in widths))
    for r in rows[1:]:
        print(fmt(r))


def _load_longmemory_package(longmemory_dir: str) -> str:
    """构造一个临时包，支持 LongMemory 内部相对导入。

    Args:
        longmemory_dir: LongMemory 目录绝对路径。

    Returns:
        临时包名。
    """

    package_name = f"_longmemory_qdrant_perf_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [longmemory_dir]  # type: ignore[attr-defined]
    sys.modules[package_name] = pkg
    return package_name


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块。

    Args:
        module_qualname: 模块的全限定名。
        file_path: 目标 .py 文件路径。

    Returns:
        已加载的模块对象。

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


async def _run_concurrent(
    tasks: list[Callable[[], Any]],
    *,
    concurrency: int,
) -> None:
    """以固定并发度执行一组异步任务。

    Args:
        tasks: 任务（函数）列表。
        concurrency: 并发度。
    """

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def run_one(fn: Callable[[], Any]) -> None:
        async with sem:
            result = fn()
            if asyncio.iscoroutine(result):
                await result

    await asyncio.gather(*(run_one(fn) for fn in tasks))


class _DummyVectorGenerator:
    """离线向量生成器（用于跳过联网时的压测）。

    说明：
        - 维度固定为 `dim`。
        - 对相同文本生成稳定向量（便于重复性）。

    Args:
        dim: 向量维度。
    """

    def __init__(self, dim: int) -> None:
        self._dim = int(dim)

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """为单条文本生成向量。"""

        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """为多条文本生成向量。"""

        if not texts:
            raise ValueError("texts 列表不能为空")

        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            b = t.encode("utf-8", errors="ignore")
            acc = np.zeros((self._dim,), dtype=np.float32)
            for j, v in enumerate(b):
                acc[j % self._dim] += float(v)
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


def _make_descriptions(uid: int, n: int, *, base_len: int) -> list[str]:
    """生成用户画像描述文本。

    Args:
        uid: 用户 ID。
        n: 描述条数。
        base_len: 目标文本长度（近似）。

    Returns:
        描述列表。
    """

    seed = int(uid) % 1_000_000
    rng = random.Random(seed)

    words = [
        "喜欢", "讨厌", "经常", "偶尔", "游戏", "音乐", "学习", "编程", "旅行", "吃饭", "运动", "猫", "狗",
        "二次元", "小说", "电影", "历史", "地理", "物理", "数学",
    ]

    out: list[str] = []
    for i in range(max(0, int(n))):
        parts = [f"uid={uid}", f"desc={i}"]
        while len(" ".join(parts)) < max(16, int(base_len)):
            parts.append(rng.choice(words))
        out.append(" ".join(parts))
    return out


async def run_perf(
    *,
    qdrant_url: str,
    qdrant_api_key: str | None,
    openai_url: str,
    openai_model: str,
    openai_key: str,
    skip_network: bool,
    vector_dim: int,
    collection_name: str | None,
    users: int,
    descs_per_user: int,
    embed_batch: int,
    queries: int,
    n_results: int,
    concurrency: int,
    warmup: int,
    repeats: int,
    base_desc_len: int,
    skip_delete: bool,
    skip_cleanup: bool,
) -> int:
    """运行性能测试。

    Args:
        qdrant_url: Qdrant 服务 URL。
        qdrant_api_key: Qdrant API Key。
        openai_url: embeddings URL。
        openai_model: embeddings 模型名。
        openai_key: embeddings key。
        skip_network: 是否跳过联网。
        vector_dim: 跳过联网时向量维度。
        collection_name: collection 名称。
        users: 用户数。
        descs_per_user: 每用户描述数。
        embed_batch: embedding 压测 batch。
        queries: 查询次数。
        n_results: 每次检索返回条数。
        concurrency: 并发度。
        warmup: 预热次数。
        repeats: 重复次数。
        base_desc_len: 描述长度（近似）。
        skip_delete: 是否跳过删除。
        skip_cleanup: 是否跳过清理 collection。

    Returns:
        退出码。
    """

    if not qdrant_url:
        print("[FATAL] qdrant_url 不能为空")
        return 1

    longmemory_dir = os.path.dirname(os.path.abspath(__file__))
    pkg = _load_longmemory_package(longmemory_dir)

    _load_module_from_path(f"{pkg}.model", os.path.join(longmemory_dir, "model.py"))
    vec_mod = _load_module_from_path(f"{pkg}.VectorGenerator", os.path.join(longmemory_dir, "VectorGenerator.py"))
    qdrant_mod = _load_module_from_path(f"{pkg}.qdrant_user_profile", os.path.join(longmemory_dir, "qdrant_user_profile.py"))

    QdrantUserProfile = getattr(qdrant_mod, "QdrantUserProfile")

    try:
        from qdrant_client.async_qdrant_client import AsyncQdrantClient
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] 无法导入 qdrant-client: {exc!r}")
        return 1

    OpenaiVectorGenerator = getattr(vec_mod, "OpenaiVectorGenerator")

    if skip_network:
        vector_generator: Any = _DummyVectorGenerator(vector_dim)
        vector_size = int(vector_dim)
    else:
        if not openai_key:
            print("[FATAL] 未提供 openai_key（可用 --openai-key 或环境变量 OPENAI_API_KEY）")
            return 1
        vector_generator = OpenaiVectorGenerator(
            model_name=openai_model,
            api_key=openai_key,
            api_url=openai_url,
        )
        probe = await vector_generator.embed_query("dimension probe")
        vector_size = int(getattr(probe, "shape", [0])[0])

    name = collection_name or f"user_profile_qdrant_perf_{uuid4().hex}"
    print(f"collection_name={name}")

    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    try:
        service: Any = await QdrantUserProfile.create(
            collection_name=name,
            client=client,
            vector_generator=vector_generator,
            vector_size=int(vector_size),
        )

        # -------- 生成数据 --------
        user_ids = [10_000_000 + i for i in range(int(users))]
        user_docs: list[tuple[int, list[str]]] = [
            (uid, _make_descriptions(uid, int(descs_per_user), base_len=int(base_desc_len))) for uid in user_ids
        ]
        all_texts: list[str] = [t for _, docs in user_docs for t in docs]
        query_texts = random.sample(all_texts, k=min(int(queries), len(all_texts))) if all_texts else []

        metrics: list[Metric] = []

        # -------- 1) embed_documents 延迟 --------
        embed_lat_ms: list[float] = []
        batch_texts = [f"perf embed sample {i}" for i in range(max(1, int(embed_batch)))]

        for _ in range(max(0, int(warmup))):
            await vector_generator.embed_documents(batch_texts)

        for _ in range(max(1, int(repeats))):
            t0 = _now()
            await vector_generator.embed_documents(batch_texts)
            t1 = _now()
            embed_lat_ms.append(_ms(t1 - t0))

        metrics.append(Metric(name="embed_documents(batch)", unit="ms", values=embed_lat_ms))

        # -------- 2) 写入 add_user_profile 吞吐/延迟 --------
        add_lat_ms: list[float] = []

        async def add_one(uid: int, docs: list[str]) -> None:
            t0 = _now()
            await service.add_user_profile(uid, docs)
            t1 = _now()
            add_lat_ms.append(_ms(t1 - t0))

        add_tasks = [lambda uid=uid, docs=docs: add_one(uid, docs) for uid, docs in user_docs]

        for _ in range(max(0, int(warmup))):
            if user_docs:
                uid0, docs0 = user_docs[0]
                await service.add_user_profile(uid0, docs0[:1])

        t0_all = _now()
        await _run_concurrent(add_tasks, concurrency=int(concurrency))
        t1_all = _now()

        total_docs = int(users) * int(descs_per_user)
        add_throughput = (total_docs / (t1_all - t0_all)) if (t1_all - t0_all) > 0 else 0.0
        metrics.append(Metric(name="add_user_profile(per user)", unit="ms", values=add_lat_ms))
        metrics.append(Metric(name="add_throughput", unit="docs/s", values=[add_throughput]))

        # 获取一些 doc_id（用于 update/delete）
        doc_ids_for_update: list[str] = []
        for uid in random.sample(user_ids, k=min(len(user_ids), 5)):
            profile = await service.get_user_profiles(user_id=uid, limit=min(10, int(descs_per_user)))
            doc_ids_for_update.extend([d.doc_id for d in profile.description])

        # -------- 3) 单用户 get（metadata 排序） --------
        get_meta_ms: list[float] = []

        async def get_meta_one(uid: int) -> None:
            t0 = _now()
            await service.get_user_profiles(user_id=uid, limit=10, sort_by="last_updated")
            t1 = _now()
            get_meta_ms.append(_ms(t1 - t0))

        sampled_uids = random.sample(user_ids, k=min(len(user_ids), max(1, int(queries) or 1)))
        get_tasks = [lambda uid=uid: get_meta_one(uid) for uid in sampled_uids]
        await _run_concurrent(get_tasks, concurrency=int(concurrency))
        metrics.append(Metric(name="get_user_profiles(meta)", unit="ms", values=get_meta_ms))

        # -------- 4) 单用户 text 相似度检索 --------
        get_text_ms: list[float] = []

        async def get_text_one(uid: int, q: str) -> None:
            t0 = _now()
            await service.get_user_profiles(user_id=uid, limit=10, sort_by="text", text=q)
            t1 = _now()
            get_text_ms.append(_ms(t1 - t0))

        text_tasks: list[Callable[[], Any]] = []
        for q in query_texts[: max(1, int(queries))]:
            uid = random.choice(user_ids)
            text_tasks.append(lambda uid=uid, q=q: get_text_one(uid, q))

        for _ in range(max(0, int(warmup))):
            if user_ids and query_texts:
                await service.get_user_profiles(user_id=user_ids[0], limit=3, sort_by="text", text=query_texts[0])

        await _run_concurrent(text_tasks, concurrency=int(concurrency))
        metrics.append(Metric(name="get_user_profiles(text)", unit="ms", values=get_text_ms))

        # -------- 5) 跨用户检索 search_user_profiles --------
        search_ms: list[float] = []

        async def search_one(q: str) -> None:
            t0 = _now()
            await service.search_user_profiles(q, n_results=int(n_results))
            t1 = _now()
            search_ms.append(_ms(t1 - t0))

        search_tasks = [lambda q=q: search_one(q) for q in query_texts[: max(1, int(queries))]]

        for _ in range(max(0, int(warmup))):
            if query_texts:
                await service.search_user_profiles(query_texts[0], n_results=int(n_results))

        await _run_concurrent(search_tasks, concurrency=int(concurrency))
        metrics.append(Metric(name="search_user_profiles", unit="ms", values=search_ms))

        # -------- 6) 更新 update_by_doc_id（meta-only） --------
        update_ms: list[float] = []

        async def update_one(doc_id: str) -> None:
            t0 = _now()
            await service.update_by_doc_id(doc_id, last_read=time.time())
            t1 = _now()
            update_ms.append(_ms(t1 - t0))

        update_tasks = [lambda did=did: update_one(did) for did in doc_ids_for_update[: max(1, int(queries))]]
        await _run_concurrent(update_tasks, concurrency=int(concurrency))
        metrics.append(Metric(name="update_by_doc_id(meta-only)", unit="ms", values=update_ms))

        # -------- 7) 删除 --------
        if not skip_delete:
            del_doc_ms: list[float] = []

            async def delete_one(doc_id: str) -> None:
                t0 = _now()
                await service.delete_by_doc_id(doc_id)
                t1 = _now()
                del_doc_ms.append(_ms(t1 - t0))

            delete_tasks = [lambda did=did: delete_one(did) for did in doc_ids_for_update[: max(1, int(queries))]]
            await _run_concurrent(delete_tasks, concurrency=int(concurrency))
            metrics.append(Metric(name="delete_by_doc_id", unit="ms", values=del_doc_ms))

            del_user_ms: list[float] = []

            async def delete_user(uid: int) -> None:
                t0 = _now()
                await service.delete_all_by_user_id(uid, page_size=1000)
                t1 = _now()
                del_user_ms.append(_ms(t1 - t0))

            del_users = random.sample(user_ids, k=min(len(user_ids), max(1, int(users) // 10)))
            del_user_tasks = [lambda uid=uid: delete_user(uid) for uid in del_users]
            await _run_concurrent(del_user_tasks, concurrency=max(1, int(concurrency) // 2))
            metrics.append(Metric(name="delete_all_by_user_id", unit="ms", values=del_user_ms))

        print("=== QdrantUserProfile 性能测试指标 ===")
        _print_table(metrics)
        return 0

    finally:
        if not skip_cleanup:
            try:
                await client.delete_collection(collection_name=name)
            except Exception:
                pass
        try:
            await client.close()
        except Exception:
            pass


def main() -> int:
    """脚本入口。

    Returns:
        进程退出码。
    """

    parser = argparse.ArgumentParser(description="QdrantUserProfile 性能测试")
    parser.add_argument(
        "--qdrant-url",
        "-qdrant-url",
        type=str,
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant 服务 URL",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=os.getenv("QDRANT_API_KEY", ""),
        help="Qdrant API Key（可选）",
    )
    parser.add_argument(
        "--openai-url",
        type=str,
        default=os.getenv("OPENAI_URL", "http://172.26.226.57:30020/v1/embeddings"),
        help="OpenAI 兼容 embeddings 接口 URL",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "qwen3-embedding:0.6b"),
        help="Embedding 模型名",
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API Key（也可用环境变量 OPENAI_API_KEY）",
    )
    parser.add_argument("--skip-network", action="store_true", help="跳过联网（使用离线向量生成器）")
    parser.add_argument("--vector-dim", type=int, default=1024, help="--skip-network 时使用的向量维度")
    parser.add_argument("--collection-name", type=str, default="", help="collection 名称（默认随机生成）")
    parser.add_argument("--users", type=int, default=50, help="用户数")
    parser.add_argument("--descs-per-user", type=int, default=5, help="每用户描述数")
    parser.add_argument("--base-desc-len", type=int, default=64, help="描述长度（近似）")
    parser.add_argument("--embed-batch", type=int, default=32, help="embedding 压测 batch")
    parser.add_argument("--queries", type=int, default=50, help="查询次数")
    parser.add_argument("--n-results", type=int, default=10, help="检索返回条数")
    parser.add_argument("--concurrency", type=int, default=10, help="并发度")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数")
    parser.add_argument("--repeats", type=int, default=3, help="重复次数")
    parser.add_argument("--skip-delete", action="store_true", help="跳过删除阶段")
    parser.add_argument("--skip-cleanup", action="store_true", help="跳过 collection 清理")
    args = parser.parse_args()

    qdrant_url = str(args.qdrant_url).strip()
    qdrant_api_key = str(args.qdrant_api_key).strip() or None
    openai_url = str(args.openai_url).strip()
    openai_model = str(args.openai_model).strip()
    openai_key = str(args.openai_key).strip()

    name = str(args.collection_name).strip() or None

    try:
        return asyncio.run(
            run_perf(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                openai_url=openai_url,
                openai_model=openai_model,
                openai_key=openai_key,
                skip_network=bool(args.skip_network),
                vector_dim=int(args.vector_dim),
                collection_name=name,
                users=int(args.users),
                descs_per_user=int(args.descs_per_user),
                embed_batch=int(args.embed_batch),
                queries=int(args.queries),
                n_results=int(args.n_results),
                concurrency=int(args.concurrency),
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                base_desc_len=int(args.base_desc_len),
                skip_delete=bool(args.skip_delete),
                skip_cleanup=bool(args.skip_cleanup),
            )
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 用户中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] 脚本运行失败: {exc!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
