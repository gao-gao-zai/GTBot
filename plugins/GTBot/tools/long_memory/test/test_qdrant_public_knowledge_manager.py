"""Qdrant 公共知识管理器（QdrantPublicKnowledge）端到端测试脚本。

用法示例：
    python plugins/GTBot/services/LongMemory/test/test_qdrant_public_knowledge_manager.py \
        --qdrant-url http://127.0.0.1:6333 \
        --collection public_knowledge_test \
        --embedding-url http://localhost:11434/v1/embeddings \
        --embedding-model qwen3-embedding:0.6b

说明：
    - 该脚本会真实连接 Qdrant 与 Embedding 服务。
    - 运行前请确保 Qdrant 已启动。
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from uuid import uuid4

from qdrant_client.async_qdrant_client import AsyncQdrantClient


def _load_module_from_path(module_name: str, file_path: str) -> types.ModuleType:
    """从指定路径动态加载模块。

    Args:
        module_name: 模块名。
        file_path: 文件路径。

    Returns:
        types.ModuleType: 已加载模块。
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: name={module_name}, path={file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_longmemory_package(longmemory_dir: str) -> str:
    """构造一个临时包，支持 LongMemory 内部相对导入。

    Args:
        longmemory_dir: LongMemory 目录绝对路径。

    Returns:
        str: 临时包名。
    """

    package_name = f"_longmemory_pk_testpkg_{uuid4().hex}"
    pkg = types.ModuleType(package_name)
    pkg.__path__ = [longmemory_dir]  # type: ignore[attr-defined]
    sys.modules[package_name] = pkg
    return package_name


async def main() -> None:
    """脚本入口。"""

    parser = argparse.ArgumentParser(description="QdrantPublicKnowledge 端到端测试")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="public_knowledge_test")
    parser.add_argument("--embedding-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--api-key", default=None)

    # 兼容参数别名（与项目其它位置常用命名保持一致）
    parser.add_argument("--openai-url", default=None)
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--openai-key", default=None)
    args = parser.parse_args()

    embedding_url = (
        str(args.embedding_url)
        if args.embedding_url is not None
        else str(args.openai_url) if args.openai_url is not None else "http://localhost:11434/v1/embeddings"
    )
    embedding_model = (
        str(args.embedding_model)
        if args.embedding_model is not None
        else str(args.openai_model) if args.openai_model is not None else "qwen3-embedding:0.6b"
    )
    api_key = (
        str(args.api_key)
        if args.api_key is not None
        else str(args.openai_key) if args.openai_key is not None else ""
    )

    longmemory_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    pkg = _load_longmemory_package(longmemory_dir)

    _load_module_from_path(f"{pkg}.model", os.path.join(longmemory_dir, "model.py"))
    vec_mod = _load_module_from_path(f"{pkg}.VectorGenerator", os.path.join(longmemory_dir, "VectorGenerator.py"))
    pk_mod = _load_module_from_path(f"{pkg}.PublicKnowledge", os.path.join(longmemory_dir, "PublicKnowledge.py"))

    OpenaiVectorGenerator = getattr(vec_mod, "OpenaiVectorGenerator")
    PublicKnowledge = getattr(sys.modules[f"{pkg}.model"], "PublicKnowledge")
    QdrantPublicKnowledge = getattr(pk_mod, "QdrantPublicKnowledge")

    client = AsyncQdrantClient(url=str(args.qdrant_url))
    vector_generator = OpenaiVectorGenerator(
        model_name=embedding_model,
        api_url=embedding_url,
        api_key=api_key,
    )

    service = await QdrantPublicKnowledge.create(
        collection_name=str(args.collection),
        client=client,
        vector_generator=vector_generator,
    )

    # 清理 collection（测试用）
    if await client.collection_exists(collection_name=str(args.collection)):
        await client.delete_collection(collection_name=str(args.collection))
    service = await QdrantPublicKnowledge.create(
        collection_name=str(args.collection),
        client=client,
        vector_generator=vector_generator,
    )

    # 1) 写入
    ids = await service.add_public_knowledge(
        [
            PublicKnowledge(title="梗：下次一定", content="当别人说‘下次一定’，通常表示大概率不会做。"),
            PublicKnowledge(title="聊天技巧：复述确认", content="先复述对方需求再给方案，能显著减少误解。"),
        ]
    )
    assert len(ids) == 2

    # 2) 检索
    hits = await service.search_public_knowledge("下次一定是什么意思", n_results=3)
    assert isinstance(hits, list)
    assert hits, "应至少命中 1 条"

    # 3) upsert_by_similarity：应更新而不是新建
    doc_id = await service.upsert_by_similarity(
        PublicKnowledge(title="梗：下次一定", content="一般用于敷衍，表示可能不会做。"),
        threshold=0.5,
    )
    assert doc_id

    got = await service.get_by_doc_id(doc_id)
    assert "敷衍" in got.content

    # 4) 删除
    n = await service.delete_by_doc_id(ids[1])
    assert n == 1

    await client.close()

    print("OK: QdrantPublicKnowledge 测试通过")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
