
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Protocol, overload, runtime_checkable

import aiohttp
import numpy as np
from numpy.typing import NDArray
from pydantic import SecretStr


DEFAULT_URL = "https://api.openai.com/v1/embeddings"


class VectorGenerationError(RuntimeError):
    """向量生成失败异常。"""


@runtime_checkable
class VectorGenerator(Protocol):
    """向量生成器统一接口（上游只依赖此接口）。

    上游推荐调用：
        - `embed(text)`：单入口，支持单条/批量
        - `embed_query(text)`：单条文本（需要区分 query/doc 逻辑时使用）
        - `embed_documents(texts)`：多条文本（需要区分 query/doc 逻辑时使用）

    兼容旧接口：
        - `generate_vector(text)`：支持 `str | list[str]`
    """

    @overload
    async def embed(self, text: str) -> NDArray[np.float32]:
        """生成单条文本向量。"""

    @overload
    async def embed(self, text: list[str]) -> NDArray[np.float32]:
        """生成多条文本向量。"""

    async def embed(self, text: str | list[str]) -> NDArray[np.float32]:
        """生成文本向量（Embedding），支持单条/批量。

        Args:
            text: 单条文本或文本列表。

        Returns:
            - 输入为 `str` 时返回 shape 为 `(dim,)` 的 `np.ndarray`（dtype=float32）。
            - 输入为 `list[str]` 时返回 shape 为 `(n, dim)` 的 `np.ndarray`（dtype=float32）。
        """

        ...

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """为单条文本生成向量。

        Args:
            text: 输入文本。

        Returns:
            向量（Embedding）。
        """

        ...

    async def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """为多条文本生成向量。

        Args:
            texts: 输入文本列表。

        Returns:
            shape 为 `(n, dim)` 的 `np.ndarray`（dtype=float32）。
        """

        ...


class BaseVectorGenerator(ABC):
    """向量生成器抽象基类。

    设计目标：
        - 让上游只依赖稳定的 `embed_query/embed_documents`。
        - 具体实现（OpenAI/Ollama/本地模型等）只需实现两个方法。
        - 保留 `generate_vector` 作为兼容层，避免上游逐个适配。
    """

    @abstractmethod
    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """为单条文本生成向量。"""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """为多条文本生成向量。"""

    @overload
    async def embed(self, text: str) -> NDArray[np.float32]:
        """生成单条文本向量。"""

    @overload
    async def embed(self, text: list[str]) -> NDArray[np.float32]:
        """生成多条文本向量。"""

    async def embed(self, text: str | list[str]) -> NDArray[np.float32]:
        """生成文本向量（Embedding），支持单条/批量。

        Args:
            text: 单条文本或文本列表。

        Returns:
            - 输入为 `str` 时返回 shape 为 `(dim,)` 的 `np.ndarray`（dtype=float32）。
            - 输入为 `list[str]` 时返回 shape 为 `(n, dim)` 的 `np.ndarray`（dtype=float32）。

        Raises:
            ValueError: 输入为空或列表为空。
        """

        if isinstance(text, str):
            return await self.embed_query(text)
        return await self.embed_documents(text)

    @overload
    async def generate_vector(self, text: str) -> NDArray[np.float32]:
        """生成文本向量（Embedding）。"""

    @overload
    async def generate_vector(self, text: list[str]) -> NDArray[np.float32]:
        """生成文本向量（Embedding）。"""

    async def generate_vector(self, text: str | list[str]) -> NDArray[np.float32]:
        """生成文本向量（Embedding）。

        说明：
            - 当 `text` 为 `str` 时，返回该文本的向量（`list[float]`）。
            - 当 `text` 为 `list[str]` 时，返回与输入等长的向量列表（`list[list[float]]`）。

        Args:
            text: 待生成向量的文本或文本列表。

        Returns:
            - 输入为 `str` 时返回 shape 为 `(dim,)` 的 `np.ndarray`（dtype=float32）。
            - 输入为 `list[str]` 时返回 shape 为 `(n, dim)` 的 `np.ndarray`（dtype=float32）。

        Raises:
            ValueError: 输入为空或列表为空。
        """

        return await self.embed(text)

    async def close(self) -> None:
        """关闭生成器持有的外部资源。

        默认实现不执行任何操作，方便不需要网络会话的实现类复用同一接口。
        具体子类若持有 HTTP 会话、文件句柄或模型连接，可按需覆盖该方法。

        Returns:
            None: 该方法仅负责资源回收，不返回业务结果。
        """

        return None


class OpenaiVectorGenerator(BaseVectorGenerator):
    """OpenAI 兼容 Embedding API 的向量生成器。

    该实现适用于：
        - OpenAI 官方 embeddings API
        - Ollama 的 OpenAI 兼容 embeddings API
        - 其他实现了 `/v1/embeddings` 的服务

    Args:
        model_name: Embedding 模型名。
        api_key: API Key（支持明文或 `SecretStr`）。
        api_url: Embedding 接口地址。
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | SecretStr,
        api_url: str = DEFAULT_URL,
    ) -> None:
        self.model_name: str = model_name
        self.api_key: str | SecretStr = api_key
        self.api_url: str = api_url

        api_key_value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self.headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key_value}",
        }
        self._timeout = aiohttp.ClientTimeout(total=60)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取可复用的 HTTP 会话。

        该方法会在首次请求时懒创建 `aiohttp.ClientSession`，并在已有会话已关闭时
        自动重建。这样可避免每次 embedding 都重新建立连接，降低本地服务场景下的
        固定网络开销，同时不改变请求协议和返回解析逻辑。

        Returns:
            aiohttp.ClientSession: 可直接用于发送 embedding 请求的会话对象。
        """

        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(timeout=self._timeout)
            return self._session

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        """为单条文本生成向量。

        Args:
            text: 输入文本。

        Returns:
            向量（Embedding）。

        Raises:
            ValueError: 输入为空。
            VectorGenerationError: 请求失败或返回格式不符合预期。
        """

        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """为多条文本生成向量。

        Args:
            texts: 输入文本列表。

        Returns:
            与输入等长的向量列表。

        Raises:
            ValueError: 输入为空列表，或包含空白字符串。
            VectorGenerationError: 请求失败或返回格式不符合预期。
        """

        if not texts:
            raise ValueError("texts 列表不能为空")
        for idx, item in enumerate(texts):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"texts[{idx}] 不能为空")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }

        session = await self._get_session()
        async with session.post(self.api_url, headers=self.headers, json=payload) as resp:
            if resp.status < 200 or resp.status >= 300:
                body = await resp.text()
                raise VectorGenerationError(
                    f"Embedding 请求失败: status={resp.status}, body={body}"
                )

            data: dict[str, Any] = await resp.json()

        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise VectorGenerationError(f"Embedding 返回格式异常: data={data}")

        if len(items) != len(texts):
            raise VectorGenerationError(
                f"Embedding 数量不匹配: input={len(texts)}, output={len(items)}"
            )

        first_item = items[0]
        if not isinstance(first_item, dict) or "embedding" not in first_item:
            raise VectorGenerationError(f"Embedding 返回格式异常: item={first_item}")
        first_embedding = first_item["embedding"]
        if not isinstance(first_embedding, list) or not first_embedding:
            raise VectorGenerationError(f"Embedding 返回格式异常: embedding={first_embedding}")

        dim = len(first_embedding)
        embeddings: NDArray[np.float32] = np.empty((len(items), dim), dtype=np.float32)
        embeddings[0] = np.asarray(first_embedding, dtype=np.float32)

        for i in range(1, len(items)):
            item = items[i]
            if not isinstance(item, dict) or "embedding" not in item:
                raise VectorGenerationError(f"Embedding 返回格式异常: item={item}")
            embedding = item["embedding"]
            if not isinstance(embedding, list) or not embedding:
                raise VectorGenerationError(f"Embedding 返回格式异常: embedding={embedding}")
            if len(embedding) != dim:
                raise VectorGenerationError(
                    f"Embedding 维度不一致: expected={dim}, actual={len(embedding)}"
                )
            embeddings[i] = np.asarray(embedding, dtype=np.float32)

        return embeddings

    async def close(self) -> None:
        """关闭当前生成器复用的 HTTP 会话。

        该方法用于在进程退出、容器销毁或测试清理阶段释放连接资源。
        若会话尚未创建或已关闭，会静默返回，避免给调用方增加额外状态判断。

        Returns:
            None: 该方法只负责资源清理。
        """

        async with self._session_lock:
            session = self._session
            self._session = None
        if session is not None and not session.closed:
            await session.close()


async def main():
    generator = OpenaiVectorGenerator(
        model_name="qwen3-embedding:0.6b",
        api_key="your_apiKeyHere",
        api_url="http://localhost:11434/v1/embeddings"
        # api_url="http://172.26.226.57:30020/v1/embeddings"
    )

    texts = [str(i) for i in range(12)]
    t1 = time.time()
    vectors = await generator.generate_vector(texts)
    t2 = time.time()
    print(f"批量生成 {len(texts)} 个向量耗时: {t2 - t1:.4f} 秒")

    # for i in texts:
    #     vec = await generator.generate_vector(i)

    # print(f"单独生成 {len(texts)} 个向量耗时: {time.time() - t2:.2f} 秒")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
