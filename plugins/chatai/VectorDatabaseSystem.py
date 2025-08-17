import asyncio
import json
import aiohttp
import chromadb
import chromadb.api
import chromadb.api.models.AsyncCollection
import chromadb.api.models.Collection
import chromadb.api.types
import chromadb.base_types
import numpy as np
import time
from typing import List, Optional, Tuple, Dict, Any, Union, Type
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datetime import datetime
from chromadb.utils import embedding_functions
import uuid
import chromadb.api.models
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from urllib.parse import urlparse
from pathlib import Path
import sys

dir_path = Path(__file__).parent
sys.path.append(str(dir_path))
from SQLiteManager import Message, GroupMessage, PrivateMessage




OneOrMany = chromadb.api.types.OneOrMany




import asyncio
import time
from collections import defaultdict
from functools import wraps

# 存储统计数据的全局字典
_func_stats = defaultdict(lambda: {
    'total_time': 0.0,
    'count': 0,
    'min_time': float('inf'),
    'max_time': 0.0
})
_stats_lock = asyncio.Lock()  # 异步锁保证线程安全

def async_timer(func):
    """异步函数计时装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        async with _stats_lock:
            stats = _func_stats[func.__name__]
            stats['total_time'] += elapsed
            stats['count'] += 1
            stats['min_time'] = min(stats['min_time'], elapsed)
            stats['max_time'] = max(stats['max_time'], elapsed)
        
        return result
    return wrapper

def sync_timer(func):
    """同步函数计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        # 同步函数不需要锁，因为GIL保证线程安全
        stats = _func_stats[func.__name__]
        stats['total_time'] += elapsed
        stats['count'] += 1
        stats['min_time'] = min(stats['min_time'], elapsed)
        stats['max_time'] = max(stats['max_time'], elapsed)
        
        return result
    return wrapper

def print_stats():
    """打印函数执行统计信息"""
    if not _func_stats:
        print("No function statistics available.")
        return
        
    print("\nFunction Performance Statistics")
    print("-" * 90)
    print("{:<20} | {:>8} | {:>12} | {:>12} | {:>12} | {:>12}".format(
        "Function", "Calls", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)"
    ))
    print("-" * 90)
    
    # 按总耗时排序（降序）
    sorted_stats = sorted(
        _func_stats.items(), 
        key=lambda x: x[1]['total_time'], 
        reverse=True
    )
    
    for name, data in sorted_stats:
        if data['count'] > 0:
            avg_time = data['total_time'] / data['count']
            print("{:<20} | {:>8} | {:>12.4f} | {:>12.4f} | {:>12.4f} | {:>12.4f}".format(
                name[:20],  # 限制函数名长度
                data['count'],
                data['total_time'],
                avg_time,
                data['min_time'],
                data['max_time']
            ))
    
    print("-" * 90)





class ChromeType:
    """专门存放ChromeDB类型"""
    Collection = chromadb.api.models.Collection.Collection
    AsyncCollection = chromadb.api.models.AsyncCollection.AsyncCollection
    ClientAPI = chromadb.api.ClientAPI
    AsyncClientAPI = chromadb.api.AsyncClientAPI
    QueryResult = chromadb.api.types.QueryResult
    GetResult = chromadb.api.types.GetResult
    Embedding = chromadb.api.types.Embedding
    PyEmbedding = chromadb.api.types.PyEmbedding
    WhereDocument = chromadb.base_types.WhereDocument
    Documents = chromadb.api.types.Documents
    Embeddings = chromadb.api.types.Embeddings
    Metadata = chromadb.api.types.Metadata
    ID = chromadb.api.types.ID
    IDs = chromadb.api.types.IDs



class ChromaData:
    """chromadb单条数据"""
    id: ChromeType.ID
    document: str|None
    metadata: ChromeType.Metadata | None
    embedding: ChromeType.Embedding|ChromeType.PyEmbedding|None
    __slots__ = ('id', 'document', 'metadata', 'embedding')

    def __init__(
        self,
        id: ChromeType.ID|None = None,
        document: str|None = None,
        metadata: ChromeType.Metadata | None = None,
        embedding: ChromeType.Embedding|ChromeType.PyEmbedding|None = None,
    ) -> None:
        if id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = id
        self.document = document
        self.metadata = metadata




class MetadataFormatError(Exception):
    """表示元数据格式错误的异常"""
    pass



class DatabaseConsistencyError(Exception):
    """数据库一致性错误"""
    pass




DEFAULT_TENANT = "default_tenant"



class OllamaEmbeddingService:
    """管理 Ollama API 配置和嵌入操作（异步优化版）"""
    def __init__(
        self, 
        base_url: str = "http://localhost:11434", 
        model_name: str = "nomic-embed-text",
        max_concurrent_requests: int = 50  # 增加并发请求数
    ):
        self.base_url = base_url
        self.model_name = model_name
        self._session = None
        # 创建固定的超时配置
        self._short_timeout = aiohttp.ClientTimeout(total=5)
        self._long_timeout = aiohttp.ClientTimeout(total=30)
        # 并发控制
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp 会话（单例模式）"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """关闭 aiohttp 会话"""
        if self._session is not None:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self):
        """支持异步上下文管理器"""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时关闭会话"""
        await self.close()
    
    async def _verify_connection(self):
        """验证 Ollama 连接（异步）"""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/tags", 
                timeout=self._short_timeout
            ) as response:
                response.raise_for_status()
                return True
        except Exception as e:
            raise ConnectionError(f"Ollama 连接失败: {str(e)}")
    
    async def update_config(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        """更新 Ollama 配置（异步）"""
        if base_url:
            self.base_url = base_url.rstrip('/')
        if model_name:
            self.model_name = model_name
        await self._verify_connection()
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量（异步批量优化版）"""
        session = await self._get_session()
        
        async def fetch_embedding(text: str) -> List[float]:
            """获取单个文本的嵌入向量"""
            async with self._semaphore:  # 使用信号量控制并发
                payload = {"model": self.model_name, "prompt": text}
                try:
                    async with session.post(
                        f"{self.base_url}/api/embeddings",
                        json=payload,
                        timeout=self._long_timeout
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data["embedding"]
                except Exception as e:
                    # 可以选择重试或者直接抛出异常
                    raise RuntimeError(f"嵌入生成失败（文本：{text[:20]}...）: {str(e)}")
        
        # 并行处理所有请求
        tasks = [fetch_embedding(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        
        # 归一化嵌入向量（解决相似度计算问题）
        normalized_embeddings = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized_embeddings.append((np.array(emb) / norm).tolist())
            else:
                normalized_embeddings.append(emb)
                
        return normalized_embeddings
    
    async def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（异步）"""
        embeddings = await self.generate_embeddings([text1, text2])
        return self._cosine_similarity(embeddings[0], embeddings[1])
    
    async def calculate_bulk_similarities(
        self, 
        text_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """批量计算多组文本对的相似度"""
        # 收集所有文本
        all_texts = []
        for text1, text2 in text_pairs:
            all_texts.append(text1)
            all_texts.append(text2)
        
        # 一次性获取所有嵌入
        embeddings = await self.generate_embeddings(all_texts)
        
        # 分组计算相似度
        results = []
        for i in range(0, len(embeddings), 2):
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]
            results.append(self._cosine_similarity(vec1, vec2))
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度（CPU 计算，保持同步）"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0.0

class AsyncChromaDBManager:
    """异步 Chroma 数据库管理器"""
    def __init__(self, chroma_client) -> None:
        self.chroma_client: ChromeType.AsyncClientAPI = chroma_client
        
    @classmethod
    async def init(cls, host: str = "localhost", port: int = 8000, tenant: str = DEFAULT_TENANT):
        """初始化 Chroma 客户端（异步）"""
        chroma_client = await chromadb.AsyncHttpClient(host=host, port=port, tenant=tenant)
        cls = cls(chroma_client)
        if not await cls._is_connected():
            raise ConnectionError("无法连接到 ChromaDB 服务器")
        return cls
    
    async def _is_connected(self, timeout: float = 0.2) -> bool:
            """检查是否与 ChromaDB 服务器连接
            
            Args:
                timeout: 超时时间（秒），默认 0.2 秒
                
            Returns:
                bool: 连接状态，True 表示已连接，False 表示未连接或超时
            """
            try:
                # 使用 asyncio.wait_for 设置超时
                heartbeat_ns = await asyncio.wait_for(
                    self.chroma_client.heartbeat(),
                    timeout=timeout
                )
                
                # 可选：验证心跳是否为有效的纳秒时间戳
                if isinstance(heartbeat_ns, (int, float)) and heartbeat_ns > 0:
                    return True
                else:
                    return False
                    
            except asyncio.TimeoutError:
                # 超时情况
                return False
            except Exception:
                # 其他异常（连接错误、网络错误等）
                return False

    async def create_collection(self, collection_name: str, metadata: Optional[dict[str, Any]] = None, embedding_function = None, get_or_create: bool = False) -> ChromeType.AsyncCollection:
        """
        创建一个具有给定名称和元数据的全新集合。
        ---
        Args:
            name: 要创建的集合的名称。
            metadata:可选的与集合关联的元数据。
            embedding_function:可选用于嵌入文档的功能。如果未提供，则使用默认嵌入功能。
            get_or_create:如果为 True，存在时返回现有集合。
        Returns:
            Collection:新创建的集合对象。

        Raises:
            ValueError：
            - 如果在 get_or_create 为 False 时，该集合已存在。
            - 如果提供的集合名称无效。"""
        return await self.chroma_client.create_collection(collection_name, metadata=metadata, embedding_function=embedding_function, get_or_create=get_or_create)

    async def get_or_create_collection(self, collection_name: str, metadata: Optional[dict[str, Any]] = None, embedding_function = None) -> ChromeType.AsyncCollection:
        """获取或创建一个具有给定名称和元数据的集合。
        ---
        Args:
            name: 要获取或创建的集合的名称。
            metadata:可选的与集合关联的元数据。
            embedding_function:可选用于嵌入文档的功能。如果未提供，则使用默认嵌入功能。
        Returns:
            Collection:新创建的集合对象。
        """
        return await self.chroma_client.get_or_create_collection(collection_name, metadata=metadata, embedding_function=embedding_function)

    async def get_collection(self, collection_name: str) -> ChromeType.AsyncCollection:
        """获取一个具有给定名称的集合。
        ---
        Args:
            name: 要获取的集合的名称。
        Returns:
            Collection:集合对象。
        """
        if collection_name not in await self.list_collections_name():
            raise ValueError(f"集合 {collection_name} 不存在")
        return await self.chroma_client.get_collection(collection_name)

    async def list_collections(self, limit:int = 100, offset:int = 0) -> List[ChromeType.AsyncCollection]:
        """列出所有集合

        Args:
            limit: 返回集合的最大数量
            offset: 在返回之前跳过的条目数
        """
        return list(await self.chroma_client.list_collections(limit=limit, offset=offset))

    async def list_collections_name(self, limit:int = 100, offset:int = 0) -> List[str]:
        """列出所有集合的名称

        Args:
            limit: 返回集合的最大数量
            offset: 在返回之前跳过的条目数
        """
        return [i.name for i in list(await self.chroma_client.list_collections(limit=limit, offset=offset))]

    async def delete_collection(self, collection_name: str):
        """删除一个集合

        Args:
            collection_name: 要删除的集合的名称
        """
        await self.chroma_client.delete_collection(collection_name)

    async def count_collection_records(self, collection: str|ChromeType.AsyncCollection) -> int:
        """返回集合中的记录的数量"""
        if isinstance(collection, str):
            if collection not in await self.list_collections_name():
                raise ValueError(f"集合 {collection} 不存在")
            collection = await self.get_or_create_collection(collection)
        return await collection.count()

    async def get_top_records(self, collection: str|ChromeType.AsyncCollection, top: int = 10) -> ChromeType.GetResult:
        """从指定集合中获取前N条记录。

        Args:
            collection (str | ChromeType.AsyncCollection): 集合名称字符串或集合对象。
                如果传入字符串，会检查该名称集合是否存在。
            top (int, optional): 要返回的记录数量，默认为10。

        Returns:
            ChromeType.GetResult: 包含查询结果的返回值对象。

        Raises:
            ValueError: 当传入的集合名称不存在时抛出。

        Example:
            >>> await client.get_top_records("my_collection", 5)
            >>> await client.get_top_records(collection_obj)
        """
        
        if isinstance(collection, str):
            if collection not in await self.list_collections_name():
                raise ValueError(f"集合 {collection} 不存在")
            collection = await self.get_or_create_collection(collection)
        return await collection.peek(limit=top)

    @async_timer
    async def add_records_to_collection(
        self,
        collection: str | ChromeType.AsyncCollection,
        documents: list[str] | None,
        ids: Optional[list[str]] = None,
        metadatas: Optional[OneOrMany[ChromeType.Metadata]] = None,
        embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]] = None
    ) -> list[str]:
        """向指定集合中添加记录。

        可以接受集合名称或集合对象作为输入，当传入集合名称时会自动检查集合是否存在。
        若集合不存在将抛出异常，存在则自动获取该集合引用。

        Args:
            collection (str | ChromeType.AsyncCollection): 目标集合，可以是集合名称字符串或集合对象
            documents (list[str] | None): 要添加的文档内容列表，可为 None（例如仅使用 embedding）
            ids (list[str] | None): 要添加记录的ID列表，长度应与 documents 或 embeddings 一致
            metadatas (OneOrMany[ChromeType.Metadata] | None): 可选的元数据，默认为 None
            embeddings (OneOrMany[ChromeType.Embedding] | OneOrMany[ChromeType.PyEmbedding] | None): 可选的嵌入向量，默认为 None

        Returns:
            list[str]: 成功添加的记录 ID 列表

        Raises:
            ValueError: 当 documents 和 embeddings 均为 None，或长度不匹配时抛出

        Example:
            >>> await client.add_records_to_collection("my_collection", ["Hello, world!"], ["id1"])
            >>> await client.add_records_to_collection("my_collection", None, ["id2"], embeddings=[[0.1, 0.2, 0.3]])
        """
        # 确保至少提供 documents 或 embeddings 之一
        if documents is None and embeddings is None:
            raise ValueError("documents 和 embeddings 不能同时为 None")

        # 确定数据长度用于后续校验
        if documents is not None:
            data_length = len(documents)
        else:
            # 如果 documents 为 None，尝试从 embeddings 推断长度
            if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
                data_length = len(embeddings)
            else:
                # 处理单个 embedding 的情况
                data_length = 1

        if ids is None:
            ids = [uuid.uuid4().hex for _ in range(data_length)]
        elif len(ids) != data_length:
            raise ValueError("ids 的长度必须与 documents 或 embeddings 的数量一致")

        if isinstance(collection, str):
            if collection not in await self.list_collections_name():
                raise ValueError(f"集合 {collection} 不存在")
            collection = await self.get_or_create_collection(collection)

        # 调用底层 add 方法，documents 可为 None
        await collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return ids

    async def delete_records_from_collection(
        self, 
        collection: str|ChromeType.AsyncCollection, 
        ids: Optional[list[str]] = None,
        wheres: Optional[dict[str, Any]] = None
    ) -> None:
        """从指定集合中删除记录。
        
        根据提供的ID列表或条件字典从集合中删除对应的记录。必须至少提供其中一种参数。

        Args:
            collection (str|ChromeType.AsyncCollection): 要操作的集合，可以是集合名称字符串或AsyncCollection对象
            ids (Optional[list[str]]): 要删除的记录ID列表。当为None时，必须提供wheres参数
            wheres (Optional[dict[str, Any]]): 删除条件字典。当为None时，必须提供ids参数

        Returns:
            None: 该方法没有返回值

        Raises:
            ValueError: 如果同时未提供ids和wheres参数
            ValueError: 如果传入的集合名称不存在于数据库中

        Example:
            >>> # 删除特定ID的记录
            >>> await delete_records_from_collection("my_collection", ids=["id1", "id2"])
            >>> 
            >>> # 根据条件删除记录
            >>> await delete_records_from_collection("my_collection", wheres={"color": "red"})
        """
        if ids is None and wheres is None:
            raise ValueError("必须提供ids或wheres参数")
        if isinstance(collection, str):
            if collection not in await self.list_collections_name():
                raise ValueError(f"集合 {collection} 不存在")
            collection = await self.get_or_create_collection(collection)
        await collection.delete(ids=ids, where=wheres)

    async def query_records_from_collection(
        self,
        collection: str|ChromeType.AsyncCollection,
        query_texts: Optional[str] = None,
        query_embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]] = None,
        n_results: int = 10,
        ids: Optional[list[str]] = None,
        wheres: Optional[dict[str, Any]] = None,
        where_documents: Optional[ChromeType.WhereDocument] = None,
    ) -> ChromeType.QueryResult:
        """从指定集合中查询相似记录。

        根据提供的查询文本或嵌入向量，在指定集合中检索最相似的记录。支持基于ID筛选和元数据过滤。

        Args:
            collection: 要查询的集合名称或AsyncCollection对象。
            query_texts: 查询文本字符串。与query_embeddings参数互斥。
            query_embeddings: 查询嵌入向量，支持单个或多个向量输入。与query_texts参数互斥。
            n_results: 返回结果的最大数量，默认为10。
            ids: 可选的ID列表，用于限定在特定记录范围内查询。
            wheres: 可选的条件字典，用于基于元数据的过滤（例如：{"key": "value"}）。
            where_documents: 可选的文档过滤条件。

        Returns:
            ChromeType.QueryResult: 包含查询结果的对象。

        Raises:
            ValueError: 
                - 当既未提供query_texts也未提供query_embeddings时
                - 当同时提供query_texts和query_embeddings时
                - 当指定集合不存在时

        Example:
            >>> result = await query_records_from_collection(
                    collection="my_collection",
                    query_texts="搜索文本",
                    n_results=5
                )
        """
        if query_embeddings is None and query_texts is None:
            raise ValueError("必须提供query_texts或query_embeddings参数")
        if not query_embeddings is None and not query_texts is None:
            raise ValueError("只能提供query_texts或query_embeddings参数中的一个")
        if isinstance(collection, str):
            if collection not in await self.list_collections_name():
                raise ValueError(f"集合 {collection} 不存在")
            collection = await self.get_or_create_collection(collection)
        if query_texts:
            return await collection.query(query_texts=query_texts, n_results=n_results, ids=ids, where=wheres, where_document=where_documents)
        else:
            return await collection.query(query_embeddings=query_embeddings, n_results=n_results, ids=ids, where=wheres, where_document=where_documents)

    async def get_records_from_collection(
        self,
        collection: str|ChromeType.AsyncCollection,
        ids: Optional[list[str]] = None,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[ChromeType.WhereDocument] = None,
        limits: int = 100,
        offsets: int = 0,
    ) -> ChromeType.GetResult:
        """从指定集合中异步获取记录。

        从数据存储中获取嵌入向量及其关联数据。若未提供ID或where筛选条件，则返回从offset开始至limit范围内的所有嵌入向量。

        参数:
            collection (str | ChromeType.AsyncCollection): 集合名称或已加载的集合对象。
                如果是字符串，会先调用get_collection()方法加载集合。
            ids (Optional[list[str]], 可选): 要查询的文档ID列表。
                默认为None，表示查询所有记录。
            where (Optional[dict[str, Any]], 可选): 基于元数据的过滤条件字典。
                默认为None。
            where_document (Optional[ChromeType.WhereDocument], 可选): 文档内容的过滤条件。
                默认为None。
            limits (int, 可选): 返回记录的最大数量。
                默认为100。
            offsets (int, 可选): 查询结果的起始偏移量。
                默认为0，表示从第一条记录开始。

        返回:
            ChromeType.GetResult: 包含查询结果的GetResult对象，其中包含匹配的文档列表。

        示例:
            >>> # 按ID查询
            >>> await get_records_from_collection("my_collection", ids=["doc1", "doc2"])
            >>> # 带条件查询
            >>> await get_records_from_collection("my_collection", where={"category": "news"})
        """
        if isinstance(collection, str):
            collection = await self.get_collection(collection)
        return await collection.get(ids=ids, where=where, where_document=where_document, limit=limits, offset=offsets)

    async def update_records_in_collection(
        self,
        collection: str|ChromeType.AsyncCollection,
        ids: list[str],
        embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]] = None,
        documents: Optional[list[str]] = None,
        metadatas: Optional[OneOrMany[ChromeType.Metadata]] = None,
    ) -> None:
        """
        更新提供的 ids 的嵌入、元数据或文档。

        ### 元数据更新规则
        - 原元数据的键在新的元数据中不存在，保留原元数据中的键值对。
        - 原元数据的键在新的元数据中存在，替换原元数据中的值。
        - 新的元数据中存在原元数据中不存在的键，添加新的键值对。

        ### 文档和嵌入更新规则
        - 未提供文档, 不更新文档和嵌入
        - 提供文档, 新文档等于原文档, 更新嵌入, 不更新文档
        - 提供文档, 新文档不等于原文档, 更新文档和嵌入

        Args:
            collection: 要更新的集合名称或AsyncCollection对象。
            ids: 要更新的文档ID列表。
            embeddings: 要更新的嵌入向量, 如果该项为空将自动生成.
            documents: 要更新的文档内容列表。
            metadatas: 要更新的元数据列表。
            use_ollama: 是否使用Ollama服务生成嵌入向量, 默认为True
        """
        if isinstance(collection, str):
            collection = await self.get_collection(collection)
        return await collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

class OlromaDBManager(AsyncChromaDBManager):
    def __init__(self, ol, ch):
        self.ollama_client:OllamaEmbeddingService = ol
        super().__init__(ch)


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """关闭所有资源"""
        if hasattr(self, 'ollama_client'):
            await self.ollama_client.close()

    @classmethod
    async def init(
        cls, 
        chromadb_url = "http://127.0.0.1:8000", 
        ollama_url = "http://127.0.0.1:8001", 
        tenant = DEFAULT_TENANT, 
        model_name = "nomic-embed-text",
        max_concurrent_requests: int = 50
    ):
        ol = OllamaEmbeddingService(ollama_url, model_name, max_concurrent_requests)
        parsed = urlparse(chromadb_url)
        ch_host = parsed.hostname
        ch_port = parsed.port
        if not ch_host or not ch_port:
            raise ValueError("Invalid chromadb_url")
        ch = await AsyncChromaDBManager.init(host=ch_host, port=ch_port, tenant=tenant)
        return cls(ol, ch)
    

    async def get_embeddings(self, texts: list[str]):
        """异步获取文本嵌入向量。
        
        该方法通过连接的Ollama服务将输入的文本列表转换为嵌入向量，并返回一个包含
        numpy数组的列表，每个数组对应一个输入文本的嵌入向量。

        Args:
            texts: 字符串列表，每个字符串代表需要转换为嵌入向量的文本。

        Returns:
            list[np.ndarray]: 包含np.float32类型numpy数组的列表，每个数组对应输入
            文本的嵌入向量表示。数组的具体维度取决于所使用的嵌入模型。

        Example:
            >>> embeddings = await get_embeddings(["这是一个示例文本", "另一个文本"])
            >>> len(embeddings)
            2
        """
        return [np.array(i) for i in await self.ollama_client.generate_embeddings(texts)]
    
    async def add_records_to_collection(
        self, 
        collection: str|ChromeType.AsyncCollection, 
        documents:list[str]|None = None, 
        ids:Optional[list[str]] = None, 
        metadatas: Optional[OneOrMany[ChromeType.Metadata]] = None,
        embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]] = None,
        use_ollama: bool = True
    ) -> list[str]:
        """
        向指定集合中添加记录。

        该方法可以根据提供的集合名称或集合对象，向ChromaDB数据库中添加新的记录。
        可以选择使用Ollama服务自动生成嵌入向量，或者直接提供预计算的嵌入向量。

        Args:
            collection (str | ChromeType.AsyncCollection): 目标集合，可以是集合名称字符串或集合对象。
                如果传入字符串，会自动检查该名称集合是否存在。如果不存在，则抛出异常。
            ids (list[str]): 要添加记录的ID列表，长度应与documents参数一致。
            documents (list[str]): 要添加的文档内容列表。
            metadata (Optional[dict[str, Any]]): 可选的元数据字典，用于存储与文档相关的附加信息。
                默认为None。
            embeddings (Optional[OneOrMany[ChromeType.Embedding]]): 可选的嵌入向量列表。
                如果提供，应与documents参数一一对应。默认为None。
            use_ollama (bool, optional): 是否使用Ollama服务生成嵌入向量。
                如果为True且未提供embeddings参数，则使用Ollama生成嵌入向量。
                如果为False，则必须提供embeddings参数。默认为True。

        Raises:
            ValueError: 当传入的集合名称不存在时。
            ValueError: 当use_ollama为True但同时提供了embeddings参数时。
            ValueError: 当use_ollama为False但未提供embeddings参数时。
            ValueError: 当ids和documents参数长度不一致时。

        Example:
            # 使用Ollama生成嵌入向量
            >>> await client.add_records_to_collection(
            ...     collection="my_collection",
            ...     ids=["id1", "id2"],
            ...     documents=["文档1", "文档2"],
            ...     metadata={"author": "张三"},
            ...     use_ollama=True
            ... )
        """
        if use_ollama:
            if embeddings:
                raise ValueError("当use_ollama为True时，embeddings参数不能提供")
            if not documents:
                raise ValueError("当use_ollama为True时，documents参数不能为空")
            embeddings = await self.get_embeddings(documents)
        return await super().add_records_to_collection(collection, ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        
    async def query_records_from_collection(
        self,
        collection: str | ChromeType.AsyncCollection,
        query_texts: Optional[str] = None,
        query_embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]] = None,
        n_results: int = 10,
        ids: Optional[list[str]] = None,
        wheres: Optional[dict[str, Any]] = None,
        where_documents: Optional[ChromeType.WhereDocument] = None,
        use_ollama: bool = True
    ) -> ChromeType.QueryResult:
        """从指定集合中查询相似记录。

        根据提供的查询文本或嵌入向量，在指定集合中检索最相似的记录。支持基于ID筛选和元数据过滤。

        Args:
            collection: 要查询的集合名称或AsyncCollection对象。
            query_texts: 查询文本字符串。与query_embeddings参数互斥。
            query_embeddings: 查询嵌入向量，支持单个或多个向量输入。与query_texts参数互斥。
            n_results: 返回结果的最大数量，默认为10。
            ids: 可选的ID列表，用于限定在特定记录范围内查询。
            wheres: 可选的条件字典，用于基于元数据的过滤（例如：{"key": "value"}）。
            where_documents: 可选的文档过滤条件。
            use_ollama: 是否使用Ollama生成query_texts的嵌入向量，默认为True。

        Returns:
            ChromeType.QueryResult: 包含查询结果的对象。

        Raises:
            ValueError: 
                - 当既未提供query_texts也未提供query_embeddings时
                - 当同时提供query_texts和query_embeddings时
                - 当指定集合不存在时
                - 当use_ollama为True但同时提供了query_embeddings时

        Example:
            >>> # 使用文本查询（自动生成嵌入）
            >>> result = await query_records_from_collection(
            ...     collection="my_collection",
            ...     query_texts="搜索文本",
            ...     n_results=5
            ... )
        """
        # 参数验证
        self._validate_query_parameters(query_texts, query_embeddings, use_ollama)
        
        # 获取集合对象
        collection_obj = await self._get_collection_object(collection)
        
        # 处理查询参数
        final_query_embeddings = await self._prepare_query_embeddings(
            query_texts, query_embeddings, use_ollama
        )
        
        # 执行查询
        return await self._execute_query(
            collection_obj=collection_obj,
            query_embeddings=final_query_embeddings,
            query_texts=query_texts if not use_ollama else None,
            n_results=n_results,
            ids=ids,
            wheres=wheres,
            where_documents=where_documents
        )

    def _validate_query_parameters(
        self,
        query_texts: Optional[str],
        query_embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]],
        use_ollama: bool
    ) -> None:
        """验证查询参数的有效性。"""
        # 检查是否提供了查询参数
        if query_texts is None and query_embeddings is None:
            raise ValueError("必须提供query_texts或query_embeddings参数")
        
        # 检查是否同时提供了两种查询参数
        if query_texts is not None and query_embeddings is not None:
            raise ValueError("只能提供query_texts或query_embeddings参数中的一个")
        
        # 检查use_ollama与query_embeddings的冲突
        if use_ollama and query_embeddings is not None:
            raise ValueError("当use_ollama为True时，不能同时提供query_embeddings参数")
        
        # 检查use_ollama但没有query_texts的情况
        if use_ollama and query_texts is None:
            raise ValueError("当use_ollama为True时，必须提供query_texts参数")

    async def _get_collection_object(
        self,
        collection: str | ChromeType.AsyncCollection
    ) -> ChromeType.AsyncCollection:
        """获取集合对象。"""
        if isinstance(collection, str):
            available_collections = await self.list_collections_name()
            if collection not in available_collections:
                raise ValueError(f"集合 '{collection}' 不存在。可用集合: {available_collections}")
            return await self.get_or_create_collection(collection)
        return collection

    async def _prepare_query_embeddings(
        self,
        query_texts: Optional[str],
        query_embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]],
        use_ollama: bool
    ) -> Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]]:
        """准备查询用的嵌入向量。"""
        if use_ollama and query_texts is not None:
            # 使用Ollama生成嵌入向量
            try:
                embeddings_list = await self.get_embeddings([query_texts])
                return embeddings_list[0].tolist()  # 转换为列表格式
            except Exception as e:
                raise RuntimeError(f"使用Ollama生成嵌入向量失败: {str(e)}")
        
        return query_embeddings

    async def _execute_query(
        self,
        collection_obj: ChromeType.AsyncCollection,
        query_embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]],
        query_texts: Optional[str],
        n_results: int,
        ids: Optional[list[str]],
        wheres: Optional[dict[str, Any]],
        where_documents: Optional[ChromeType.WhereDocument]
    ) -> ChromeType.QueryResult:
        """执行实际的查询操作。"""
        try:
            if query_embeddings is not None:
                return await collection_obj.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    ids=ids,
                    where=wheres,
                    where_document=where_documents
                )
            else:
                return await collection_obj.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    ids=ids,
                    where=wheres,
                    where_document=where_documents
                )
        except Exception as e:
            raise RuntimeError(f"查询执行失败: {str(e)}")

    async def update_records_in_collection(
        self, 
        collection: str | chromadb.api.models.AsyncCollection.AsyncCollection, 
        ids: List[str], 
        embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]] = None,
        documents: Optional[list[str]] = None,
        metadatas: Optional[OneOrMany[ChromeType.Metadata]] = None,
        use_ollama: bool = True
    ) -> None:
        if use_ollama and not embeddings and documents:
            embeddings = await self.get_embeddings(documents)
        return await super().update_records_in_collection(
            collection=collection, 
            ids=ids, 
            embeddings=embeddings, 
            documents=documents, 
            metadatas=metadatas
        )









from tabulate import tabulate
import argparse

class ChromaDBCLI:
    def __init__(self, manager):
        self.manager = manager
        self.current_collection = None
    
    async def run(self):
        """主CLI交互循环"""
        print("🟢 ChromaDB 管理界面已启动 (输入 'help' 查看命令列表)")
        
        while True:
            try:
                # 显示当前状态
                status = f"[{self.current_collection}]" if self.current_collection else "[未选择集合]"
                cmd = input(f"\n🔷 ChromaDB {status} > ").strip().lower()
                
                if not cmd:
                    continue
                
                # 退出命令
                if cmd in ["exit", "quit", "q"]:
                    print("🛑 正在关闭 ChromaDB 连接...")
                    await self.manager.close()
                    print("👋 已退出")
                    break
                
                # 帮助命令
                elif cmd in ["help", "h"]:
                    self.show_help()
                
                # 列出集合
                elif cmd in ["list", "ls"]:
                    await self.list_collections()
                
                # 选择集合
                elif cmd.startswith("use "):
                    collection_name = cmd[4:].strip()
                    await self.select_collection(collection_name)
                
                # 创建集合
                elif cmd.startswith("create "):
                    collection_name = cmd[7:].strip()
                    await self.create_collection(collection_name)
                
                # 删除集合
                elif cmd.startswith("drop "):
                    collection_name = cmd[5:].strip()
                    await self.delete_collection(collection_name)
                
                # 显示集合信息
                elif cmd in ["info", "i"]:
                    await self.show_collection_info()
                
                # 添加记录
                elif cmd.startswith("add "):
                    parts = cmd[4:].split(maxsplit=1)
                    if len(parts) < 2:
                        print("❌ 格式错误。使用: add <文档内容> [--id 自定义ID] [--meta key=value]")
                    else:
                        await self.add_record(parts[1])
                
                # 查询记录
                elif cmd.startswith("find "):
                    query_text = cmd[5:].strip()
                    await self.query_records(query_text)
                
                # 删除记录
                elif cmd.startswith("del "):
                    identifier = cmd[4:].strip()
                    await self.delete_records(identifier)
                
                # 显示记录
                elif cmd in ["show", "s"]:
                    await self.show_records()
                
                # 清屏
                elif cmd == "clear":
                    print("\n" * 50)
                
                else:
                    print(f"❌ 未知命令: '{cmd}'。输入 'help' 查看可用命令")

            except Exception as e:
                print(f"🔥 发生错误: {str(e)}")

    def show_help(self):
        """显示帮助信息"""
        help_text = """
        🆘 ChromaDB 管理命令:
        
        🔹 集合操作:
          list (ls)          - 列出所有集合
          use <集合名>       - 选择当前操作的集合
          create <集合名>    - 创建新集合
          drop <集合名>      - 删除集合
          info (i)           - 显示当前集合信息
        
        🔹 记录操作:
          show (s)           - 显示当前集合中的记录
          add <文档内容>      - 添加新记录 (支持 --id 和 --meta 选项)
          find <查询文本>     - 查询相似记录
          del <ID或条件>      - 删除记录 (支持ID或元数据条件)
        
        🔹 系统命令:
          help (h)           - 显示此帮助信息
          clear              - 清屏
          exit (quit, q)     - 退出程序
        
        📌 示例:
          use my_collection      # 选择集合
          add "文档内容" --id doc1 --meta author=John
          find "搜索关键词"       # 查找相似文档
          del doc1               # 按ID删除
          del --meta category=old # 按元数据删除
        """
        print(help_text)

    async def list_collections(self):
        """列出所有集合"""
        collections = await self.manager.list_collections_name()
        if not collections:
            print("ℹ️ 数据库中没有集合")
            return
        
        print("\n📚 集合列表:")
        for i, name in enumerate(collections, 1):
            count = await self.manager.count_collection_records(name)
            print(f"  {i}. {name} ({count} 条记录)")
    
    async def select_collection(self, collection_name: str):
        """选择当前操作的集合"""
        collections = await self.manager.list_collections_name()
        if collection_name not in collections:
            print(f"❌ 集合 '{collection_name}' 不存在")
            return
        
        self.current_collection = collection_name
        count = await self.manager.count_collection_records(collection_name)
        print(f"✅ 已选择集合: {collection_name} (包含 {count} 条记录)")
    
    async def create_collection(self, collection_name: str):
        """创建新集合"""
        try:
            await self.manager.create_collection(collection_name, get_or_create=True)
            self.current_collection = collection_name
            print(f"✅ 已创建并选择集合: {collection_name}")
        except Exception as e:
            print(f"❌ 创建集合失败: {str(e)}")
    
    async def delete_collection(self, collection_name: str):
        """删除集合"""
        if input(f"⚠️ 确定要删除集合 '{collection_name}' 及其所有记录吗? (y/n): ").lower() == "y":
            try:
                await self.manager.delete_collection(collection_name)
                if self.current_collection == collection_name:
                    self.current_collection = None
                print(f"✅ 已删除集合: {collection_name}")
            except Exception as e:
                print(f"❌ 删除集合失败: {str(e)}")
        else:
            print("🗑️ 删除操作已取消")
    
    async def show_collection_info(self):
        """显示当前集合信息"""
        if not self.current_collection:
            print("ℹ️ 请先使用 'use' 命令选择集合")
            return
        
        count = await self.manager.count_collection_records(self.current_collection)
        print(f"\n📊 集合 '{self.current_collection}' 信息:")
        print(f"  - 记录数量: {count}")
        
        # 显示前3条记录示例
        if count > 0:
            print("\n🔍 示例记录:")
            records = await self.manager.get_top_records(self.current_collection, top=3)
            self._display_records(records)
    
    async def add_record(self, cmd_str: str):
        """添加新记录"""
        if not self.current_collection:
            print("ℹ️ 请先使用 'use' 命令选择集合")
            return
        
        # 解析命令参数
        parts = cmd_str.split()
        document = ""
        custom_id = None
        metadata = {}
        
        i = 0
        while i < len(parts):
            if parts[i].startswith('"') or parts[i].startswith("'"):
                # 处理带引号的文档内容
                quote_char = parts[i][0]
                doc_parts = [parts[i][1:]]
                i += 1
                
                while i < len(parts) and not parts[i].endswith(quote_char):
                    doc_parts.append(parts[i])
                    i += 1
                
                if i < len(parts):
                    doc_parts.append(parts[i][:-1])
                document = " ".join(doc_parts)
            elif parts[i] == "--id" and i + 1 < len(parts):
                custom_id = parts[i + 1]
                i += 1
            elif parts[i] == "--meta" and i + 1 < len(parts):
                meta_str = parts[i + 1]
                if "=" in meta_str:
                    key, value = meta_str.split("=", 1)
                    metadata[key.strip()] = value.strip()
                i += 1
            i += 1
        
        if not document:
            print("❌ 必须提供文档内容")
            return
        
        try:
            ids = [custom_id] if custom_id else None
            await self.manager.add_records_to_collection(
                self.current_collection,
                documents=[document],
                ids=ids,
                metadata=metadata
            )
            print(f"✅ 已添加1条记录到 '{self.current_collection}'")
        except Exception as e:
            print(f"❌ 添加记录失败: {str(e)}")
    
    async def query_records(self, query_text: str):
        """查询相似记录"""
        if not self.current_collection:
            print("ℹ️ 请先使用 'use' 命令选择集合")
            return
        
        if not query_text:
            print("ℹ️ 请输入查询文本")
            return
        
        try:
            print(f"🔍 正在查询: '{query_text}'...")
            results = await self.manager.query_records_from_collection(
                self.current_collection,
                query_texts=query_text,
                n_results=5
            )
            
            if not results['documents']:
                print("ℹ️ 未找到匹配的记录")
                return
            
            print(f"\n✅ 找到 {len(results['documents'][0])} 条相关记录:")
            
            # 准备数据表格
            data = []
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ), 1):
                data.append([
                    i,
                    results['ids'][0][i-1],
                    doc[:70] + "..." if len(doc) > 70 else doc,
                    meta or {},
                    f"{dist:.4f}"
                ])
            
            # 显示表格
            print(tabulate(
                data, 
                headers=["#", "ID", "内容摘要", "元数据", "相似度"], 
                tablefmt="pretty"
            ))
            
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
    
    async def delete_records(self, identifier: str):
        """删除记录"""
        if not self.current_collection:
            print("ℹ️ 请先使用 'use' 命令选择集合")
            return
        
        if not identifier:
            print("ℹ️ 请提供要删除的ID或条件")
            return
        
        try:
            # 按ID删除
            if not identifier.startswith("--"):
                await self.manager.delete_records_from_collection(
                    self.current_collection,
                    ids=[identifier]
                )
                print(f"✅ 已删除ID为 '{identifier}' 的记录")
            # 按元数据删除
            elif identifier.startswith("--meta "):
                meta_str = identifier[7:]
                if "=" in meta_str:
                    key, value = meta_str.split("=", 1)
                    where = {key.strip(): value.strip()}
                    await self.manager.delete_records_from_collection(
                        self.current_collection,
                        wheres=where
                    )
                    print(f"✅ 已删除所有元数据 {key}='{value}' 的记录")
                else:
                    print("❌ 元数据格式错误，使用: --meta key=value")
            else:
                print("❌ 不支持的删除条件格式")
        
        except Exception as e:
            print(f"❌ 删除失败: {str(e)}")
    
    async def show_records(self, limit: int = 10):
        """显示当前集合中的记录"""
        if not self.current_collection:
            print("ℹ️ 请先使用 'use' 命令选择集合")
            return
        
        try:
            records = await self.manager.get_top_records(
                self.current_collection,
                top=limit
            )
            
            if not records['documents']:
                print("ℹ️ 集合中没有记录")
                return
            
            print(f"\n📋 '{self.current_collection}' 的前 {len(records['documents'])} 条记录:")
            self._display_records(records)
            
        except Exception as e:
            print(f"❌ 获取记录失败: {str(e)}")
    
    def _display_records(self, records: dict):
        """格式化显示记录结果"""
        # 准备数据表格
        data = []
        for i, (doc, meta, id_) in enumerate(zip(
            records['documents'], 
            records['metadatas'], 
            records['ids']
        ), 1):
            data.append([
                i,
                id_,
                doc[:50] + "..." if len(doc) > 50 else doc,
                meta or {}
            ])
        
        # 显示表格
        print(tabulate(
            data, 
            headers=["#", "ID", "内容摘要", "元数据"], 
            tablefmt="pretty"
        ))


async def CIL():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ChromaDB 管理界面")
    parser.add_argument("--chroma-url", default="http://localhost:8000", 
                        help="ChromaDB 服务器地址 (默认: http://localhost:8000)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", 
                        help="Ollama 服务器地址 (默认: http://localhost:11434)")
    parser.add_argument("--model", default="nomic-embed-text", 
                        help="嵌入模型名称 (默认: nomic-embed-text)")
    args = parser.parse_args()
    
    print("🚀 正在初始化 ChromaDB 管理器...")
    print(f"  - ChromaDB 地址: {args.chroma_url}")
    print(f"  - Ollama 地址: {args.ollama_url}")
    print(f"  - 嵌入模型: {args.model}")
    
    try:
        # 初始化管理器
        manager = await OlromaDBManager.init(
            chromadb_url=args.chroma_url,
            ollama_url=args.ollama_url,
            model_name=args.model
        )
        
        # 启动CLI
        cli = ChromaDBCLI(manager)
        await cli.run()
        
    except Exception as e:
        print(f"🔥 初始化失败: {str(e)}")
        sys.exit(1)









async def main():

    db = await AsyncChromaDBManager.init(host="127.0.0.1", port=30004)
    collection = await db.get_or_create_collection("test1_collection", metadata={"hnsw:space": "cosine"})
    # await db.add_records_to_collection(collection, documents=["编程", "感冒", "服务器", "代码", "debug", "细菌", "废物"])
    # await db.add_records_to_collection(collection, documents=["编程","服务器", "失败"], metadata=[{"type": "single", "index": 1}, {"type": "single", "index": 2}, {"type": "single", "index": 3}])
    # await db.add_records_to_collection(collection, documents=["t1"], metadata=[{"t":"a"}])
    # await db.add_records_to_collection(collection, documents=["t2"], metadata=[{"t":"b"}])
    # await db.add_records_to_collection(collection, documents=["t3"], metadata=[{"t":"b"}])
    # await db.add_records_to_collection(collection, documents=["t4"], metadata=[{"t":"a"}])
    # await db.add_records_to_collection(collection, documents=["t5"], metadata=[{"t":"b"}])
    # await db.add_records_to_collection(collection, documents=["t6"], metadata=[{"t":"a"}])
    # await db.add_records_to_collection(collection, documents=["t7"], metadata=[{"t":"a"}])
    # await db.add_records_to_collection(collection, documents=["t8"], metadata=[{"t":"b"}])

    # print((await db.get_records_from_collection(collection))["documents"])
    # print((await db.get_top_records(collection))["documents"])
    id = ["1"]

    id = await db.add_records_to_collection(collection, id, documents=["t1"], embeddings=[[0.0]], metadatas=[{"t":"a", "f":"a"}])
    input("press any key to continue...")
    await db.update_records_in_collection(collection, id, documents=["t1"], metadatas=[{"t":"b", "g":"a"}])
    input("press any key to continue...")

    
    await db.delete_collection("test1_collection")

    ollama = OllamaEmbeddingService(model_name="quentinz/bge-base-zh-v1.5:latest")
    await ollama.close()
    



    result = await db.get_top_records(collection, 2)
    if result["metadatas"]:
        print(result["metadatas"][0])
    await ollama.close()
    await db.delete_collection("test1_collection")




async def main2():
    rag = await GroupRAGManager.init(
        "http://127.0.0.1:30004",
        "http://127.0.0.1:11434",
        "quentinz/bge-base-zh-v1.5:latest",
        group_collection_name="test_group_collection"
    )
    test_msg = GroupMessage()
    test_msg.msg_id = 123
    test_msg.content = "1111"
    test_msg.group_id = 123
    test_msg.user_id = 456
    test_msg.user_name = "test_user"
    test_msg.send_time = 1234567890
    db = rag.olromadb
    # 重置集合
    if "test_group_collection" in await db.list_collections_name():
        await db.delete_collection("test_group_collection")
    rag.group_collection = await db.create_collection("test_group_collection", metadata={"hnsw:space": "cosine"})
    
    t1 = time.time()
    for i in range(32):
        await rag.add_message(test_msg)
    t2 = time.time()
    print(t2 - t1)
    input("press any key to continue...")
    await db.delete_collection("test_group_collection")
    await db.close()
    


async def test():
    """性能检测"""
    db = await AsyncChromaDBManager.init(host="127.0.0.1", port=30004)
    ollama = OllamaEmbeddingService(model_name="quentinz/bge-base-zh-v1.5:latest")
    # 重置集合
    if "test1_collection" in await db.list_collections_name():
        await db.delete_collection("test1_collection")
    collection = await db.create_collection("test1_collection", metadata={"hnsw:space": "cosine"})

    import tqdm

    # 测试单个数据的插入速度
    times = {
        "all": [],
        "generate": [],
        "add": [],
        "get": [],
        "update": [],
        "delete": []
    }
    alt = time.time()
    for i in tqdm.tqdm(range(100)):
        t = time.time()
        vector = await ollama.generate_embeddings(["你好世界"])
        vector = np.array(vector).astype(np.float32)
        times["generate"].append(time.time() - t)
        t = time.time()
        id = await db.add_records_to_collection(collection, documents=["你好世界"], embeddings=vector)
        times["add"].append(time.time() - t)
        t = time.time()
        g = await db.get_records_from_collection(collection, id)
        g["data"]
        times["get"].append(time.time() - t)
        t = time.time()
        await db.update_records_in_collection(collection, id, metadatas=[{"t":"a"}])
        times["update"].append(time.time() - t)
        t = time.time()
        await db.delete_records_from_collection(collection, id)
        times["delete"].append(time.time() - t)
    times["all"].append(time.time() - alt)

    print(f"平均生成向量时间: {np.mean(times['generate']) * 1000}, 最小值: {np.min(times['generate'])* 1000}, 最大值: {np.max(times['generate'])* 1000}")
    print(f"平均添加记录时间: {np.mean(times['add'])* 1000}, 最小值: {np.min(times['add'])* 1000}, 最大值: {np.max(times['add'])* 1000}")
    print(f"平均获取记录时间: {np.mean(times['get'])* 1000}, 最小值: {np.min(times['get'])* 1000}, 最大值: {np.max(times['get'])* 1000}")
    print(f"平均更新记录时间: {np.mean(times['update'])* 1000}, 最小值: {np.min(times['update'])* 1000}, 最大值: {np.max(times['update'])* 1000}")
    print(f"平均删除记录时间: {np.mean(times['delete'])* 1000}, 最小值: {np.min(times['delete'])* 1000}, 最大值: {np.max(times['delete'])* 1000}")
    print(f"总时间: {np.sum(times['all'])* 1000}")









    
if __name__ == "__main__":
    asyncio.run(test())
    # print_stats()
print("程序执行完毕。")
