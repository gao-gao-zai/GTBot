import asyncio
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


OneOrMany = chromadb.api.types.OneOrMany
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

    async def add_records_to_collection(
        self, 
        collection: str|ChromeType.AsyncCollection, 
        ids:Optional[list[str]], 
        documents:list[str], 
        metadata: Optional[OneOrMany[ChromeType.Metadata]] = None,
        embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]] = None
    ) -> None:
        """向指定集合中添加记录。

        可以接受集合名称或集合对象作为输入，当传入集合名称时会自动检查集合是否存在。
        若集合不存在将抛出异常，存在则自动获取该集合引用。

        Args:
            collection (str|ChromeType.AsyncCollection): 目标集合，可以是集合名称字符串或集合对象
            ids (list[str]): 要添加记录的ID列表，长度应与documents参数一致
            documents (list[str]): 要添加的文档内容列表
            metadata (Optional[dict[str, Any]]): 可选的元数据字典，默认为None
            embeddings (Optional[ChromeType.OneOrMany[ChromeType.Embedding]]): 可选的嵌入向量，默认为None
        Raises:
            ValueError: 当传入的集合名称对应的集合不存在时抛出

        Example:
            >>> await client.add_record("my_collection", ["id1"], ["doc1"])
            >>> await client.add_record(collection_obj, ["id2"], ["doc2"], {"key": "value"})
        """
        if ids is None:
            ids = [uuid.uuid4().hex for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("ids和documents的长度不一致")
        if isinstance(collection, str):
            if collection not in await self.list_collections_name():
                raise ValueError(f"集合 {collection} 不存在")
            collection = await self.get_or_create_collection(collection)
        await collection.add(ids=ids, documents=documents, metadatas=metadata, embeddings=embeddings)

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
        documents:list[str], 
        ids:Optional[list[str]] = None, 
        metadata: Optional[OneOrMany[ChromeType.Metadata]] = None,
        embeddings: Optional[OneOrMany[ChromeType.Embedding]] | Optional[OneOrMany[ChromeType.PyEmbedding]] = None,
        use_ollama: bool = True
    ) -> None:
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
            embeddings = await self.get_embeddings(documents)
        await super().add_records_to_collection(collection, ids, documents, metadata, embeddings)
        
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






class RAGManager:
    def __init__(self, chromadb:AsyncChromaDBManager, ollama: OllamaEmbeddingService, oldb: OlromaDBManager):
        self.chromadb = chromadb
        self.ollama = ollama
        self.olromadb = oldb

    @classmethod
    async def init(cls, chromadb_url: str, ollama_url: str, model_name: str):
        chromadb = await AsyncChromaDBManager.init(chromadb_url)
        ollama = OllamaEmbeddingService(ollama_url, model_name)
        db = await OlromaDBManager.init(chromadb_url="http://127.0.0.1:30004", ollama_url="http://127.0.0.1:11434", model_name="quentinz/bge-small-zh-v1.5:latest")
        return cls(chromadb, ollama, db)

    async def add_message(self, content):
        """添加单条消息到数据库"""


async def main():

    db = await OlromaDBManager.init(chromadb_url="http://127.0.0.1:30004", ollama_url="http://127.0.0.1:11434", model_name="quentinz/bge-base-zh-v1.5:latest")
    collection = await db.get_or_create_collection("test1_collection", metadata={"hnsw:space": "cosine"})
    # await db.add_records_to_collection(collection, documents=["编程", "感冒", "服务器", "代码", "debug", "细菌", "废物"])
    await db.add_records_to_collection(collection, documents=["编程","服务器", "失败"])

    ollama = OllamaEmbeddingService(model_name="quentinz/bge-base-zh-v1.5:latest")
    await ollama.close()
    
    embedding = await ollama.generate_embeddings(["生物"])
    
    embedding = embedding[0]
    about = await db.query_records_from_collection(collection, query_embeddings=embedding, n_results=5, use_ollama=False)
    print(f"{about['ids']}")
    print(f"{about['documents']}")
    print(f"{about["distances"]}")
    embedding = await ollama.generate_embeddings(["胜利"])
    
    embedding = embedding[0]
    about = await db.query_records_from_collection(collection, query_embeddings=embedding, n_results=5, use_ollama=False)
    print(f"{about['ids']}")
    print(f"{about['documents']}")
    print(f"{about["distances"]}")
    print(await ollama.calculate_text_similarity("生物", "代码"))
    print(await ollama.calculate_text_similarity("生物", "天气"))
    print(await ollama.calculate_text_similarity("生物", "服务器"))
    print(await ollama.calculate_text_similarity("失败", "胜利"))
    v1 = [1.0, 0]
    v2 = [-1.0, 0]
    print(ollama._cosine_similarity(v1, v2))
    
    await ollama.close()
    await db.delete_collection("test1_collection")
    await db.close()

    
if __name__ == "__main__":
    asyncio.run(main())
print("程序执行完毕。")
