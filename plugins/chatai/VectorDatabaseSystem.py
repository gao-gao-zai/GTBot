import asyncio
import aiohttp
import chromadb.api
import chromadb.api.models.Collection
import chromadb.api.types
import numpy as np
import time
from typing import List, Optional, Tuple, Dict, Any, Union, Type
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datetime import datetime
from chromadb.utils import embedding_functions
import uuid
import chromadb.api.models

class ChromeType:
    """专门存放ChromeDB类型"""
    Collection = chromadb.api.models.Collection.Collection
    ClientAPI = chromadb.api.ClientAPI
    QueryResult = chromadb.api.types.QueryResult










class OllamaConfigManager:
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
                print(f"✅ 成功连接到 Ollama (模型: {self.model_name})")
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

class ChromaDBManager:
    """简化类型的ChromaDB向量数据库管理器"""
    
    def __init__(self, path: str = ".chroma_db", embedding_function=None):
        self.path = path
        self.client: ChromeType.ClientAPI = chromadb.PersistentClient(path=path)
        
        # 默认嵌入函数
        self.default_ef = None
        self.embedding_function = embedding_function or self.default_ef
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def heartbeat(self) -> int:
        return self.client.heartbeat()
    
    def reset(self):
        self.client.reset()
    
    async def create_collection(self, name: str, metadata=None, embedding_function=None):
        """创建集合"""
        ef = embedding_function or self.embedding_function
        metadata = metadata or {}
        metadata["created"] = str(datetime.now())
        
        # 确保使用余弦相似度
        if "hnsw:space" not in metadata:
            metadata["hnsw:space"] = "cosine"
        
        return self.client.create_collection(
            name=name,
            metadata=metadata,
            embedding_function=ef # type: ignore
        )
    
    async def get_collection(self, name: str, embedding_function=None):
        """获取集合"""
        ef = embedding_function or self.embedding_function
        return self.client.get_collection(name=name, embedding_function=ef) # type: ignore
    
    async def get_or_create_collection(self, name: str, metadata=None, embedding_function=None):
        """获取或创建集合"""
        ef = embedding_function or self.embedding_function
        metadata = metadata or {}
        
        # 确保使用余弦相似度
        if "hnsw:space" not in metadata:
            metadata["hnsw:space"] = "cosine"
            
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata,
            embedding_function=ef # type: ignore
        )
    
    async def list_collections(self, limit: int = 100, offset: int = 0) -> List[str]:
        """列出所有集合的名称"""
        collections = self.client.list_collections(limit=limit, offset=offset)
        return [collection.name for collection in collections]
    
    async def collection_exists(self, collection_name: str) -> bool:
        """判断集合是否存在"""
        collections = await self.list_collections()
        return collection_name in collections
    
    async def modify_collection(self, collection: ChromeType.Collection, new_name: Optional[str] = None, metadata=None):
        """修改集合属性"""
        if new_name or metadata:
            collection.modify(name=new_name, metadata=metadata)
    
    async def delete_collection(self, name: str):
        """删除集合"""
        self.client.delete_collection(name=name)
    
    async def add_documents(
        self,
        collection: ChromeType.Collection,
        ids: list,
        documents=None,
        metadatas=None,
        embeddings=None
    ):
        """添加文档"""
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
    
    async def update_documents(
        self,
        collection: ChromeType.Collection,
        ids: list,
        documents=None,
        metadatas=None,
        embeddings=None
    ):
        """更新文档"""
        collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
    
    async def upsert_documents(
        self,
        collection: ChromeType.Collection,
        ids: list,
        documents=None,
        metadatas=None,
        embeddings=None
    ):
        """更新或插入文档"""
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
    
    async def delete_documents(
        self,
        collection: ChromeType.Collection,
        ids: Optional[list] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None
    ):
        """删除文档"""
        collection.delete(
            ids=ids,
            where=where,
            where_document=where_document
        )
    
    async def query_collection(
        self,
        collection: ChromeType.Collection,
        query_texts=None,
        query_embeddings=None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: list = ["documents", "metadatas", "distances"]
    ) -> ChromeType.QueryResult:
        """查询集合"""
        return collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
    
    async def get_documents(
        self,
        collection: ChromeType.Collection,
        ids: Optional[list] = None,
        where: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[dict] = None,
        include: list = ["documents", "metadatas"]
    ):
        """获取文档"""
        return collection.get(
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
            where_document=where_document,
            include=include
        )
    
    async def count_documents(self, collection: ChromeType.Collection) -> int:
        """计算文档数量"""
        return collection.count()
    
    async def peek_collection(self, collection: ChromeType.Collection, limit: int = 10):
        """查看集合样本"""
        return collection.peek(limit=limit)
    
    @staticmethod
    def build_metadata_filter(field: str, operator: str, value) -> dict:
        """构建元数据过滤器"""
        return {field: {operator: value}}
    
    @staticmethod
    def build_logical_filter(operator: str, conditions: list) -> dict:
        """构建逻辑过滤器"""
        return {operator: conditions}
    
    @staticmethod
    def build_document_filter(operator: str, value: str) -> dict:
        """构建文档内容过滤器"""
        return {operator: value}
    
    @staticmethod
    def cosine_similarity(vec1, vec2) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0.0

class VectorStoreManager:
    """整合向量生成和存储的统一管理类"""
    
    def __init__(
        self, 
        db_path: str = ".chroma_db",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        max_concurrent_requests: int = 50
    ):
        """
        初始化向量存储管理器
        
        参数:
            db_path: ChromaDB数据库存储路径
            ollama_base_url: Ollama服务地址
            embedding_model: 嵌入模型名称
            max_concurrent_requests: 最大并发请求数
        """
        self.ollama = OllamaConfigManager(
            base_url=ollama_base_url,
            model_name=embedding_model,
            max_concurrent_requests=max_concurrent_requests
        )
        self.chroma = ChromaDBManager(path=db_path)
        self.collections: Dict[str, Any] = {}  # 存储集合名称到集合对象的映射
    
    async def __aenter__(self):
        """异步上下文入口"""
        await self.ollama._get_session()  # 初始化Ollama会话
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文出口"""
        await self.ollama.close()
    
    async def create_collection(self, name: str, metadata: Optional[dict] = None) -> Any:
        """
        创建新的向量集合
        
        参数:
            name: 集合名称
            metadata: 集合元数据
        
        返回:
            ChromaDB集合对象
        """
        collection = await self.chroma.create_collection(name, metadata)
        self.collections[name] = collection

        return collection
    
    async def get_collection(self, name: str) -> Any:
        """
        获取现有集合（不存在则自动创建）
        
        参数:
            name: 集合名称
        
        返回:
            ChromaDB集合对象
        """
        if name not in self.collections:
            collection = await self.chroma.get_or_create_collection(name)
            self.collections[name] = collection

        return self.collections[name]
    
    async def list_collections(self) -> List[str]:
        """
        获取所有集合名称列表
        
        返回:
            所有集合的名称列表
        """
        return await self.chroma.list_collections()
    
    async def collection_exists(self, collection_name: str) -> bool:
        """
        检查集合是否存在
        
        参数:
            collection_name: 集合名称
        
        返回:
            集合是否存在
        """
        return await self.chroma.collection_exists(collection_name)
    
    async def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        添加文档到指定集合（自动生成嵌入向量）
        
        参数:
            collection_name: 目标集合名称
            documents: 文档列表
            metadatas: 元数据列表（可选）
            ids: 文档ID列表（可选）
        
        返回:
            添加的文档数量
        """
        collection = await self.get_collection(collection_name)


        
        # 自动生成ID（如果未提供）
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("文档数量和ID数量不匹配")
        
        # 生成嵌入向量
        embeddings = await self.ollama.generate_embeddings(documents)
        
        # 添加文档到集合
        await self.chroma.add_documents(
            collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        return len(documents)
    
    async def query_similarity(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        include: list = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Any]:
        """
        查询相似文档
        
        参数:
            collection_name: 目标集合名称
            query_text: 查询文本
            n_results: 返回结果数量
            include: 包含的返回字段
        
        返回:
            查询结果字典（包含文档、元数据、相似度百分比等信息）
        """
        collection = await self.get_collection(collection_name)
        
        # 生成查询嵌入向量
        query_embedding = (await self.ollama.generate_embeddings([query_text]))[0]
        
        # 执行查询
        results = await self.chroma.query_collection(
            collection,
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include
        )
        
        # 转换结果格式并计算相似度百分比
        distances = results["distances"][0]
        # 将余弦距离转换为相似度百分比 (1 - 距离) * 100
        similarities = [round((1 - d) * 100, 2) for d in distances]
        
        formatted_results = {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "similarities": similarities  # 使用百分比相似度替换距离
        }
        
        return formatted_results


    async def query_collection(
    self,
    collection,
    query_texts=None,
    query_embeddings=None,
    n_results: int = 10,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None,
    include: list = ["documents", "metadatas", "distances"]
    ):
        """
        异步查询集合中的相似文档
        
        Args:
            collection: ChromaDB集合对象
            query_texts: 查询文本列表
            query_embeddings: 查询嵌入向量
            n_results: 返回结果数量，默认10
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            include: 包含的字段列表，默认["documents", "metadatas", "distances"]
        
        Returns:
            查询结果字典，包含匹配的文档、元数据和距离信息
        """

        return await collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )

    
    async def batch_add_from_source(
        self,
        collection_name: str,
        source: Union[List[str], Dict[str, str]],
        batch_size: int = 100
    ) -> int:
        """
        从数据源批量添加文档（自动分批次处理）
        
        参数:
            collection_name: 目标集合名称
            source: 数据源（字典：id->文档 或 列表：文档）
            batch_size: 每批次处理量
        
        返回:
            添加的文档总数
        """
        total_added = 0
        
        # 处理字典格式输入
        if isinstance(source, dict):
            ids = list(source.keys())
            documents = list(source.values())
        # 处理列表格式输入
        else:
            documents = source
            ids = None
        
        # 分批处理
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size] if ids else None
            
            added = await self.add_documents(
                collection_name,
                batch_docs,
                ids=batch_ids
            )
            total_added += added
        

        return total_added
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        删除指定集合
        
        参数:
            collection_name: 集合名称
        
        返回:
            是否成功删除
        """
        if collection_name in self.collections:
            await self.chroma.delete_collection(collection_name)
            del self.collections[collection_name]

            return True
        return False
    
    async def collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        参数:
            collection_name: 集合名称
        
        返回:
            包含统计信息的字典
        """
        collection = await self.get_collection(collection_name)
        count = await self.chroma.count_documents(collection)
        return {
            "collection": collection_name,
            "document_count": count,
            "created_at": collection.metadata.get("created", "unknown")
        }

"""
群聊集合元数据结构:
{
    "group_id": "群聊ID"(int),
    "user_id": "用户ID"(int),
    "message_id": "消息ID"(int),
    "timestamp": "时间戳"(float),
}
私聊集合元数据结构:
{
    "user_id": "用户ID"(int),
    "message_id": "消息ID"(int),
    "timestamp": "时间戳"(float),
}
"""





class GroupVectorStoreManager(VectorStoreManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_name = "group"


    async def add_message(
        self,
        group_id: int,
        user_id: int,
        message_id: int,
        message_text: str,
        timestamp: float
    ):
        """添加信息到群聊集合"""    
        await self.add_documents(
            self.collection_name, 
            [message_text], 
            metadatas=[{"group_id": group_id, "user_id": user_id, "message_id": message_id, "timestamp": timestamp}]
        )

    # async def get_message_from_id(self, message_id: int):
    #     """根据消息ID获取信息"""
    #     return await self.query_collection()



async def main():
    # 初始化向量存储管理器
    async with VectorStoreManager(
        db_path="ai_database",
        embedding_model="quentinz/bge-small-zh-v1.5:latest"
    ) as vector_store:
        
        # 检查集合是否存在，不存在则创建
        collection_name = "ai_concepts"
        if not await vector_store.collection_exists(collection_name):
            print(f"创建新集合: {collection_name}")
            await vector_store.create_collection(collection_name, {
                "category": "technology",
                "hnsw:space": "cosine"  # 关键修复：指定使用余弦相似度
            })
        else:
            print(f"使用现有集合: {collection_name}")
            await vector_store.get_collection(collection_name)
        
        # 添加文档
        documents = [
            "Python是一种广泛使用的高级编程语言",
            "机器学习是人工智能的核心研究领域",
            "深度学习基于神经网络构建模型",
            "向量数据库用于高效存储和检索嵌入向量",
            "Ollama是一个轻量级本地AI模型部署工具"
        ]

        # 添加文档到集合
        await vector_store.add_documents(collection_name, documents)
        
        # 批量添加文档
        large_dataset = {
            f"doc_{i}": f"这是第{i}个文档内容" for i in range(100)
        }
        await vector_store.batch_add_from_source(collection_name, large_dataset, batch_size=20)
        
        # 查询相似文档
        results = await vector_store.query_similarity(
            collection_name,
            "什么是神经网络?",
            n_results=3
        )
        
        # 打印结果（使用百分比相似度）
        print("\n相似文档查询结果:")
        for i, (doc, sim) in enumerate(zip(results["documents"], results["similarities"]), 1):
            print(f"{i}. [相似度: {sim}%] {doc[:60]}...")
        
        # 查看集合统计
        stats = await vector_store.collection_stats(collection_name)
        print(f"\n集合统计: {stats}")
        
        # 测试新增的集合管理方法
        print("\n测试集合管理方法:")
        # 获取所有集合列表
        collections = await vector_store.list_collections()
        print(f"当前所有集合: {collections}")
        
        # 检查特定集合是否存在
        exists = await vector_store.collection_exists(collection_name)
        print(f"'{collection_name}'集合存在: {exists}")
        
        # 检查不存在的集合
        not_exists = await vector_store.collection_exists("non_existent_collection")
        print(f"'non_existent_collection'集合存在: {not_exists}")



async def test_chroma_async():
    # 创建异步客户端
    client = await chromadb.AsyncHttpClient()
    try:
        # 测试创建集合（异步）
        collection = await client.create_collection(name="my_collection")
        print("✅ 集合创建成功")
        
        # 测试添加数据（异步）
        await collection.add(
            ids=["id1", "id2"],
            documents=["异步方法测试文档1", "异步方法测试文档2"],
            metadatas=[{"category": "test"}, {"category": "test"}]
        )
        print("✅ 数据添加成功")
        
        # 测试查询（异步）
        results = await collection.query(
            query_texts=["异步方法测试"],
            n_results=2
        )
        print("✅ 查询成功")
        print("查询结果:", results)
        
        # 测试删除（异步）
        await collection.delete(ids=["id1"])
        print("✅ 删除成功")
        
        # 验证删除结果
        results_after_delete = await collection.query(
            query_texts=["异步方法测试"],
            n_results=2
        )
        print("删除后剩余文档数:", len(results_after_delete['ids'][0]))
        
    finally:
        # 清理测试数据库
        await client.delete_collection("test_async")


# 运行测试
asyncio.run(test_chroma_async())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())