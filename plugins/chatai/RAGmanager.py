import asyncio
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Dict, Any, Union, Type, TypedDict, cast
import uuid
import sys
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from deepdiff import DeepDiff
from collections import deque
import traceback
import logging as logger
from timeer import *

sys.path.append(str(Path(__file__).parent.parent))

from VectorDatabaseSystem import OlromaDBManager, MetadataFormatError, DatabaseConsistencyError, ChromaData, ChromaType, GroupMessage, PrivateMessage, async_timer, sync_timer, print_stats
from fun import replace_cq_codes



DEFAULT_TENANT = "default_tenant"
GROUP_MESSAGE_COLLECTION_NAME = "group_messages"
"""群消息集合名称"""
# ---------------临时常量存区
MAX_SINGLE_NUMBER = 5
"""最大单条消息数量"""
SINGLE_FORMAT = "[{$month}月{$day}日 {$hour}:{$minute}] {$user_name}({$user_id}): {$content}"
"""单条消息格式"""
# ----------------------------

class SingleMetadataFormat(TypedDict):
    type: Literal["single"]
    index: int
    group_id: int
    related_user_id: List[int]
    related_msg_id: List[int]
    earliest_send_time: float
    latest_send_time: float
    message_count: int

class ChunkMetadataFormat(TypedDict):
    type: Literal["chunk"]
    chunk_index: int
    group_id: int
    related_user_id: List[int]
    related_msg_id: List[int]
    earliest_send_time: float
    latest_send_time: float
    is_locked: bool
    merge_count: int
    message_count: int

# 使用 Union 表示两种可能的元数据格式
MetadataFormat = SingleMetadataFormat | ChunkMetadataFormat

METADATA_FORMAT = {
    "type": str, # 数据类型, single, chunk
    "index": int, # 单条消息索引, 仅在type为single时有效
    "chunk_index": int, # 分块消息索引, 仅在type为chunk时有效
    "group_id": int, # 群聊ID
    "related_user_id": str, # 相关用户ID列表, json格式的字符串, 解析后为list[int]
    "related_msg_id": str, # 相关消息ID列表, json格式的字符串, 解析后为list[int]
    "earliest_send_time": float, # 最早的消息的发送时间
    "latest_send_time": float, # 最晚的消息的发送时间, 如果type为single, 则与earliest_send_time相同
    "is_locked": bool, # 是否锁定, 锁定的数据块不会被修改, 仅在type为chunk时有效
    "message_count": int, # 消息块包含的消息数量, 或叫块大小, 如果type为single, 则为1
    "merge_count": int, # 合并次数, 仅在type为chunk时有效
}

class ChromaResultItem:
    """
    表示 ChromaDB 查询结果中的单个完整条目
    包含所有相关信息：ID、文档、元数据、嵌入向量和距离
    """
    __slots__ = ('id', 'document', 'metadata', 'embedding', 'distance')
    
    def __init__(
        self,
        id: str,
        document: Optional[str] = None,
        metadata: Optional[ChromaType.Metadata] = None,
        embedding: Optional[ChromaType.PyEmbedding|ChromaType.Embedding] = None,
        distance: Optional[float] = None
    ) -> None:
        """
        初始化结果项
        
        参数:
            id: 唯一标识符
            document: 文档内容
            metadata: 元数据字典
            embedding: 嵌入向量
            distance: 相似度距离（值越小越相似）
        """
        self.id = id
        self.document = document
        self.metadata = metadata
        self.embedding = embedding
        self.distance = distance
    
    def similarity(self) -> float:
        """
        获取相似度分数（0-1之间，1表示完全相似）
        注意：距离值越小表示相似度越高
        """
        if self.distance is None:
            return 0.0
        # 根据距离类型调整计算方式（这里假设是余弦距离）
        return max(0.0, min(1.0, 1.0 - self.distance))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        return {
            "id": self.id,
            "document": self.document,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "distance": self.distance,
            "similarity": self.similarity()
        }
    
    def __repr__(self) -> str:
        return f"ChromaResultItem(id={self.id}, distance={self.distance:.4f}, doc_len={len(self.document) if self.document else 0})"



def query_result_to_full_results(
    query_result: ChromaType.QueryResult,
    query_index: int = 0
) -> List[ChromaResultItem]:
    """
    将 QueryResult 转换为完整的 ChromaResultItem 列表
    保留所有关键信息，包括距离值
    
    参数:
        query_result: ChromaDB 查询结果
        query_index: 要处理的查询批次索引 (默认取第一个查询)
    
    返回:
        单个查询结果转换后的 ChromaResultItem 列表
    """
    # 1. 检查查询结果是否存在
    if not query_result['ids'] or len(query_result['ids']) <= query_index:
        return []
    
    # 2. 获取指定查询批次的结果
    batch_ids = query_result['ids'][query_index]
    n_results = len(batch_ids)
    
    # 3. 安全获取各字段的对应批次数据（处理可能缺失的情况）
    batch_embeddings = (
        query_result['embeddings'][query_index] 
        if query_result['embeddings'] and len(query_result['embeddings']) > query_index 
        else [None] * n_results
    )
    batch_documents = (
        query_result['documents'][query_index] 
        if query_result['documents'] and len(query_result['documents']) > query_index 
        else [None] * n_results
    )
    batch_metadatas = (
        query_result['metadatas'][query_index] 
        if query_result['metadatas'] and len(query_result['metadatas']) > query_index 
        else [None] * n_results
    )
    batch_distances = (
        query_result['distances'][query_index] 
        if query_result['distances'] and len(query_result['distances']) > query_index 
        else [None] * n_results
    )
    
    # 4. 构建完整结果列表
    result = []
    for i in range(n_results):
        result.append(ChromaResultItem(
            id=batch_ids[i],
            document=batch_documents[i],
            metadata=batch_metadatas[i],
            embedding=batch_embeddings[i] if batch_embeddings else None,
            distance=batch_distances[i] if batch_distances else None
        ))
    
    return result

class ValidatedChromaData:
    """已验证过metadata存在的ChromaData"""
    id: ChromaType.ID
    document: Optional[str]
    metadata: MetadataFormat
    embedding: Optional[ChromaType.Embedding|ChromaType.PyEmbedding]
    __slots__ = ('id', 'document', 'metadata', 'embedding')

    def __init__(
        self,
        metadata: ChromaType.Metadata,  # 接收只读的Mapping
        id: Optional[ChromaType.ID] = None,
        document: Optional[str] = None,
        embedding: Optional[ChromaType.Embedding|ChromaType.PyEmbedding] = None,
    ) -> None:
        if id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = id
        self.document = document
        
        # 创建可变字典副本进行转换
        processed_metadata = self._process_metadata(metadata)
        self.metadata = cast(MetadataFormat, processed_metadata)
        self.embedding = embedding
        
        # 初始化时立即验证
        self.validate_metadata_format()

    def _process_metadata(self, metadata: ChromaType.Metadata) -> dict:
        """处理元数据：转换相关ID字段为列表，处理只读映射"""
        # 创建可变字典副本
        processed = dict(metadata)
        
        # 处理 related_user_id
        user_id_field = processed.get("related_user_id")
        if user_id_field is not None:
            if isinstance(user_id_field, str):
                try:
                    processed["related_user_id"] = json.loads(user_id_field)
                except json.JSONDecodeError:
                    raise MetadataFormatError("related_user_id 必须是有效的JSON字符串")
            elif not isinstance(user_id_field, list):
                raise MetadataFormatError("related_user_id 必须是字符串或列表")
        
        # 处理 related_msg_id
        msg_id_field = processed.get("related_msg_id")
        if msg_id_field is not None:
            if isinstance(msg_id_field, str):
                try:
                    processed["related_msg_id"] = json.loads(msg_id_field)
                except json.JSONDecodeError:
                    raise MetadataFormatError("related_msg_id 必须是有效的JSON字符串")
            elif not isinstance(msg_id_field, list):
                raise MetadataFormatError("related_msg_id 必须是字符串或列表")
        
        return processed

    def validate_metadata_format(self) -> None:
        """验证metadata格式"""
        if not self.metadata:
            raise MetadataFormatError("数据缺少metadata")
        
        # 验证所有必需字段（基础字段）
        base_fields = {
            "type": str,
            "group_id": int,
            "related_user_id": list,
            "related_msg_id": list,
            "earliest_send_time": float,
            "latest_send_time": float,
            "message_count": int
        }
        
        # 检查基础字段
        for key, expected_type in base_fields.items():
            if key not in self.metadata:
                raise MetadataFormatError(f"数据缺少metadata字段: {key}")
            
            value = self.metadata[key]
            if not isinstance(value, expected_type):
                actual_type = type(value).__name__
                raise MetadataFormatError(
                    f"字段类型错误: {key} "
                    f"预期 {expected_type.__name__}, 实际 {actual_type}"
                )
        
        # 额外验证列表内容
        self._validate_list_contents("related_user_id", int)
        self._validate_list_contents("related_msg_id", int)
        
        # 验证业务逻辑
        msg_type = self.metadata["type"]
        if msg_type == "single":
            self._validate_single_metadata()
        elif msg_type == "chunk":
            self._validate_chunk_metadata()
        else:
            raise MetadataFormatError(f"无效的消息类型: {msg_type}")
        
        # 验证时间逻辑
        if self.metadata["earliest_send_time"] > self.metadata["latest_send_time"]:
            raise MetadataFormatError("最早发送时间不能晚于最晚发送时间")
        
        # single 类型下时间必须相等
        if msg_type == "single" and self.metadata["earliest_send_time"] != self.metadata["latest_send_time"]:
            raise MetadataFormatError("单条消息的最早和最晚发送时间必须相同")

    def _validate_list_contents(self, field: str, element_type: type) -> None:
        """验证列表字段的内容类型"""
        for i, item in enumerate(self.metadata[field]):
            if not isinstance(item, element_type):
                raise MetadataFormatError(
                    f"{field} 列表中第 {i} 项类型错误，"
                    f"预期 {element_type.__name__}, 实际 {type(item).__name__}"
                )

    def _validate_single_metadata(self) -> None:
        """验证 single 类型元数据"""
        # 检查互斥字段
        if "chunk_index" in self.metadata:
            raise MetadataFormatError("单条消息不应包含 chunk_index 字段")
        
        if "index" not in self.metadata:
            raise MetadataFormatError("单条消息必须包含 index 字段")
        
        if not isinstance(self.metadata["index"], int):
            raise MetadataFormatError("single 类型的 index 必须是整数")
        
        # 验证 message_count
        if self.metadata["message_count"] != 1:
            raise MetadataFormatError("单条消息的 message_count 必须为 1")

    def _validate_chunk_metadata(self) -> None:
        """验证 chunk 类型元数据"""
        # 检查互斥字段
        if "index" in self.metadata:
            raise MetadataFormatError("块消息不应包含 index 字段")
        
        if "chunk_index" not in self.metadata:
            raise MetadataFormatError("块消息必须包含 chunk_index 字段")

        if "is_locked" not in self.metadata:
            raise MetadataFormatError("块消息必须包含 is_locked 字段")

        if "merge_count" not in self.metadata:
            raise MetadataFormatError("块消息必须包含 merge_count 字段")
        
        if not isinstance(self.metadata["chunk_index"], int):
            raise MetadataFormatError("chunk 类型的 chunk_index 必须是整数")
        
        # 验证 message_count
        if self.metadata["message_count"] <= 0:
            raise MetadataFormatError("块消息的 message_count 必须大于 0")

    @classmethod
    def from_chroma_data(cls, data: ChromaData) -> 'ValidatedChromaData':
        """安全转换方法 (自动验证)"""
        if data.metadata is None:
            raise MetadataFormatError("数据缺少metadata")
        
        # 直接使用原始数据（只读映射），在__init__中处理转换
        return cls(
            metadata=data.metadata,
            id=data.id,
            document=data.document,
            embedding=data.embedding
        )

    # 便捷方法（现在直接返回列表）
    def get_related_user_ids(self) -> List[int]:
        """获取用户ID列表"""
        return self.metadata["related_user_id"]

    def get_related_msg_ids(self) -> List[int]:
        """获取消息ID列表"""
        return self.metadata["related_msg_id"]

    def to_raw_metadata(self) -> ChromaType.Metadata:
        """将验证后的元数据转回原始存储格式（列表→JSON字符串）"""
        # 创建符合 Metadata 类型的新字典
        raw_metadata: ChromaType.Metadata = {}
        
        # 复制所有基础字段（确保值类型正确）
        base_fields = [
            "type", "group_id", "earliest_send_time", 
            "latest_send_time", "message_count"
        ]
        for field in base_fields:
            raw_metadata[field] = self.metadata[field]
        
        # 特殊处理列表字段（转换为JSON字符串）
        raw_metadata["related_user_id"] = json.dumps(self.metadata["related_user_id"])
        raw_metadata["related_msg_id"] = json.dumps(self.metadata["related_msg_id"])
        
        # 处理类型特定字段
        if self.metadata["type"] == "single":
            raw_metadata["index"] = self.metadata["index"]
        else:  # "chunk"
            raw_metadata["chunk_index"] = self.metadata["chunk_index"]
        
        return raw_metadata

    def to_chroma_data(self) -> ChromaData:
        """将验证后的数据转回原始存储格式"""
        return ChromaData(
            metadata=self.to_raw_metadata(),
            id=self.id,
            document=self.document,
            embedding=self.embedding
        )



class GroupDataCache:
    GROUP_ID_KEY = "group_id"
    
    def __init__(self, datas: Optional[list[ChromaData]] = None) -> None:
        """初始化群聊数据缓存

        Args:
            datas: ChromaData对象列表，必须包含group_id且所有group_id相同（如果传入非空列表）
        """
        # 允许 datas 为 None 或 空列表 —— 此时创建空缓存而不是抛错
        if not datas:
            self.data_dict: dict[str, ValidatedChromaData] = {}
            self.data_list: list[ValidatedChromaData] = []
            self.group_id: Optional[int] = None # type: ignore
            return

        # 旧逻辑（datas 非空时保留原有验证/构建流程）
        self._validate_group_ids(datas)
        self.data_dict = self._build_data_cache(datas)
        self._overwrite_data_list_from_data_dict()

        # 此处 data_list 肯定非空
        self.group_id: int = self.data_list[0].metadata[self.GROUP_ID_KEY]
        
    def _validate_group_ids(self, datas: list['ChromaData']) -> None:
        """验证所有数据的group_id是否一致且存在"""
        if not datas or not datas[0].metadata or self.GROUP_ID_KEY not in datas[0].metadata:
            raise MetadataFormatError("群聊数据缺少group_id")
            
        expected_group_id = datas[0].metadata[self.GROUP_ID_KEY]
        seen_ids = set()
        
        for data in datas:
            if not data.metadata or self.GROUP_ID_KEY not in data.metadata:
                raise MetadataFormatError("群聊数据缺少group_id")
                
            if data.metadata[self.GROUP_ID_KEY] != expected_group_id:
                raise MetadataFormatError("群聊数据group_id不一致")
                
            if data.id in seen_ids:
                raise DatabaseConsistencyError("ID重复")
            seen_ids.add(data.id)


    
    def _build_data_cache(self, datas: list['ChromaData']) -> dict[str, ValidatedChromaData]:
        """构建数据缓存字典"""
        return {data.id: ValidatedChromaData.from_chroma_data(data) for data in datas}

    @classmethod
    async def from_db(cls, db: OlromaDBManager, group_collection: ChromaType.AsyncCollection|str, group_id: int) -> 'GroupDataCache':
        """从数据库中获取群聊数据并构建缓存

        Args:
            db: 数据库管理器
            group_collection: 群聊集合
            group_id: 群聊ID

        Returns:
            GroupDataCache: 群聊数据缓存对象
        """
        datas = await GroupDataCache._get_data_from_db(db, group_collection, group_id)
        return cls(datas)

    @staticmethod
    async def _get_single_list_from_db(
        db: OlromaDBManager, 
        group_collection: ChromaType.AsyncCollection|str,
        group_id: int, 
        sort: bool = True, 
        order: str = "asc",
        order_by: str = "index",
    ) -> list[ChromaData]:
        """
        从数据库中获取一个群的单条消息<br>
        只能提取前100条消息

        Args:
            group_id (int): 群ID
            sort (bool, optional): 是否排序. Defaults to True.
            order (str, optional): 排序顺序. Defaults to "asc".
            order_by (str, optional): 排序字段. Defaults to "index".
        Returns:
            list[ChromaData]: 群消息列表

        """
        chroma_data_list = []
        if order not in ["asc", "desc"]:
            raise ValueError("排序方式必须是asc或desc")
        
        # 获取记录并转换为 ChromaData 列表
        records = await db.get_records_from_collection(
            group_collection,
            where={"$and": [{"group_id": group_id}, {"type": "single"}]}
        )
        records = GroupDataCache._rusult_to_chromadata_list(records)

        # 检查元数据是否为空，并确保元数据包含排序字段
        if sort:
            valid_records = []
            for record in records:
                if record.metadata is None:
                    raise ValueError("单条消息元数据为空")
                if order_by not in record.metadata:
                    raise ValueError(f"单条消息元数据中不存在{order_by}字段")
                valid_records.append(record)
            
            # 对有效记录进行排序
            valid_records.sort(key=lambda x: x.metadata[order_by], reverse=False if order == "asc" else True)
            return valid_records
        
        return records

    @staticmethod
    async def _get_chunk_list_from_db(
        db: OlromaDBManager, 
        group_collection: ChromaType.AsyncCollection|str,
        group_id: int,
        sort: bool = True, 
        order: str = "asc",
        order_by: str = "chunk_index"
    ) -> list[ChromaData]:
        """
        从数据库中获取指定群组的未被锁定的块列表<br>
        只能提取前100条数据
        Args:
            group_id (int): 群组ID
            sort (bool, optional): 是否排序. Defaults to True.
            order (str, optional): 排序方式. Defaults to "asc".
            order_by (str, optional): 排序字段. Defaults to "index".
        Return: 
            list[ChromaData]: 块列表
        """
        chroma_data_list = []
        if order not in ["asc", "desc"]:
            raise ValueError("排序方式必须是asc或desc")
        
        # 获取记录并转换为 ChromaData 列表
        records = await db.get_records_from_collection(
            group_collection,
            where={"$and": [{"group_id": group_id}, {"type": "chunk"}, {"is_locked": False}]}
        )
        records = GroupDataCache._rusult_to_chromadata_list(records)

        # 检查元数据是否为空，并确保元数据包含排序字段
        if sort:
            valid_records = []
            for record in records:
                if record.metadata is None:
                    raise ValueError("单条消息元数据为空")
                if order_by not in record.metadata:
                    raise ValueError(f"单条消息元数据中不存在{order_by}字段")
                valid_records.append(record)
            
            # 对有效记录进行排序
            valid_records.sort(key=lambda x: x.metadata[order_by], reverse=False if order == "asc" else True)
            return valid_records
        
        return records

    @staticmethod
    async def _get_data_from_db(db: OlromaDBManager, group_collection: str|ChromaType.AsyncCollection, group_id: int) -> list[ChromaData]:
        """
        从数据库中获取数据列表
        """
        single_list = await GroupDataCache._get_single_list_from_db(db, group_collection, group_id, sort=True)
        chunk_list = await GroupDataCache._get_chunk_list_from_db(db, group_collection, group_id, sort=True)
        data_list = single_list + chunk_list
        return data_list

    @staticmethod
    def _rusult_to_chromadata_list(
        result: ChromaType.GetResult
    ) -> List['ChromaData']:
        """
        GetResult 转换为 list[ChromaData]。

        Args:
            result (QueryResult | GetResult): 查询或获取结果。

        Returns:
            List[ChromaData]: 转换后的 ChromaData 列表。
        """
        chroma_data_list = []

        ids = result["ids"]
        embeddings = result["embeddings"]
        documents = result["documents"]
        metadatas = result["metadatas"]

        for i in range(len(ids)):
            chroma_data = ChromaData()
            chroma_data.id = ids[i]
            chroma_data.embedding = embeddings[i] if embeddings is not None else None
            chroma_data.document = documents[i] if documents is not None else None
            chroma_data.metadata = metadatas[i] if metadatas is not None else None
            chroma_data_list.append(chroma_data)

        return chroma_data_list

    def _overwrite_data_list_from_data_dict(self):
        self.data_list = list(self.data_dict.values())
        self.sort_data_list()

    def _override_data_dict_from_data_list(self):
        self.data_dict = {data.id: data for data in self.data_list}
        
    @sync_timer
    def sort_data_list(self, single_order_by: str = "index", chunk_order_by: str = "chunk_index"):
        singles = self.get_singles()
        chunks = self.get_chunks()
        singles.sort(key=lambda x: x.metadata[single_order_by])
        chunks.sort(key=lambda x: x.metadata[chunk_order_by])
        self.data_list = chunks + singles

        


    def get_singles(
        self, 
        sort: bool = False, 
        order: str = "asc",
        order_by: str = "index"
    ) -> list[ValidatedChromaData]:
        """获取单条消息列表"""
        single_list: list[ValidatedChromaData] = []
        for data in self.data_list:
            if data.metadata["type"] == "single":
                single_list.append(data)

        if sort:
            single_list.sort(key=lambda x: x.metadata[order_by], reverse=False if order == "asc" else True)

        return single_list

    def get_chunks(
        self,
        sort: bool = False,
        order: str = "asc",
        order_by: str = "chunk_index"
    ) -> list[ValidatedChromaData]:
        """获取块列表"""
        chunk_list: list[ValidatedChromaData] = []
        for data in self.data_list:
            if data.metadata["type"] == "chunk":
                chunk_list.append(data)
        if sort:
            chunk_list.sort(key=lambda x: x.metadata[order_by], reverse=False if order == "asc" else True)
        return chunk_list

    def get_last_single_index(self) -> int:
        """获取最后一个单条消息的索引，缓存为空时返回0"""
        self.sort_data_list()
        singles = self.get_singles()
        if not singles:
            return 0
        
        last_data = singles[-1]
        if last_data.metadata["type"] != "single":
            return 0
        return last_data.metadata["index"]

    def get_last_chunk_index(self) -> int:
        """获取最后一个块的索引，缓存为空时返回0"""
        self.sort_data_list()
        chunks = self.get_chunks()
        if not chunks:
            return 0
        
        last_data = chunks[-1]
        if last_data.metadata["type"] != "chunk":
            return 0
        return last_data.metadata["chunk_index"]

    def get_data_list(self) -> list[ValidatedChromaData]:
        """获取数据列表的深拷贝"""
        return deepcopy(self.data_list)

    def get_data_dict(self) -> dict[str, ValidatedChromaData]:
        """获取数据字典的深拷贝"""
        return deepcopy(self.data_dict)

    def get(self, ids) -> list[ValidatedChromaData]:
        """获取指定ID的数据"""
        return [self.data_dict[id] for id in ids]

    def add(
        self, 
        data: ChromaData|ValidatedChromaData|None = None,
        id: Optional[str] = None,
        document: str|None = None,
        metadata: ChromaType.Metadata | None = None,
        embedding: ChromaType.Embedding|ChromaType.PyEmbedding|None = None
    ) -> str:
        """添加数据"""
        if data and (id or document or metadata or embedding):
            raise ValueError("data 和 id, document, metadata, embedding 不能同时存在")
        elif not metadata and not data:
            raise ValueError("metadata 不能为空")

        if not data:
            data = ChromaData(id=id, document=document, metadata=metadata, embedding=embedding)

        if isinstance(data, ChromaData):
            data = ValidatedChromaData.from_chroma_data(data)

        # 如果当前缓存还没有 group_id（即是空缓存），第一次添加时初始化 group_id
        if self.group_id is None:
            self.group_id = data.metadata["group_id"]
        else:
            if data.metadata["group_id"] != self.group_id:
                raise ValueError(f"数据ID {data.id} 的 group_id 不匹配\n预期为: {self.group_id}\n实际为: {data.metadata['group_id']}")

        if data.id in self.data_dict:
            raise ValueError(f"数据ID {data.id} 已存在")

        self.data_dict[data.id] = data
        self._overwrite_data_list_from_data_dict()

        return data.id

    def update(
        self, 
        data: Optional[ChromaData | ValidatedChromaData] = None,
        id: Optional[str] = None,
        document: str|None = None,
        metadata: ChromaType.Metadata | None = None,
        embedding: ChromaType.Embedding|ChromaType.PyEmbedding|None = None
    ):
        """更新数据"""
        if data and (id or document or metadata or embedding):
            raise ValueError("data 和 id, document, metadata, embedding 不能同时存在")
        elif not data and not id:
            raise ValueError("data 和 id 不能同时为空")

        # 处理 data 参数
        if data is not None:
            id = data.id
            document = data.document
            embedding = data.embedding
            
            # 关键修改：使用 ValidatedChromaData 的转换方法
            if isinstance(data, ValidatedChromaData):
                metadata = data.to_raw_metadata()
            else:  # ChromaData
                metadata = data.metadata

        if id not in self.data_dict:
            raise ValueError(f"数据ID {id} 不存在")

        # 更新 document
        if document is not None:
            self.data_dict[id].document = document

        # 更新 metadata
        if metadata is not None:
            # 用新的metadata创建临时验证对象
            temp_data = ValidatedChromaData.from_chroma_data(
                ChromaData(id=id, metadata=metadata)
            )
            if temp_data.metadata["group_id"] != self.group_id:
                raise ValueError(f"数据ID {id} 的 group_id 不匹配\n预期为: {self.group_id}\n实际为: {temp_data.metadata['group_id']}")
            self.data_dict[id].metadata = temp_data.metadata

        # 更新 embedding
        if embedding is not None:
            self.data_dict[id].embedding = embedding

        self._overwrite_data_list_from_data_dict()

    def add_or_update(
        self,
        data: Optional[ChromaData | ValidatedChromaData] = None,
        id: Optional[str] = None,
        document: str|None = None,
        metadata: ChromaType.Metadata | None = None,
        embedding: ChromaType.Embedding|ChromaType.PyEmbedding|None = None
    ):
        """添加或更新数据"""
        if data is not None and id is not None:
            raise ValueError("data 和 id 不能同时存在")

        if data is not None:
            id = data.id

        if id in self.data_dict:
            self.update(data=data, id=id, document=document, metadata=metadata, embedding=embedding)
        else:
            self.add(data=data, id=id, document=document, metadata=metadata, embedding=embedding)

    def delete(self, ids: list[str]|str):
        """删除指定ID的数据"""
        if isinstance(ids, str):
            ids = [ids]
        for id in ids:
            if id in self.data_dict:
                del self.data_dict[id]
        self._overwrite_data_list_from_data_dict()

    @staticmethod
    def _is_metadata_equal(meta1: MetadataFormat, meta2: MetadataFormat) -> bool:
        """精确比较两个元数据对象"""
        # 首先检查类型是否相同
        if meta1["type"] != meta2["type"]:
            return False
        
        # 检查所有键是否相同
        if set(meta1.keys()) != set(meta2.keys()):
            return False
        
        # 逐字段精确比较
        for key in meta1:
            # 特殊处理列表字段（排序后比较）
            if key in ["related_user_id", "related_msg_id"]:
                if sorted(meta1[key]) != sorted(meta2[key]):
                    return False
            # 精确比较其他字段
            elif meta1[key] != meta2[key]:
                return False
        
        return True

    @staticmethod
    def _is_data_equal(data1: ValidatedChromaData, data2: ValidatedChromaData) -> bool:
        """精确比较两个数据对象的所有字段"""
        # 1. 比较文档
        if (data1.document is None) != (data2.document is None):
            return False
        if data1.document is not None and data2.document is not None:
            if data1.document != data2.document:
                return False
        
        # 2. 比较元数据（使用现有方法）
        if not GroupDataCache._is_metadata_equal(data1.metadata, data2.metadata):
            return False
        
        # 3. 比较嵌入向量
        if (data1.embedding is None) != (data2.embedding is None):
            return False
        if data1.embedding is not None and data2.embedding is not None:
            # 假设是数值列表，逐元素比较（考虑浮点精度）
            if len(data1.embedding) != len(data2.embedding):
                return False
            for i in range(len(data1.embedding)):
                if abs(data1.embedding[i] - data2.embedding[i]) > 1e-9:
                    return False
        
        return True

    @async_timer
    async def sync_to_db(self, db: OlromaDBManager, group_collection: ChromaType.AsyncCollection|str, group_id: int):
        """将数据同步(推送)到数据库"""

        # 获取数据库中的数据
        db_data = await GroupDataCache.from_db(db, group_collection, group_id)
        db_data = db_data.get_data_dict()
        # 获取缓存中的数据
        cache_data = self.get_data_dict()

        # 计算需要添加、删除和更新的数据
        add_data = list(cache_data.keys() - db_data.keys())
        delete_data = list(db_data.keys() - cache_data.keys())
        id_same_data = list(db_data.keys() & cache_data.keys())

        update_data = []

        for id in id_same_data:
            if not self._is_data_equal(db_data[id], cache_data[id]):
                update_data.append(id)

        tasks = []

        if add_data:
            tasks.append(self._add_data_to_db(add_data, db, group_collection))

        if delete_data:
            tasks.append(self._delete_data_from_db(delete_data, db, group_collection))

        if update_data:
            tasks.append(self._update_data_in_db(update_data, db, group_collection))
        
        
        await asyncio.gather(*tasks)

    class DBDataDict(TypedDict):
        ids: list[str]
        documents: list[str|None]
        metadatas: list
        embeddings: list

    @staticmethod
    def datas_to_db_datas(datas: list[ChromaData]) -> DBDataDict:
        """将ChromaData列表转换为数据库数据"""
        ids = [chroma_data.id for chroma_data in datas]
        documents = [chroma_data.document for chroma_data in datas]
        metadatas = [chroma_data.metadata for chroma_data in datas]
        embeddings = [chroma_data.embedding for chroma_data in datas]

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings
        }

    @async_timer
    async def _add_data_to_db(self, ids: list[str], db: OlromaDBManager, group_collection: ChromaType.AsyncCollection|str):
        """
        将指定ID的数据添加到数据库
        不保证入库顺序
        """
        # 1. 获取原始数据
        datas = [self.data_dict[id].to_chroma_data() for id in ids]
        
        # 2. 按字段存在性分组 (key: (has_doc, has_meta, has_emb))
        groups = defaultdict(list)
        for data in datas:
            key = (
                data.document is not None,
                data.metadata is not None,
                data.embedding is not None
            )
            groups[key].append(data)
        
        # 3. 为每组创建异步任务
        tasks = []
        for (has_doc, has_meta, has_emb), group_datas in groups.items():
            # 构建当前组的参数
            ids_group = [d.id for d in group_datas]
            documents = [d.document for d in group_datas] if has_doc else None
            metadatas = [d.metadata for d in group_datas] if has_meta else None
            embeddings = [d.embedding for d in group_datas] if has_emb else None
            
            # 创建异步任务（不立即执行）
            task = db.add_records_to_collection(
                collection=group_collection,
                ids=ids_group,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            tasks.append(task)
        
        # 4. 并发执行所有任务
        await asyncio.gather(*tasks)

    @async_timer
    async def _delete_data_from_db(self, ids: list[str], db: OlromaDBManager, group_collection: ChromaType.AsyncCollection|str):
        """
        将指定ID的数据从数据库删除
        """
        await db.delete_records_from_collection(group_collection, ids)

    @staticmethod
    def metadata_to_dict(metadata: MetadataFormat) -> Dict[str, Optional[Union[str, int, float, bool]]]:
        """
        将 MetadataFormat 转换为 dict，其中：
        - 所有基本类型（int, float, bool）保留
        - list或dict类型转换为json字符串
        - 其他类型转换为字符串
        """
        result: Dict[str, Optional[Union[str, int, float, bool]]] = {}
        
        for key, value in metadata.items():
            if isinstance(value, (int, float, bool)) or value is None:
                # 保留基本类型
                result[key] = value
            elif isinstance(value, str):
                result[key] = value
            elif isinstance(value, list|dict):
                result[key] = json.dumps(value)
            else:
                result[key] = str(value)
        
        return result

    @async_timer
    async def _update_data_in_db(self, ids: list[str], db: OlromaDBManager, group_collection: ChromaType.AsyncCollection|str):
        datas = self.get_data_dict()
        
        # 按字段存在性分组 (key: (has_doc, has_meta, has_emb))
        groups = defaultdict(list)
        for id in ids:
            data = datas[id]
            key = (
                data.document is not None,
                data.metadata is not None,
                data.embedding is not None
            )
            groups[key].append(data)
        
        tasks = []
        for (has_doc, has_meta, has_emb), group_datas in groups.items():
            # 构建当前组的参数
            ids_group = [d.id for d in group_datas]
            documents = [d.document for d in group_datas] if has_doc else None
            metadatas = [d.to_raw_metadata() for d in group_datas] if has_meta else None
            embeddings = [d.embedding for d in group_datas] if has_emb else None
            
            tasks.append(db.update_records_in_collection(
                collection=group_collection,
                ids=ids_group,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            ))
        
        await asyncio.gather(*tasks)

class GroupRAGManager:
    MAX_BATCH_SIZE = 10000 # 每个群聊单次处理的最大消息数

    
    def __init__(self, oldb: OlromaDBManager, group_collection: ChromaType.AsyncCollection):
        self.olromadb = oldb
        self.group_collection = group_collection
        self.messages_dict: Dict[int, deque[GroupMessage]] = defaultdict(deque)
        self.allow_process_dict: Dict[int, bool] = defaultdict(lambda: True)
        # 不再需要 processing_dict
        # self.tasks 用于跟踪每个 group_id 对应的正在运行的处理任务
        self.tasks: Dict[int, asyncio.Task] = {}

    @classmethod
    async def init(cls, chromadb_url: str, ollama_url: str, model_name: str, tenant: str = DEFAULT_TENANT, group_collection_name: str = GROUP_MESSAGE_COLLECTION_NAME):
        db = await OlromaDBManager.init(chromadb_url=chromadb_url, ollama_url=ollama_url, model_name=model_name, tenant=tenant)
        group_collection = await db.get_or_create_collection(group_collection_name, metadata={"hnsw:space": "cosine"})
        return cls(db, group_collection)


    @async_timer
    async def add_messages(self, messages: list[GroupMessage]|GroupMessage):
        """将消息按 group_id 分组，放入对应队列，并确保处理任务正在运行。"""
        if isinstance(messages, GroupMessage):
            messages = [messages]
        if not messages:
            return

        # 按 group_id 对消息进行分组，以减少循环和字典访问次数
        grouped_messages = defaultdict(list)
        for msg in messages:
            grouped_messages[msg.group_id].append(msg)

        for group_id, msg_list in grouped_messages.items():
            # 将消息批量放入队列
            self.messages_dict[group_id].extend(msg_list)

            # 检查是否需要为该 group_id 启动一个新的处理任务
            # 只有当不存在任务或任务已意外结束时，才创建新任务
            task = self.tasks.get(group_id)
            if task is None or task.done():
                if self.allow_process_dict[group_id]:
                    # print(f"为 Group {group_id} 创建处理任务") # 调试信息
                    new_task = asyncio.create_task(self.process_message_queue(group_id))
                    self.tasks[group_id] = new_task


    async def search_messages(
        self,
        text : str,
        group_id: int|None = None,
        limit: int = 10,
    ):
        where = {}
        if group_id is not None:
            where["group_id"] = group_id
        
        """在数据库中搜索与给定文本最相似的记录排行。"""
        result = await self.olromadb.query_records_from_collection(
            self.group_collection,
            query_texts = text,
            n_results = limit,
        )

        chromadatas = query_result_to_full_results(result)

        return chromadatas




    async def process_message_queue(self, group_id: int):
        """
        作为单个群组的专用处理"工人"(Worker)。
        此任务会持续运行，直到处理完该群组队列中的所有消息。
        这保证了对同一个 group_id 的数据库操作是串行执行的。
        """
        # print(f"任务 {group_id} 开始处理队列") # 调试信息
        while self.allow_process_dict[group_id] and self.messages_dict[group_id]:
            # 从队列中取出一批待处理的消息
            messages_to_process = []
            count = 0
            while self.messages_dict[group_id] and count < self.MAX_BATCH_SIZE:
                messages_to_process.append(self.messages_dict[group_id].popleft())
                count += 1
            
            if not messages_to_process:
                continue

            # 实际处理数据，这个调用现在是安全的，因为每个 group_id 只有一个任务在执行
            try:
                await self._add_messages_to_group(messages_to_process)
            except Exception as e:
                logger.error(f"处理 Group {group_id} 的消息时发生错误: {e}")
                traceback.print_exc()
                # 出现错误时，可以选择将未处理的消息放回队列头部进行重试
                # self.messages_dict[group_id].extendleft(reversed(messages_to_process))
                break # 中断任务
        
        # print(f"任务 {group_id} 处理完毕，队列为空") # 调试信息


    async def wait_all_tasks(self):
        # 等待所有当前活动的任务完成
        running_tasks = [task for task in self.tasks.values() if not task.done()]
        if running_tasks:
            await asyncio.gather(*running_tasks)


    async def _get_last_single_index_from_db(self, group_id: int) -> Tuple[int, list[str]]:
        """获取最后一个单条消息的索引

            此方法限制同一数据库同一群聊内单条消息数量不超过100

            Args:
                group_id (int): 群聊ID

            Returns:
                Tuple[int, list]: 最后一个单条消息的索引, 以及所有单条消息的ID列表(按索引升序排列)
        """
        # 获取所有的单条消息
        singles = await self.olromadb.get_records_from_collection(
            self.group_collection,
            where={"$and":[{"type": "single"}, {"group_id": group_id}]}
        )

        ids = singles["ids"]
        metadatas = singles["metadatas"]
        if not metadatas:
            raise MetadataFormatError("单条消息的元数据为空")

        if len(singles["ids"]) != len(metadatas):
            raise DatabaseConsistencyError("ID与元数据数量不匹配")
        
        if len(singles["ids"]) == 0:
            return (0, [])

        

        
        # 创建(id, index)对的列表用于排序
        id_index_pairs = []
        last_index = 0
        
        for i, metadata in enumerate(metadatas):
            if "index" not in metadata:
                raise MetadataFormatError("单条消息的元数据中缺少索引")
            
            index = metadata["index"]
            if not isinstance(index, int):
                raise MetadataFormatError("单条消息的元数据中的索引不是整数")
            
            id_index_pairs.append((ids[i], index))
            last_index = max(index, last_index)
        
        # 按索引升序排序
        id_index_pairs.sort(key=lambda x: x[1])
        
        # 提取排序后的ids列表
        sorted_ids = [pair[0] for pair in id_index_pairs]

        
        return (last_index, sorted_ids)


    async def _get_last_chunk_index_from_db(self, group_id: int) -> Tuple[int, list[str]]:
        """获取最后一个块消息的索引

            此方法限制同一数据库同一群聊内块消息数量不超过100

            Args:
                group_id (int): 群聊ID

            Returns:
                Tuple[int, list]: 最后一个块消息的索引, 以及所有块消息的ID列表(按索引升序排列)
        """
        chunks = await self.olromadb.get_records_from_collection(
            self.group_collection,
            where={"$and":[{"type": "chunk"}, {"group_id": group_id}, {"is_locked": False}]}
        )
        
        if len(chunks["ids"]) >= 3:  # 获取到3个及以上的消息块是非预期行为
            raise DatabaseConsistencyError("块消息数量>3")
            

        
        ids = chunks["ids"]
        metadatas = chunks["metadatas"]
        
        if not metadatas:
            raise MetadataFormatError("块消息的元数据为空")

        if len(chunks["ids"]) != len(metadatas):
            raise DatabaseConsistencyError("ID与元数据数量不匹配")

        if len(chunks["ids"]) == 0:
            return (0, [])
        
        # 创建(id, index)对的列表用于排序
        id_index_pairs = []
        last_index = 0
        
        for i, metadata in enumerate(metadatas):
            if "chunk_index" not in metadata:
                raise MetadataFormatError("块消息的元数据中缺少索引")
            
            index = metadata["chunk_index"]
            if not isinstance(index, int):
                raise MetadataFormatError("块消息的元数据中的索引不是整数")
            
            id_index_pairs.append((ids[i], index))
            last_index = max(index, last_index)
        
        # 按索引升序排序
        id_index_pairs.sort(key=lambda x: x[1])
        
        # 提取排序后的ids列表
        sorted_ids = [pair[0] for pair in id_index_pairs]
        
        return (last_index, sorted_ids)

    async def _lock_chunk(self, id: str) -> None:
        """锁定一个块消息"""
        await self.olromadb.update_records_in_collection(
            collection=self.group_collection,
            ids=[id],
            metadatas=[{"is_locked": True}]
        )

    def _replace_placeholders(self, text: str, placeholders: dict[str, str]) -> str:
        """替换文本中的占位符。"""
        for ph, val in placeholders.items():
            text = text.replace(ph, str(val) if val is not None else "")
        return text

    def _rusult_to_chromadata_list(
        self,
        result: ChromaType.GetResult
    ) -> List['ChromaData']:
        """
        GetResult 转换为 list[ChromaData]。

        Args:
            result (QueryResult | GetResult): 查询或获取结果。

        Returns:
            List[ChromaData]: 转换后的 ChromaData 列表。
        """
        chroma_data_list = []

        ids = result["ids"]
        embeddings = result["embeddings"]
        documents = result["documents"]
        metadatas = result["metadatas"]

        for i in range(len(ids)):
            chroma_data = ChromaData()
            chroma_data.id = ids[i]
            chroma_data.embedding = embeddings[i] if embeddings is not None else None
            chroma_data.document = documents[i] if documents is not None else None
            chroma_data.metadata = metadatas[i] if metadatas is not None else None
            chroma_data_list.append(chroma_data)

        return chroma_data_list


    async def _get_single_list_from_db(
        self,
        group_id: int, 
        sort: bool = True, 
        order: str = "asc",
        order_by: str = "index",
    ) -> list[ChromaData]:
        """
        从数据库中获取一个群的单条消息<br>
        只能提取前100条消息

        Args:
            group_id (int): 群ID
            sort (bool, optional): 是否排序. Defaults to True.
            order (str, optional): 排序顺序. Defaults to "asc".
            order_by (str, optional): 排序字段. Defaults to "index".
        Returns:
            list[ChromaData]: 群消息列表

        """
        chroma_data_list = []
        if order not in ["asc", "desc"]:
            raise ValueError("排序方式必须是asc或desc")
        
        # 获取记录并转换为 ChromaData 列表
        records = await self.olromadb.get_records_from_collection(
            self.group_collection,
            where={"$and": [{"group_id": group_id}, {"type": "single"}]}
        )
        records = self._rusult_to_chromadata_list(records)

        # 检查元数据是否为空，并确保元数据包含排序字段
        if sort:
            valid_records = []
            for record in records:
                if record.metadata is None:
                    raise ValueError("单条消息元数据为空")
                if order_by not in record.metadata:
                    raise ValueError(f"单条消息元数据中不存在{order_by}字段")
                valid_records.append(record)
            
            # 对有效记录进行排序
            valid_records.sort(key=lambda x: x.metadata[order_by], reverse=False if order == "asc" else True)
            return valid_records
        
        return records

    async def _get_chunk_list_from_db(
        self, 
        group_id: int,
        sort: bool = True, 
        order: str = "asc",
        order_by: str = "chunk_index"
    ) -> list[ChromaData]:
        """
        从数据库中获取指定群组的未被锁定的块列表<br>
        只能提取前100条数据
        Args:
            group_id (int): 群组ID
            sort (bool, optional): 是否排序. Defaults to True.
            order (str, optional): 排序方式. Defaults to "asc".
            order_by (str, optional): 排序字段. Defaults to "index".
        Return: 
            list[ChromaData]: 块列表
        """
        chroma_data_list = []
        if order not in ["asc", "desc"]:
            raise ValueError("排序方式必须是asc或desc")
        
        # 获取记录并转换为 ChromaData 列表
        records = await self.olromadb.get_records_from_collection(
            self.group_collection,
            where={"$and": [{"group_id": group_id}, {"type": "chunk"}, {"is_locked": False}]}
        )
        records = self._rusult_to_chromadata_list(records)

        # 检查元数据是否为空，并确保元数据包含排序字段
        if sort:
            valid_records = []
            for record in records:
                if record.metadata is None:
                    raise ValueError("单条消息元数据为空")
                if order_by not in record.metadata:
                    raise ValueError(f"单条消息元数据中不存在{order_by}字段")
                valid_records.append(record)
            
            # 对有效记录进行排序
            valid_records.sort(key=lambda x: x.metadata[order_by], reverse=False if order == "asc" else True)
            return valid_records
        
        return records

    @async_timer
    async def _add_messages_to_group(self, messages: list[GroupMessage]):
        """添加消息到数据库（使用 GroupDataCache 作为本次请求的缓存）"""

        if not messages:
            return

        # 检查每条消息是否包含所有必需的属性
        required_attrs = ['group_id', 'user_id', 'user_name', 'msg_id', 'send_time', 'content']
        for i, message in enumerate(messages):
            missing_attrs = [attr for attr in required_attrs if not hasattr(message, attr)]
            if missing_attrs:
                raise ValueError(f"消息列表中第 {i+1} 条消息缺少必需的属性: {', '.join(missing_attrs)}")

        if not all(messages[0].group_id == message.group_id for message in messages):
            raise ValueError("消息列表中的消息组别不一致")

        group_id = messages[0].group_id

        # --- 只从 DB 读取一次，构建本次请求的缓存 ---
        memory_data = await GroupDataCache.from_db(self.olromadb, self.group_collection, group_id)

        for message in messages:
            # 取当前缓存中最后的单条消息索引和单条消息 id 列表（只算 single）
            last_index = memory_data.get_last_single_index()
            single_ids = [s.id for s in memory_data.get_singles(sort=True)]
            if last_index != len(single_ids):
                raise DatabaseConsistencyError("最后索引与检索到的记录长度不匹配, 可能是单条消息的索引不连续")
            index = last_index + 1

            # 生成单条消息 metadata（保持与原来兼容的格式 —— related_* 使用 json 字符串）
            metadata = {
                "type": "single",
                "index": index,
                "group_id": group_id,
                "related_user_id": json.dumps([message.user_id]),
                "related_msg_id": json.dumps([message.msg_id]),
                "earliest_send_time": float(message.send_time),
                "latest_send_time": float(message.send_time),
                "message_count": 1
            }

            # 格式化单条消息文本
            text = SINGLE_FORMAT
            date = datetime.fromtimestamp(message.send_time)
            placeholders = {
                "{$user_name}": message.user_name or "",
                "{$user_id}": message.user_id,
                "{$date}": date.strftime("%Y-%m-%d %H:%M:%S"),
                "{$content}": message.content,
                "{$year}": str(date.year),
                "{$month}": str(date.month),
                "{$day}": str(date.day),
                "{$hour}": str(date.hour),
                "{$minute}": str(date.minute),
                "{$second}": str(date.second),
            }
            text = self._replace_placeholders(text, placeholders)

            # 替换 CQ 码, 添加用户名
            def _(cq):
                if cq["CQ"] == "at":
                    if message.user_name:
                        return f"[CQ:at,qq={message.user_id},name={message.user_name}]"
                                
            text = replace_cq_codes(text, _)

            
            # 将单条消息加入缓存（不立即写 DB）
            new_id = memory_data.add(document=text, metadata=metadata)
            # memory_data.add 返回的是 id 字符串（不是列表），修复原来 add_id[0] 的错误
            single_ids.append(new_id)

            if len(single_ids) != index:
                raise DatabaseConsistencyError("单条消息的索引与记录长度不匹配, 可能是单条消息的索引不连续")

            # 如果达到了 MAX_SINGLE_NUMBER，则把这些 single 合并为一个 chunk —— 全部在缓存中操作
            if index == MAX_SINGLE_NUMBER:
                # 从缓存中取出这些单条消息对象
                singles_objs = memory_data.get(single_ids)
                if not singles_objs:
                    raise ValueError("从缓存中未能获取到单条消息对象")

                texts: list[str] = []
                related_user_id_set = set()
                related_msg_id_list = []
                earliest_send_time = float('inf')
                latest_send_time = 0.0
                message_count = MAX_SINGLE_NUMBER

                for s in singles_objs:
                    if s.document is None:
                        raise ValueError("单条消息的文档为空（缓存）")
                    texts.append(s.document)

                    meta = s.metadata
                    # 注意：ValidatedChromaData._process_metadata 已将 related_* 转为列表
                    ruids = meta["related_user_id"]
                    if not isinstance(ruids, list):
                        raise MetadataFormatError("single 的 related_user_id 在缓存中不是 list")
                    related_user_id_set.update(ruids)

                    rmsgids = meta["related_msg_id"]
                    if not isinstance(rmsgids, list):
                        raise MetadataFormatError("single 的 related_msg_id 在缓存中不是 list")
                    related_msg_id_list.extend(rmsgids)

                    _ear = meta["earliest_send_time"]
                    _lat = meta["latest_send_time"]
                    if not isinstance(_ear, float) or not isinstance(_lat, float):
                        raise MetadataFormatError("单条消息的时间字段类型不正确（缓存）")
                    earliest_send_time = min(earliest_send_time, _ear)
                    latest_send_time = max(latest_send_time, _lat)

                # 生成 chunk 的索引与元数据（在缓存中计算）
                last_chunk_index = memory_data.get_last_chunk_index()
                chunk_index = last_chunk_index + 1

                chunk_metadata = {
                    "type": "chunk",
                    "chunk_index": chunk_index,
                    "group_id": group_id,
                    "related_user_id": json.dumps(list(related_user_id_set)),
                    "related_msg_id": json.dumps(related_msg_id_list),
                    "earliest_send_time": earliest_send_time,
                    "latest_send_time": latest_send_time,
                    "message_count": message_count,
                    "is_max": False,
                    "is_locked": False,
                    "merge_count": 0
                }

                chunk_text = "\n".join(texts)
                chunk_id = memory_data.add(document=chunk_text, metadata=chunk_metadata)

                # 维护 chunk 列表（在缓存中），如果 chunk 数量达到 3，则锁定最旧的一个（在缓存中直接修改）
                chunks_objs = memory_data.get_chunks(sort=True)
                if len(chunks_objs) == 3:
                    oldest_chunk = chunks_objs[0]
                    # 直接在缓存对象上修改 is_locked 标志（最终会在 sync_to_db 时写回 DB）
                    if oldest_chunk.metadata["type"] != "chunk":
                        raise MetadataFormatError("缓存中 chunk 的 type 字段不是 chunk")
                    oldest_chunk.metadata["is_locked"] = True
                    # 保证缓存内部的一致性（overwrite 列表）
                    memory_data._overwrite_data_list_from_data_dict()

                # 删除原来的 single（缓存中删除）
                memory_data.delete(single_ids)

                # 清理 single_ids 准备下一轮（这里把 single_ids 重置为空）
                single_ids = []

        # --- 请求结束时，一次性同步缓存到 DB（添加/更新/删除） ---
        await memory_data.sync_to_db(self.olromadb, self.group_collection, group_id)
        # 不返回值，发生异常时上层会捕获

            
            # TODO: 添加块合并逻辑



async def main2():
    import time
    rag = await GroupRAGManager.init(
        "http://127.0.0.1:30004",
        "http://127.0.0.1:11434",
        "quentinz/bge-base-zh-v1.5:latest",
        group_collection_name="test_group_collection"
    )
    db = rag.olromadb
    # 重置集合
    if "test_group_collection" in await db.list_collections_name():
        await db.delete_collection("test_group_collection")
    rag.group_collection = await db.create_collection("test_group_collection", metadata={"hnsw:space": "cosine"})

    msg = GroupMessage(
        1, 2, 3, content="洪水"
    )
    await rag.add_messages(msg)
    await rag.wait_all_tasks()

    result = await rag.search_messages("洪水")

    print(result)
    
    input("press any key to continue...")
    await db.delete_collection("test_group_collection")
    await db.close()
    

if __name__ == "__main__":
    asyncio.run(main2())
