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

sys.path.append(str(Path(__file__).parent.parent))

from VectorDatabaseSystem import OlromaDBManager, MetadataFormatError, DatabaseConsistencyError, ChromaData, ChromeType, GroupMessage, PrivateMessage, async_timer, sync_timer, print_stats




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

class ValidatedChromaData:
    """已验证过metadata存在的ChromaData"""
    id: ChromeType.ID
    document: Optional[str]
    metadata: MetadataFormat
    embedding: Optional[ChromeType.Embedding|ChromeType.PyEmbedding]
    __slots__ = ('id', 'document', 'metadata', 'embedding')

    def __init__(
        self,
        metadata: ChromeType.Metadata,  # 接收只读的Mapping
        id: Optional[ChromeType.ID] = None,
        document: Optional[str] = None,
        embedding: Optional[ChromeType.Embedding|ChromeType.PyEmbedding] = None,
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

    def _process_metadata(self, metadata: ChromeType.Metadata) -> dict:
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

    def to_raw_metadata(self) -> ChromeType.Metadata:
        """将验证后的元数据转回原始存储格式（列表→JSON字符串）"""
        # 创建符合 Metadata 类型的新字典
        raw_metadata: ChromeType.Metadata = {}
        
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
    
    def __init__(self, datas: list[ChromaData]) -> None:
        """初始化群聊数据缓存
        
        Args:
            datas: ChromaData对象列表，必须包含group_id且所有group_id相同
            
        Raises:
            MetadataFormatError: 当数据缺少group_id或group_id不一致时
            DatabaseConsistencyError: 当存在重复ID时
            ValueError: 当输入数据为空时
        """
        if not datas:
            raise ValueError("输入数据不能为空")
            
        self._validate_group_ids(datas)
        self.data_dict = self._build_data_cache(datas)
        self._overwrite_data_list_from_data_dict()

        self.group_id: int = self.data_list[0].metadata[self.GROUP_ID_KEY]
        
    def _validate_group_ids(self, datas: list['ChromaData']) -> None:
        """验证所有数据的group_id是否一致且存在"""
        if not datas[0].metadata or self.GROUP_ID_KEY not in datas[0].metadata:
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
    async def from_db(cls, db: OlromaDBManager, group_collection: ChromeType.AsyncCollection|str, group_id: int) -> 'GroupDataCache':
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
        group_collection: ChromeType.AsyncCollection|str,
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
        group_collection: ChromeType.AsyncCollection|str,
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
    async def _get_data_from_db(db: OlromaDBManager, group_collection: str|ChromeType.AsyncCollection, group_id: int) -> list[ChromaData]:
        """
        从数据库中获取数据列表
        """
        single_list = await GroupDataCache._get_single_list_from_db(db, group_collection, group_id, sort=True)
        chunk_list = await GroupDataCache._get_chunk_list_from_db(db, group_collection, group_id, sort=True)
        data_list = single_list + chunk_list
        return data_list

    @staticmethod
    def _rusult_to_chromadata_list(
        result: ChromeType.GetResult
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
        """获取最后一个单条消息的索引"""
        self.sort_data_list()
        last_data = self.data_list[-1]
        if last_data.metadata["type"] != "single":
            return 0
        return last_data.metadata["index"]

    def get_last_chunk_index(self) -> int:
        """获取最后一个块的索引"""
        self.sort_data_list()
        last_data = self.data_list[-1]
        if last_data.metadata["type"] != "chunk":
            return 0
        return last_data.metadata["chunk_index"]

    def get_data_list(self) -> list[ValidatedChromaData]:
        """获取数据列表的深拷贝"""
        return deepcopy(self.data_list)

    def get_data_dict(self) -> dict[str, ValidatedChromaData]:
        """获取数据字典的深拷贝"""
        return deepcopy(self.data_dict)

    def add(
        self, 
        data: ChromaData|ValidatedChromaData|None,
        id: Optional[str] = None,
        document: str|None = None,
        metadata: ChromeType.Metadata | None = None,
        embedding: ChromeType.Embedding|ChromeType.PyEmbedding|None = None
    ):
        """添加数据"""
        if data and (id or document or metadata or embedding):
            raise ValueError("data 和 id, document, metadata, embedding 不能同时存在")
        elif not metadata:
            raise ValueError("metadata 不能为空")

        if not data:
            data = ChromaData(id=id, document=document, metadata=metadata, embedding=embedding)

        if isinstance(data, ChromaData):
            data = ValidatedChromaData.from_chroma_data(data)
        if data.id in self.data_dict:
            raise ValueError(f"数据ID {data.id} 已存在")

        self.data_dict[data.id] = data
        self._overwrite_data_list_from_data_dict()

    def update(
        self, 
        data: Optional[ChromaData | ValidatedChromaData] = None,
        id: Optional[str] = None,
        document: str|None = None,
        metadata: ChromeType.Metadata | None = None,
        embedding: ChromeType.Embedding|ChromeType.PyEmbedding|None = None
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
        metadata: ChromeType.Metadata | None = None,
        embedding: ChromeType.Embedding|ChromeType.PyEmbedding|None = None
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

    async def sync_to_db(self, db: OlromaDBManager, group_collection: ChromeType.AsyncCollection|str, group_id: int):
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

        tasks = [
            self._add_data_to_db(add_data, db, group_collection),
            self._delete_data_from_db(delete_data, db, group_collection),
            self._update_data_in_db(update_data, db, group_collection)
        ]
        
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

    async def _add_data_to_db(self, ids: list[str], db: OlromaDBManager, group_collection: ChromeType.AsyncCollection|str):
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

    async def _delete_data_from_db(self, ids: list[str], db: OlromaDBManager, group_collection: ChromeType.AsyncCollection|str):
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

    async def _update_data_in_db(self, ids: list[str], db: OlromaDBManager, group_collection: ChromeType.AsyncCollection|str):
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
    def __init__(self, oldb: OlromaDBManager, group_collection: ChromeType.AsyncCollection):
        self.olromadb = oldb
        self.group_collection = group_collection

    @classmethod
    async def init(cls, chromadb_url: str, ollama_url: str, model_name: str, tenant: str = DEFAULT_TENANT, group_collection_name: str = GROUP_MESSAGE_COLLECTION_NAME):
        db = await OlromaDBManager.init(chromadb_url=chromadb_url, ollama_url=ollama_url, model_name=model_name, tenant=tenant)
        group_collection = await db.get_or_create_collection(group_collection_name, metadata={"hnsw:space": "cosine"})
        return cls(db, group_collection)

    @async_timer
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

    @async_timer
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

    @async_timer
    async def _lock_chunk(self, id: str) -> None:
        """锁定一个块消息"""
        await self.olromadb.update_records_in_collection(
            collection=self.group_collection,
            ids=[id],
            metadatas=[{"is_locked": True}]
        )

    @sync_timer
    def _replace_placeholders(self, text: str, placeholders: dict[str, str]) -> str:
        """替换文本中的占位符。"""
        for ph, val in placeholders.items():
            text = text.replace(ph, str(val) if val is not None else "")
        return text

    def _rusult_to_chromadata_list(
        self,
        result: ChromeType.GetResult
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
    async def _add_messages_to_group(self, messages:list[GroupMessage]):
        """添加单条消息到数据库"""

        if not all(messages[0].group_id == message.group_id for message in messages):
            raise ValueError("消息列表中的消息组别不一致")

        group_id = messages[0].group_id

        memory_data = await GroupDataCache.from_db(self.olromadb, self.group_collection, group_id)
        
        for message in messages: 
            last_index, single_ids = await self._get_last_single_index_from_db(message.group_id)
            if last_index != len(single_ids):
                raise DatabaseConsistencyError("最后索引与检索到的记录长度不匹配, 可能是单条消息的索引不连续")
            index = last_index + 1
            metadata = {
                "type": "single",
                "index": index,
                "group_id": group_id,
                "related_user_id": json.dumps([message.user_id]), # chromadb不允许元数据的值为列表, 使用字符串代替
                "related_msg_id": json.dumps([message.msg_id]),
                "earliest_send_time": float(message.send_time),
                "latest_send_time": float(message.send_time),
                "message_count": 1
            }


            # 格式化单条消息内容
            text = SINGLE_FORMAT
            date = datetime.fromtimestamp(message.send_time)
            placeholders = {
                "{$user_name}": message.user_name,
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
            if message.user_name:
                placeholders["{$user_name}"] = message.user_name
            else:
                placeholders["{$user_name}"] = "" # 移除{$user_name}占位符
            text = self._replace_placeholders(text, placeholders)

            # 添加单条消息到数据库
            add_id = await self.olromadb.add_records_to_collection(self.group_collection, documents=[text], metadatas=[metadata]) # 此时index=单条消息数量
            single_ids.append(add_id[0])
            if len(single_ids) != index:
                raise DatabaseConsistencyError("单条消息的索引与记录长度不匹配, 可能是单条消息的索引不连续")



            if index == MAX_SINGLE_NUMBER:
                texts: list[str] = [] # 所有的单条消息
                related_user_id = set() # 所有单条消息的发送者
                related_msg_id = [] # 所有单条消息的ID
                earliest_send_time = float('inf') # 最早发送时间
                latest_send_time = 0.0 # 最晚发送时间
                message_count = MAX_SINGLE_NUMBER # 单条消息数量
                # 获取所有单条消息
                result = await self.olromadb.get_records_from_collection(self.group_collection, ids=single_ids)

                if not result["documents"]:
                    raise ValueError("单条消息的文档为空")
                for text in result["documents"]:
                    texts.append(text)
                
                if not result["metadatas"]:
                    raise MetadataFormatError("单条消息的元数据为空")
                for metadata in result["metadatas"]:
                    # 记录消息ID
                    _related_msg_id = metadata["related_msg_id"]
                    if not isinstance(_related_msg_id, str):
                        raise MetadataFormatError("单条消息的元数据中的related_msg_id不是字符串")
                    related_msg_id.extend(json.loads(_related_msg_id)) # chromadb不允许元数据的值为列表, 使用字符串代替
                    # 记录用户ID
                    _related_user_id = metadata["related_user_id"]
                    if not isinstance(_related_user_id, str):
                        raise MetadataFormatError("单条消息的元数据中的related_user_id不是字符串")
                    related_user_id.add(json.loads(_related_user_id)[0]) # chromadb不允许元数据的值为列表, 使用字符串代替
                    # 记录最早发送时间
                    _earliest_send_time = metadata["earliest_send_time"]
                    if not isinstance(_earliest_send_time, float):
                        raise MetadataFormatError("单条消息的元数据中的earliest_send_time不是浮点数")
                    earliest_send_time = min(earliest_send_time, _earliest_send_time)
                    # 记录最晚发送时间
                    _latest_send_time = metadata["latest_send_time"]
                    if not isinstance(_latest_send_time, float):
                        raise MetadataFormatError("单条消息的元数据中的latest_send_time不是浮点数")
                    latest_send_time = max(latest_send_time, _latest_send_time)

                last_index, chunk_ids = await self._get_last_chunk_index_from_db(message.group_id)
                index = last_index + 1

                # 生成消息块元数据
                metadata = {
                    "type": "chunk",
                    "chunk_index": index,
                    "group_id": group_id,
                    "related_user_id": json.dumps(list(related_user_id)),
                    "related_msg_id": json.dumps(related_msg_id),
                    "earliest_send_time": earliest_send_time,
                    "latest_send_time": latest_send_time,
                    "message_count": message_count,
                    "is_max": False,
                    "is_locked": False,
                    "merge_count": 0
                }
                # 生成消息文本
                text = "\n".join(texts)
                # 添加消息块到数据库
                add_id = await self.olromadb.add_records_to_collection(self.group_collection, documents=[text], metadatas=[metadata])
                chunk_ids.append(add_id[0])
                # 锁定前面第2个消息块
                if len(chunk_ids) == 3:
                    await self._lock_chunk(chunk_ids[0]) 
                # 删除原来的单条消息
                await self.olromadb.delete_records_from_collection(self.group_collection, ids=single_ids)
            
            # TODO: 添加块合并逻辑
