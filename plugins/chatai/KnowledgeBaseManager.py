import asyncio
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
import traceback
import uuid
from typing import List, Literal, Optional, Dict, Any, Sequence, Set, TypedDict, Union, overload
from dataclasses import dataclass, field
import chromadb
from copy import deepcopy
import logging as loger

from pydantic import BaseModel, Field, ValidationError


dir_path = Path(__file__).parent
GLOBAL_GROUP_ID = -1

sys.path.append(str(dir_path))

from VectorDatabaseSystem import OpenAIChromaDBManager, ChromaType
from chatgpt import ChatGPT, OneMessage

@dataclass
class KBMetadata:
    """知识库条目的元数据结构"""
    type: Literal["global", "group"]
    group_id: int
    creation_time: float
    last_called_time: float


  
    @staticmethod
    def from_metadata(metadata: chromadb.Metadata) -> "KBMetadata":
        """
        从标准元数据构造 KBMetadata 实例，确保输入端元数据拥有所需的字段且类型正确
      
        Args:
            meta 标准的 Metadata 字典
          
        Returns:
            KBMetadata 实例
          
        Raises:
            ValueError: 当必需字段缺失或类型不正确时
        """
        if metadata is None:
            raise ValueError("Metadata cannot be None")
      
        # 验证和转换 type 字段
        type_value = metadata.get("type")
        if type_value is None:
            raise ValueError("Missing required field 'type' in metadata")
        if not isinstance(type_value, str):
            raise ValueError(f"Field 'type' must be str, got {type(type_value)}")

      
        # 验证和转换 group_id 字段
        group_id_value = metadata.get("group_id")
        if group_id_value is None:
            raise ValueError("Missing required field 'group_id' in metadata")
        if not isinstance(group_id_value, int):
            raise ValueError(f"Field 'group_id' must be int, got {type(group_id_value)}")
      
        # 验证和转换 creation_time 字段
        creation_time_value = metadata.get("creation_time")
        if creation_time_value is None:
            raise ValueError("Missing required field 'creation_time' in metadata")
        if not isinstance(creation_time_value, (int, float)):
            raise ValueError(f"Field 'creation_time' must be int or float, got {type(creation_time_value)}")
        creation_time_value = float(creation_time_value)
      
        # 验证和转换 last_called_time 字段
        last_called_time_value = metadata.get("last_called_time")
        if last_called_time_value is None:
            raise ValueError("Missing required field 'last_called_time' in metadata")
        if not isinstance(last_called_time_value, (int, float)):
            raise ValueError(f"Field 'last_called_time' must be int or float, got {type(last_called_time_value)}")
        last_called_time_value = float(last_called_time_value)

        if type_value not in ["global", "group"]:
            raise ValueError(f"Field 'type' must be 'global' or 'group', got {type_value}")

        return KBMetadata(
            type=type_value, # type: ignore
            group_id=group_id_value,
            creation_time=creation_time_value,
            last_called_time=last_called_time_value,
        )
  
    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        """
        将 KBMetadata 转换为标准字典格式
      
        Returns:
            包含所有元数据字段的字典
        """
        return {
            "type": self.type,
            "group_id": self.group_id,
            "creation_time": self.creation_time,
            "last_called_time": self.last_called_time,
        }



@dataclass
class KBEntry:
    """代表知识库中的一条数据"""
    id: str
    document: str
    metadata: KBMetadata



ACTION_MARK = "[ACTION]"
THINKING_MARK = "[THINKING]"
add_parameters = ["document", "type"]
delete_parameters = ["id"]
update_parameters = ["id"]
all_optional_update_parameters = ["id", "document", "type", "group_id"]




class KnowledgeBaseManager:
    """
    一个支持批量操作的知识库管理类，使用 OpenAIChromaDBManager 作为后端。
    （使用 ChromaType 进行类型注解）
    """
    def __init__(self, db_manager: OpenAIChromaDBManager, collection:  ChromaType.AsyncCollection):
        self.db_manager: OpenAIChromaDBManager = db_manager
        self.collection: str | ChromaType.AsyncCollection = collection

    @classmethod
    async def create(
        cls,
        collection: str | ChromaType.AsyncCollection = "knowledge_base",
        chromadb_url: str = "http://127.0.0.1:8000",
        openai_api_key: str = "",
        openai_base_url: str = "https://api.openai.com/v1  ",
        tenant: str = "default_tenant",
        model_name: str = "text-embedding-ada-002",
        max_concurrent_requests: int = 50
    ) -> "KnowledgeBaseManager":
        """
        异步类方法，通过提供配置直接构建并初始化一个 KnowledgeBaseManager 实例。
        """
        # 1. 异步创建底层的数据库管理器
        db_manager = await OpenAIChromaDBManager.init(
            chromadb_url=chromadb_url,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            tenant=tenant,
            model_name=model_name,
            max_concurrent_requests=max_concurrent_requests
        )
      
        # 2. 使用标准的 __init__ 方法创建 KnowledgeBaseManager 实例
        if isinstance(collection, str):
            collection = await db_manager.get_or_create_collection(collection)
        
        instance = cls(db_manager=db_manager, collection=collection)
      
        # 3. 确保集合存在，完成初始化流程
        await instance.initialize()
      
        # 4. 返回完全就绪的实例
        return instance

    async def initialize(self):
        if isinstance(self.collection, str):
            self.collection = await self.db_manager.get_or_create_collection(self.collection)


    async def add_entry(
        self, 
        document_or_entries: Union[str, KBEntry, List[KBEntry]], 
        entry_type: Optional[Literal["global", "group"]] = None, 
        group_id: int = GLOBAL_GROUP_ID,
        id: Optional[str] = None
    ) -> Union[str, List[str]]:
        """
        添加知识库条目，支持单条和批量操作，允许自定义ID

        Args:
            document_or_entries: 
                - 单条时：文档内容字符串 或 KBEntry 实例
                - 批量时：KBEntry 列表
            entry_type: 条目类型（仅单条 str 时需要）
            group_id: 群组ID（仅单条 str 时需要）
            id: 自定义ID（仅单条 str 时需要）
          
        Returns:
            - 单条时：添加的条目ID
            - 批量时：成功添加的条目ID列表
        """
      
        # 处理批量添加情况（List[KBEntry]）
        if isinstance(document_or_entries, list):
            entries: List[KBEntry] = document_or_entries
            if not entries:
                return []
              
            ids = [entry.id for entry in entries]
            documents = [entry.document for entry in entries]
            metadatas = [entry.metadata for entry in entries]

            dict_metadatas: list[ChromaType.Metadata] = [metadata.to_dict() for metadata in metadatas]

            await self.db_manager.add_records_to_collection(
                collection=self.collection,
                documents=documents,
                ids=ids,
                metadatas=dict_metadatas
            )
            return ids

        # 处理单条添加情况（str 或 KBEntry）
        else:
            if isinstance(document_or_entries, KBEntry):
                entry = document_or_entries
                await self.db_manager.add_records_to_collection(
                    collection=self.collection,
                    documents=[entry.document],
                    ids=[entry.id],
                    metadatas=[entry.metadata] # type: ignore
                )
                return entry.id
            else:
                # 原始字符串输入方式
                document = document_or_entries
                if entry_type is None:
                    raise ValueError("单条添加时必须提供 entry_type 参数")

                if entry_type not in ["global", "group"]:
                    raise ValueError("条目类型 'type' 必须是 'global' 或 'group'")
              
                if entry_type == "global":
                    group_id = GLOBAL_GROUP_ID

                current_time = int(time.time())
                metadata = KBMetadata(
                    type=entry_type,
                    group_id=group_id,
                    creation_time=current_time,
                    last_called_time=current_time,
                )
              
                entry_id = id or uuid.uuid4().hex
              
                await self.db_manager.add_records_to_collection(
                    collection=self.collection,
                    documents=[document],
                    ids=[entry_id],
                    metadatas=[metadata]  # type: ignore
                )
                return entry_id

    async def delete_entry(
        self, 
        entry_id: Union[str, List[str]]
    ) -> None:
        """
        删除知识库条目，支持单条和批量操作
      
        Args:
            entry_id: 单个ID字符串或ID列表
        """
        ids = [entry_id] if isinstance(entry_id, str) else entry_id
        if not ids:
            return
      
        await self.db_manager.delete_records_from_collection(
            collection=self.collection,
            ids=ids
        )

    async def update_entry(
        self, 
        *,
        document: Optional[str] = None,
        entry_type: Optional[Literal["global", "group"]] = None,
        group_id: Optional[int] = None,
        id: Optional[str] = None,
        entries: Optional[Union[KBEntry, List[KBEntry]]] = None
    ) -> None:
        """
        更新知识库条目，支持多种参数形式
      
        支持以下调用方式：
        1. 通过提供单个条目ID和更新字段
        2. 通过提供KBEntry对象或KBEntry对象列表
      
        Args:
            document: 新的文档内容（仅当更新单个条目时使用）
            entry_type: 新的条目类型（仅当更新单个条目时使用）
            group_id: 新的群组ID（仅当更新单个条目时使用）
            id: 要更新的条目ID（仅当更新单个条目时使用）
            entries: KBEntry对象或KBEntry对象列表（用于批量更新）
          
        Raises:
            ValueError: 当参数组合不正确时
        """
        # 处理批量更新情况
        if entries is not None:
            entries_list = entries if isinstance(entries, list) else [entries]
            if not entries_list:
                return
              
            ids = [entry.id for entry in entries_list]
            documents = [entry.document for entry in entries_list]
            metadatas: list = [entry.metadata.to_dict() for entry in entries_list]
          
            await self.db_manager.update_records_in_collection(
                collection=self.collection,
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            return
      
        # 处理单个条目更新
        if id is None:
            raise ValueError("必须提供要更新的条目ID")
      
        # 获取现有条目
        existing_entries = await self.get_entry_by_id(id)
        if not existing_entries or existing_entries[0] is None:
            raise ValueError(f"找不到ID为 {id} 的条目")
      
        existing_entry = existing_entries[0]
      
        # 创建更新后的元数据
        new_metadata = deepcopy(existing_entry.metadata)
        if entry_type is not None:
            if entry_type not in ["global", "group"]:
                raise ValueError("条目类型 'type' 必须是 'global' 或 'group'")
            new_metadata.type = entry_type
            # 如果类型改为global，则重置group_id
            if entry_type == "global":
                new_metadata.group_id = GLOBAL_GROUP_ID
      
        if group_id is not None:
            if new_metadata.type == "global":
                raise ValueError("全局条目的group_id不能修改")
            new_metadata.group_id = group_id
      
        # 准备更新数据
        new_document = document if document is not None else existing_entry.document
      
        await self.db_manager.update_records_in_collection(
            collection=self.collection,
            ids=[id],
            documents=[new_document],
            metadatas=[new_metadata.to_dict()]
        )

    async def get_entry_by_id(
        self, 
        entry_id: Union[str, List[str]]
    ) -> List[Optional[KBEntry]]:
        """
        获取知识库条目，支持单条和批量操作
      
        Args:
            entry_id: 单个ID字符串或ID列表
          
        Returns:
            总是返回 KBEntry 列表（可能包含 None）
        """
        ids = [entry_id] if isinstance(entry_id, str) else entry_id
        if not ids:
            return []

        # 从数据库获取结果
        result: ChromaType.GetResult = await self.db_manager.get_records_from_collection(
            collection=self.collection,
            ids=ids
        )

        # 构建ID到结果的映射
        id_to_result = {}
        if result['ids']:
            for i, doc_id in enumerate(result['ids']):
                # 获取metadata并创建KBMetadata实例
                raw_metadata = result["metadatas"][i] if result["metadatas"] and i < len(result["metadatas"]) else None
                if raw_metadata is None:
                    id_to_result[doc_id] = None
                    continue

                metadata = KBMetadata.from_metadata(raw_metadata)

                try:
                    id_to_result[doc_id] = KBEntry(
                        id=doc_id,
                        document=result["documents"][i] if result["documents"] and i < len(result["documents"]) else "",
                        metadata=metadata
                    )
                except IndexError:
                    id_to_result[doc_id] = None


        # 按原始ID顺序返回结果（保留 None）
        return [id_to_result.get(entry_id) for entry_id in ids]


    async def find_similar_entries(
        self,
        query: str,
        n_results: int = 5,
        entry_type: Optional[str] = None,
        group_id: Optional[int] = None
    ) -> List[KBEntry]:
        """
        根据输入字符串查找相似的文档，并返回 KBEntry 列表。

        业务逻辑说明：
        - 当 entry_type="global" 时：只返回全局知识（type="global"）
        - 当 entry_type="group" 且 group_id 有效时：返回该群组知识 + 全局知识
        - 当只提供 group_id 时：返回该群组知识 + 全局知识
        - 其他情况：不应用类型过滤

        Args:
            query: 查询文本
            n_results: 返回结果数量，默认为5
            entry_type: 可选，过滤类型（"global" 或 "group"）
            group_id: 可选，群组ID

        Returns:
            匹配到的 KBEntry 列表
        """
        # 构建过滤条件
        where_filter: Dict[str, Any] | None = self._build_query_filter(entry_type, group_id)


        # 执行查询
        results: ChromaType.QueryResult = await self.db_manager.query_records_from_collection(
            collection=self.collection,
            query_texts=query,
            n_results=n_results,
            wheres=where_filter
        )

        # 处理空结果情况
        if not results['ids'] or not results['ids'][0]:
            return []

        # 提取查询结果
        found_ids = results['ids'][0]
        found_metadatas = results['metadatas'][0] if results['metadatas'] and results['metadatas'][0] else []
        found_documents = results['documents'][0] if results['documents'] and results['documents'][0] else []
        found_distances = results['distances'][0] if results['distances'] and results['distances'][0] else []

        # 确保元数据与ID长度一致（防御性处理）
        while len(found_metadatas) < len(found_ids):
            found_metadatas.append({})

        current_time = int(time.time())
        entries_to_return: List[KBEntry] = []
        ids_to_update: List[str] = []
        metadatas_to_update: List[ChromaType.Metadata] = []


        for i, entry_id in enumerate(found_ids):
            try:
                raw_metadata = found_metadatas[i] if i < len(found_metadatas) else {}
                document = found_documents[i] if i < len(found_documents) else ""
                
                # 更新 last_called_time
                updated_metadata_dict = {**raw_metadata, "last_called_time": current_time}
                
                # 构造 KBMetadata 实例
                kb_metadata = KBMetadata.from_metadata(updated_metadata_dict)

                # 构造 KBEntry 实例
                kb_entry = KBEntry(
                    id=entry_id,
                    document=document,
                    metadata=kb_metadata
                )

                entries_to_return.append(kb_entry)

                # 准备更新元数据
                ids_to_update.append(entry_id)
                metadatas_to_update.append(updated_metadata_dict)

            except Exception as e:
                loger.warning(f"Failed to process entry {entry_id}: {e}")
                continue

        # 批量更新访问时间
        if ids_to_update:
            await self.db_manager.update_records_in_collection(
                collection=self.collection,
                ids=ids_to_update,
                metadatas=metadatas_to_update,
                use_openai=False
            )

        return entries_to_return

    def _build_query_filter(
        self,
        entry_type: Optional[str],
        group_id: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """
        构建查询过滤条件
      
        Args:
            entry_type: 可选，过滤类型（"global" 或 "group"）
            group_id: 可选，群组ID
          
        Returns:
            构建好的过滤条件字典，或None（表示无过滤）
        """
        # 场景1: 明确查询全局知识
        if entry_type == "global":
            return {"type": "global"}
      
        # 场景2: 查询特定群组知识（包括全局知识）
        if entry_type == "group" and group_id is not None:
            return {
                "$or": [
                    {"group_id": group_id},
                    {"type": "global"}
                ]
            }
      
        # 场景3: 只提供group_id，查询该群组知识（包括全局知识）
        if group_id is not None:
            return {
                "$or": [
                    {"group_id": group_id},
                    {"type": "global"}
                ]
            }
      
        # 场景4: 无特殊过滤条件
        return None


class KnowledgeBaseCache:
    def __init__(
        self, 
        kb_manager: KnowledgeBaseManager,
        existing_entries: Optional[list[KBEntry]] = None
    ):
        self.kb_manager = kb_manager

      
      
        self.data_list: list[KBEntry] = existing_entries if existing_entries is not None else []
        self.data_dict: dict[str, KBEntry] = {}
        self._overwrite_dict_based_on_list()

        self.synced_data_list = deepcopy(self.data_list)
        self.synced_data_dict = deepcopy(self.data_dict)


    def _overwrite_list_based_on_dict(self):
        """
        覆写data_list，使其与data_dict保持一致
        """
        self.data_list = list(self.data_dict.values())

    def _overwrite_dict_based_on_list(self):
        """
        覆写data_dict，使其与data_list保持一致
        """
        self.data_dict = {entry.id: entry for entry in self.data_list}

    def add(
        self,
        entry: Optional[KBEntry] = None,
        document: Optional[str] = None, 
        entry_type: Optional[str] = None, 
        group_id: Optional[int] = None, 
        id: Optional[str] = None        
    ) -> str:                           
        """
        向知识库中添加一个新的条目。

        该方法提供两种添加条目的方式：
        1. 直接提供一个完整的 KBEntry 对象
        2. 提供构建条目所需的各个参数（document, entry_type, group_id）

        Args:
            entry (Optional[KBEntry]): 可选，完整的知识库条目对象。如果提供此参数，
                则不能同时提供 document、entry_type 和 group_id 参数。
            document (Optional[str]): 可选，条目的文档内容。当不提供 entry 时必须提供。
            entry_type (Optional[str]): 可选，条目的类型。当不提供 entry 时必须提供。
            group_id (Optional[int]): 可选，条目所属的组ID。当不提供 entry 时必须提供。
            id (Optional[str]): 可选，条目的唯一标识符。如果不提供，将自动生成一个UUID。

        Returns:
            str: 返回添加的条目的ID。

        Raises:
            ValueError: 在以下情况下抛出异常：
                - 同时提供了 entry 和其他参数（document, entry_type, group_id）
                - 既没有提供 entry，也没有提供完整的必需参数（document, entry_type, group_id）
                - 在使用参数构建条目时，缺少必需的参数（document, entry_type 或 group_id）
                - 要添加的条目ID已存在于缓存中
        """

        if entry and (document or entry_type or group_id):
            raise ValueError("entry 不能与 document, entry_type, group_id 同时存在")
        
        # LLM may not provide group_id for global entries, so we handle it here.
        if not entry and document and entry_type:
            if entry_type == "global" and group_id is None:
                group_id = GLOBAL_GROUP_ID
            elif entry_type == "group" and group_id is None:
                 raise ValueError("group_id is required for group entries")
        elif not entry and not (document and entry_type and group_id is not None):
            raise ValueError("entry 与 document, entry_type, group_id 必须存在一个")

        if not entry:
            if not document:
                raise ValueError("document 不能为空")
            if not entry_type:
                raise ValueError("entry_type 不能为空")
            if entry_type not in ["global", "group"]:
                raise ValueError("entry_type 只能是 global 或 group")
            if group_id is None:
                raise ValueError("group_id 不能为空")
            if not id:
                id = str(uuid.uuid4().hex)
            entry = KBEntry(
                document=document,
                id = id,
                metadata = KBMetadata(
                    type=entry_type, # type: ignore
                    group_id=group_id,
                    creation_time=int(time.time()),
                    last_called_time=int(time.time())
                )
            )
      
        if entry.id in self.data_dict:
            raise ValueError(f"Entry with id {entry.id} already exists in the cache.")

        self.data_list.append(entry)
        self.data_dict[entry.id] = entry

        return entry.id

    def delete(self, id: str):
        """
        从缓存中删除指定ID的数据条目。
      
        参数:
            id (str): 要删除的数据条目的唯一标识符
          
        异常:
            ValueError: 当指定的ID在缓存中不存在时抛出
          
        说明:
            该方法会从数据字典中删除指定ID的条目，并调用内部方法
            _overwrite_list_based_on_dict()来更新列表结构，保持数据一致性
        """
        if id not in self.data_dict:
            raise ValueError(f"Entry with id {id} does not exist in the cache.")

        del self.data_dict[id]
        self._overwrite_list_based_on_dict()

    def get(self, id: str) -> KBEntry:
        """
        根据给定的ID获取知识库条目。

        Args:
            id (str): 要查找的知识库条目的唯一标识符。

        Returns:
            KBEntry: 与给定ID关联的知识库条目。

        Raises:
            KeyError: 如果给定的ID在知识库中不存在。
        """
        return self.data_dict[id]

    def get_all(self) -> list[KBEntry]:
        """
        获取所有的知识库条目。

        返回:
            list[KBEntry]: 包含所有知识库条目的列表。
        """
        return self.data_list

    def get_all_ids(self) -> list[str]:
        """
        获取数据字典中所有的键（ID）列表。

        返回:
            list[str]: 包含数据字典中所有键（ID）的列表。
        """
        return list(self.data_dict.keys())

    def update(
        self, 
        id: str,
        document: Optional[str] = None,
        entry_type: Optional[str] = None,
        group_id: Optional[int] = None,
        last_called_time: Optional[int] = None
    ):
        """
        更新缓存中指定ID的条目信息。

        Args:
            id (str): 要更新的条目ID
            document (Optional[str], optional): 要更新的文档内容。默认为None。
            entry_type (Optional[str], optional): 要更新的条目类型。默认为None。
            group_id (Optional[int], optional): 要更新的组ID。默认为None。
            last_called_time (Optional[int], optional): 要更新的最后调用时间。默认为None。

        Raises:
            ValueError: 如果指定的ID不存在于缓存中
            ValueError: ~~如果没有提供任何要更新的参数(已弃用)~~

        Returns:
            None
        """
        if id not in self.data_dict:
            raise ValueError(f"Entry with id {id} does not exist in the cache.")

        # if not document and not entry_type and not group_id and not last_called_time:
        #     raise ValueError("至少需要提供一个参数来更新")

        if entry_type is not None and entry_type not in ["global", "group"]:
            raise ValueError("entry_type 只能是 global 或 group")

        if document:
            self.data_dict[id].document = document
        if entry_type:
            self.data_dict[id].metadata.type = entry_type # type: ignore
        if group_id:
            self.data_dict[id].metadata.group_id = group_id
        if last_called_time:
            self.data_dict[id].metadata.last_called_time = last_called_time

        self._overwrite_list_based_on_dict()

    def _compare_knowledge_entries(self, a: KBEntry, b: KBEntry, tolerance=1e-6) -> bool:
        """
        比较两个知识库条目是否相同。

        Args:
            a (KBEntry): 第一个知识库条目。
            b (KBEntry): 第二个知识库条目。

        Returns:
            bool: 如果两个条目相同，则返回True；否则返回False。
        """

        if a.id != b.id:
            return False
        if a.document != b.document:
            return False
        if a.metadata.type != b.metadata.type:
            return False
        if a.metadata.group_id != b.metadata.group_id:
            return False
        if abs(a.metadata.creation_time - b.metadata.creation_time) > tolerance:
            return False
        if abs(a.metadata.last_called_time - b.metadata.last_called_time) > tolerance:
            return False
        return True

    async def sync(self):

        # 更新原始列表
        raw_ids = [data.id for data in self.synced_data_list]

        new_datas = await self.kb_manager.get_entry_by_id(raw_ids)


        # 清空原始字典
        self.synced_data_dict: dict[str, KBEntry] = {}
        self.synced_data_list: list[KBEntry] = []
        if new_datas:
            # 更新原始字典
            for data in new_datas:
                if data is None:
                    continue
                self.synced_data_dict[data.id] = data
                self.synced_data_list.append(data)

        # print("raw_data_list:", self.raw_data_list)

        db_data_dict = self.synced_data_dict.copy() # 数据库中的数据
        cache_data_dict = self.data_dict.copy() # 缓存中的数据

        db_set = set(db_data_dict.keys()) # 数据库中的数据集合
        cache_set = set(cache_data_dict.keys()) # 缓存中的数据集合

        add_datas = cache_set - db_set # 缓存中有，数据库中没有的数据，即新增的数据
        delete_datas = db_set - cache_set # 数据库中有，缓存中没有的数据，即删除的数据
        same_datas = db_set & cache_set # 数据库和缓存中都有的数据，即可能需要更新的数据
        update_datas = [] # 需要更新的数据

        for id in same_datas:
            if not self._compare_knowledge_entries(db_data_dict[id], cache_data_dict[id]):
                update_datas.append(id)

        tasks = []
        if add_datas:
            tasks.append(asyncio.create_task(self._batch_add_data_to_db([cache_data_dict[id] for id in add_datas])))
        if delete_datas:
            tasks.append(asyncio.create_task(self._batch_delete_data_from_db(list(delete_datas))))
        if update_datas:
            tasks.append(asyncio.create_task(self._batch_update_data_to_db([cache_data_dict[id] for id in update_datas])))

        # print("add_datas:", add_datas)
        # print("delete_datas:", delete_datas)
        # print("update_datas:", update_datas)

        await asyncio.gather(*tasks)

        self.synced_data_dict = self.data_dict.copy()
        self.synced_data_list = list(self.data_dict.values())


    async def _batch_add_data_to_db(self, datas: list[KBEntry]):
        """
        批量添加数据到数据库。

        Args:
            datas (list[KBEntry]): 要添加的数据列表。

        Returns:
            None
        """

        await self.kb_manager.add_entry(datas)

    async def _batch_delete_data_from_db(self, ids: list[str]):
        """
        批量从数据库中删除数据。

        Args:
            ids (list[str]): 要删除的数据ID列表。

        Returns:
            None
        """
        await self.kb_manager.delete_entry(ids)

    async def _batch_update_data_to_db(self, datas: list[KBEntry]):
        """
        批量更新数据库中的数据。

        Args:
            datas (list[KBEntry]): 要更新的数据列表。

        Returns:
            None
        """
        await self.kb_manager.update_entry(entries=datas)


# with open(r"D:\QQBOT\nonebot\ggz\plugins\知识库管理员提示词.txt", "r", encoding="utf-8") as f:
#     PROMPT = f.read()



class LLMKBController:
    def __init__(
        self,
        chatgpt: ChatGPT,
        knowledge_base_manager: KnowledgeBaseManager,
        prompt: str,
        synced_knowledge_data: Optional[list[KBEntry]] = None,
        
    ) -> None:
        self.chatgpt = chatgpt
        self.knowledge_base_manager = knowledge_base_manager
        if synced_knowledge_data is None:
            self.synced_knowledge_data = []
        else:
            self.synced_knowledge_data: list[KBEntry] = synced_knowledge_data
        self.prompt = prompt



    async def process(self, chat_history: str, group_id: int, remarks: Optional[str] = None ):
        """
        处理输入并执行操作
        """ 
        # 构建上下文
        await self._build_chat_context(chat_history, group_id, remarks)

        # 获取LLM响应
        llm_response = await self._get_llm_response()

        # 解析LLM响应
        action = self.parse_request(llm_response)
        if action is None:
            return None

        # 执行操作
        await self._call_tools(action)
        return action

    async def _build_chat_context(self, chat_history: str, group_id: int, remarks: Optional[str] = None):
        """
        生成并添加AI上下文
        """
        # 清空AI记忆
        self.chatgpt.context = []

        # 添加系统提示词
        await self.chatgpt.add_dialogue(self.prompt, "system")


        # 添加已有的知识库条目
        if self.synced_knowledge_data:
            knowledge_data_text = "# 相关的知识库内容:\n"
            data_json: list[dict] = []
            for entry in self.synced_knowledge_data:
                data_json.append({
                    "id": entry.id,
                    "document": entry.document,
                    "metadata": entry.metadata.to_dict()
                })
            data_text = json.dumps(data_json, ensure_ascii=False, indent=2)
            knowledge_data_text += data_text
            await self.chatgpt.add_dialogue(knowledge_data_text, "system")

        # 添加备注
        if remarks:
            await self.chatgpt.add_dialogue(f"# 备注:\n{remarks}", "system")


        # 添加聊天记录
        chat_history_text = f"群聊ID: {group_id}\n{chat_history}"
        await self.chatgpt.add_dialogue(chat_history_text, "user")

    async def _get_llm_response(self) -> str:
        """
        获取LLM请求
        """
        result = await self.chatgpt.get_response(
            no_input=True, # 应当使用_build_chat_context建立
            add_to_context=False,
            is_reasoning_model=False, # 这里不需要提取reasoning
            return_reasoning=False
        )
        if not isinstance(result, str):
            raise ValueError("LLM返回值类型错误")
        return result

    def parse_request(self, llm_response: str):
        """
        解析LLM请求
        """
        if not ACTION_MARK in llm_response:
            loger.warning("LLM响应中未找到ACTION_MARK")
            return None

        tool_call_content = llm_response.split(ACTION_MARK)[1]
        try:
            tool_call = json.loads(tool_call_content)
        except json.JSONDecodeError:
            loger.warning("LLM响应中ACTION_MARK后的内容无法解析为JSON")
            return None
        if not isinstance(tool_call, list):
            loger.warning("LLM响应中ACTION_MARK后的内容不是JSON数组")
            return None

        return tool_call

    async def _call_tools(self, tool_call: list[dict]):
        """
        调用工具
        """
        # 生成缓存
        cache = KnowledgeBaseCache(self.knowledge_base_manager, self.synced_knowledge_data)
        for tool in tool_call:
            if tool["name"] == "add_entry":
                if not all([param in tool["parameters"] for param in add_parameters]):
                    loger.warning("add_entry参数不完整")
                    continue
                if tool["parameters"]["type"] not in ["global", "group"]:
                    loger.warning("add_entry参数type错误")
                    continue
                if tool["parameters"]["type"] == "group" and ("group_id" not in tool["parameters"] or tool["parameters"]["group_id"] == -1):
                    loger.warning("add_entry参数group_id错误")
                    continue
                if tool["parameters"]["type"] == "global":
                    tool["parameters"]["group_id"] = -1
                cache.add(
                    document=tool["parameters"]["document"],
                    entry_type=tool["parameters"]["type"],
                    group_id=tool["parameters"]["group_id"]
                )
            elif tool["name"] == "delete_entry":
                if not all([param in tool["parameters"] for param in delete_parameters]):
                    loger.warning("delete_entry参数不完整")
                    continue
                cache.delete(
                    id=tool["parameters"]["id"]
                )
            elif tool["name"] == "update_entry":
                if not all([param in tool["parameters"] for param in update_parameters]):
                    loger.warning("update_entry参数不完整")
                    continue
                if set(tool["parameters"].keys()) - set(all_optional_update_parameters): # 检查是否有多余参数
                    loger.warning("update_entry参数多余")
                    continue
                if "type" in tool["parameters"]: 
                    if tool["parameters"]["type"] not in ["global", "group"]:
                        loger.warning("add_entry参数type错误")
                        continue
                    if tool["parameters"]["type"] == "global":
                        tool["parameters"]["group_id"] = -1
                if "group_id" in tool["parameters"]:
                    pass # TODO: 检查group_id是否合法
                

                cache.update(**tool["parameters"])

        # 同步缓存
        await cache.sync()

        self.synced_knowledge_data: List[KBEntry] = deepcopy(cache.synced_data_list)



test_text = """
[12-08 19:06] 星河(234567890)：我知道这家！在万达3楼，上周刚去过，毛肚特别新鲜

[12-08 19:07] 柠檬茶(345678901)：活动细则我看了，工作日下午5点前入座才能享受5折

[12-08 19:09] 豆豆(456789012)：啊...我周末才能休息[哭哭表情]

[12-08 19:10] 清风徐来(123456789)：那周六中午11点半去？那时候人应该不多

[12-08 19:11] 星河(234567890)：可以啊，建议提前订位，他们家有8人位的包间

[12-08 19:13] 小太阳(567890123)：刚健完身，正好需要补充蛋白质💪

[12-08 19:14] 柠檬茶(345678901)：对了，用招商银行信用卡还能再减50元

[12-08 19:16] 豆豆(456789012)：我查了下路线，坐地铁2号线到人民广场站最近

[12-08 19:17] 清风徐来(123456789)：那就这么定了？周六11:30万达3楼蜀香火锅

[12-08 19:18] 星河(234567890)：没问题，我先把地址发群里[位置分享]

[12-08 19:20] 小太阳(567890123)：收到！我先去冲个蛋白粉
"""

async def test():
    chromadbmanager = await OpenAIChromaDBManager.init(
        "https://127.0.0.1:30004",
        "http://127.0.0.1:11434/v1",
        "1",
        model_name="quentinz/bge-base-zh-v1.5:latest",
    )
    if "kb_test_collection" in await chromadbmanager.list_collections_name():
        await chromadbmanager.delete_collection("kb_test_collection")
    col = await chromadbmanager.create_collection("kb_test_collection")
    kb = KnowledgeBaseManager(chromadbmanager, col)
    chat = ChatGPT(
        "sk-8UUdoFZpK2NPGvVzoTarods0Hxixbx6Y",
        "http://gaozaiya.cloudns.org/v1",
        model="gemini-2.5-flash-lite"
    )
    llm = LLMKBController(chat, kb)
    result = await llm.process(
        test_text,
        123456
    )
    print(result)
    input("Press Enter to continue...")
    await chromadbmanager.delete_collection("kb_test_collection")




if __name__ == "__main__":
    asyncio.run(test())

