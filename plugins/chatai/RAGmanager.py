from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union, Type
import uuid
import sys
import json
import numpy as np
from datetime import datetime

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

    async def _get_data_from_db(self, group_id: int) -> list[ChromaData]:
        """
        从数据库中获取数据列表
        """
        single_list = await self._get_single_list_from_db(group_id, sort=True)
        chunk_list = await self._get_chunk_list_from_db(group_id, sort=True)
        data_list = single_list + chunk_list
        return data_list

    def _get_last_single_index_from_memory(self, data_list: list[ChromaData]) -> Tuple[int, list[str]]:
        """
        从内存中获取最后一条单条记录的索引
        Args:
            data_list (list[ChromaData]): 数据列表
        Returns:
            Tuple[int, list[str]]: 索引和单条记录ID列表(按升序排列)
        """
        single_list: list[ChromaData] = []
        for data in data_list:
            metdata = data.metadata
            if not metdata:
                raise MetadataFormatError("记录中没有元数据")
            if not metdata.get("type"):
                raise MetadataFormatError("记录的元数据中没有类型")
            if metdata["type"] == "chunk":
                single_list.append(data)

        last_index = 0
        record_ids = []
        id_pairs = []
        if len(single_list) == 0:
            return 0, []
        for i in single_list:
            if not i.metadata:
                raise MetadataFormatError("单条消息元数据为空")
            if not isinstance(i.metadata["index"], int):
                raise MetadataFormatError("单条消息索引类型不为int")
            if i.metadata["index"] <= 0:
                raise MetadataFormatError("单条消息索引小于等于0")
            last_index = max(last_index, i.metadata["index"])
            id_pairs.append((i.id, i.metadata["index"]))

        record_ids, _ = sorted(id_pairs, key=lambda x: x[1])

        return last_index, record_ids

    def _get_last_chunk_index_from_memory(self, data_list: list[ChromaData]) -> Tuple[int, list[str]]:
        """
        从内存中获取最后一个块的索引和记录ID

        Args:
            data_list (list[ChromaData]): 数据列表
        Returns:
            Tuple[int, list[str]]: 最后一条索引和记录ID
        """
        last_index = 0
        record_ids = []
        id_paris = []
        chunk_list: list[ChromaData] = []
        for data in data_list:
            metdata = data.metadata
            if not metdata:
                raise MetadataFormatError("记录中没有元数据")
            if not metdata.get("type"):
                raise MetadataFormatError("记录的元数据中没有类型")
            if metdata["type"] == "chunk":
                chunk_list.append(data)
            
        if len(chunk_list) == 0:
            return last_index, record_ids
        for chunk in chunk_list:
            if not chunk.metadata:
                raise MetadataFormatError("单条元数据")
            if not isinstance(chunk.metadata["chunk_index"], int):
                raise MetadataFormatError("单条消息索引类型不为int")
            if chunk.metadata["chunk_index"] <= 0:
                raise MetadataFormatError("单条消息索引小于等于0")
            last_index = max(chunk.metadata["chunk_index"], last_index)
            id_paris.append((chunk.id, chunk.metadata["chunk_index"]))
        if last_index != len(chunk_list):
            raise MetadataFormatError("单条消息索引缺失或错误")
        record_ids, _ = sorted(id_paris, key=lambda x: x[1])
        return (last_index, record_ids)

    def _add_data_to_memory(self, data: ChromaData, data_list: List[ChromaData]) -> List[ChromaData]:
        """添加数据到内存（返回新列表）"""
        if any(d.id == data.id for d in data_list):
            raise DatabaseConsistencyError("数据已存在")
        return data_list + [data] 

    def _delete_data_from_memory(self, data_list: List[ChromaData], id: str) -> List[ChromaData]:
        """从内存中删除数据（返回新列表）"""
        new_list = [d for d in data_list if d.id != id]
        if len(new_list) == len(data_list):
            raise DatabaseConsistencyError("数据不存在")
        return new_list

    def _update_data_in_memory(self, data_list: List[ChromaData], new_data: ChromaData) -> List[ChromaData]:
        """更新内存中的数据（返回新列表）"""
        found = False
        new_list = []
        for d in data_list:
            if d.id == new_data.id:
                new_list.append(new_data)
                found = True
            else:
                new_list.append(d)
        if not found:
            raise ValueError(f"未找到id为{new_data.id}的数据")
        return new_list

    def _ensure_pyembedding(self, embedding: Union[ChromeType.Embedding, ChromeType.PyEmbedding]) -> ChromeType.PyEmbedding:
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding


    async def _memory_data_to_db(self, group_id: int, data_list: list[ChromaData]):
        """将数据同步到数据库"""
        # 获取当前数据库中的所有ID
        db_ids = set()
        db_data = await self._get_data_from_db(group_id)
        for d in db_data:
            db_ids.add(d.id)
        
        # 准备批量操作
        memory_ids = {d.id for d in data_list}
        to_add = [d for d in data_list if d.id not in db_ids]
        to_delete_ids = [d.id for d in db_data if d.id not in memory_ids]
        to_update = [d for d in data_list if d.id in db_ids]
        
        # 执行批量操作
        if to_delete_ids:
            await self.olromadb.delete_records_from_collection(
                self.group_collection, 
                ids=to_delete_ids
            )
        
        if to_add:
            await self._batch_add(to_add)
        
        if to_update:
            await self._batch_update(to_update)

    async def _batch_add(self, data_list: list[ChromaData]):
        """批量添加数据到数据库"""
        if not data_list:
            return
        
        # 准备数据列表
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        # 收集数据并验证格式
        has_embeddings = False
        has_metadatas = False
        
        for data in data_list:
            # 验证ID格式
            if not isinstance(data.id, str):
                raise TypeError(f"ID必须是字符串，实际类型: {type(data.id)}")
            ids.append(data.id)
            
            # 验证文档格式
            if not isinstance(data.document, str):
                raise TypeError(f"文档必须是字符串，实际类型: {type(data.document)}")
            documents.append(data.document)
            
            # 处理嵌入
            if data.embedding is not None:
                has_embeddings = True
                if isinstance(data.embedding, np.ndarray):
                    embeddings.append(data.embedding.tolist())
                else:
                    embeddings.append(data.embedding)
            else:
                embeddings.append(None)  # 占位符，稍后统一处理
                
            # 处理元数据
            if data.metadata is not None:
                has_metadatas = True
                if not isinstance(data.metadata, dict):
                    raise TypeError(f"元数据必须是字典，实际类型: {type(data.metadata)}")
                metadatas.append(data.metadata)
            else:
                metadatas.append(None)  # 占位符，稍后统一处理
        
        # 统一处理可选字段
        final_embeddings = embeddings if has_embeddings else None
        final_metadatas = metadatas if has_metadatas else None
        
        # 检查长度一致性
        n = len(ids)
        if len(documents) != n:
            raise ValueError(f"文档数量({len(documents)})与ID数量({n})不匹配")
        
        if final_embeddings and len(final_embeddings) != n:
            raise ValueError(f"嵌入数量({len(final_embeddings)})与ID数量({n})不匹配")
        
        if final_metadatas and len(final_metadatas) != n:
            raise ValueError(f"元数据数量({len(final_metadatas)})与ID数量({n})不匹配")
        
        # 调用数据库添加
        await self.olromadb.add_records_to_collection(
            collection=self.group_collection,
            ids=ids,
            documents=documents,
            embeddings=final_embeddings,
            metadata=final_metadatas,
            use_ollama=True
        )

    async def _batch_update(self, data_list: list[ChromaData]):
        """批量更新数据库中的数据"""
        if not data_list:
            return
        
        # 准备数据列表
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        # 收集数据并验证格式
        has_documents = False
        has_embeddings = False
        has_metadatas = False
        
        for data in data_list:
            # 验证ID格式
            if not isinstance(data.id, str):
                raise TypeError(f"ID必须是字符串，实际类型: {type(data.id)}")
            ids.append(data.id)
            
            # 处理文档
            if data.document is not None:
                has_documents = True
                if not isinstance(data.document, str):
                    raise TypeError(f"文档必须是字符串，实际类型: {type(data.document)}")
                documents.append(data.document)
            else:
                documents.append(None)  # 占位符，稍后统一处理
                
            # 处理嵌入
            if data.embedding is not None:
                has_embeddings = True
                if isinstance(data.embedding, np.ndarray):
                    embeddings.append(data.embedding.tolist())
                else:
                    embeddings.append(data.embedding)
            else:
                embeddings.append(None)  # 占位符，稍后统一处理
                
            # 处理元数据
            if data.metadata is not None:
                has_metadatas = True
                if not isinstance(data.metadata, dict):
                    raise TypeError(f"元数据必须是字典，实际类型: {type(data.metadata)}")
                metadatas.append(data.metadata)
            else:
                metadatas.append(None)  # 占位符，稍后统一处理
        
        # 统一处理可选字段
        final_documents = documents if has_documents else None
        final_embeddings = embeddings if has_embeddings else None
        final_metadatas = metadatas if has_metadatas else None
        
        # 检查长度一致性
        n = len(ids)
        if final_documents and len(final_documents) != n:
            raise ValueError(f"文档数量({len(final_documents)})与ID数量({n})不匹配")
        
        if final_embeddings and len(final_embeddings) != n:
            raise ValueError(f"嵌入数量({len(final_embeddings)})与ID数量({n})不匹配")
        
        if final_metadatas and len(final_metadatas) != n:
            raise ValueError(f"元数据数量({len(final_metadatas)})与ID数量({n})不匹配")
        
        # 调用数据库更新
        await self.olromadb.update_records_in_collection(
            collection=self.group_collection,
            ids=ids,
            documents=final_documents,
            embeddings=final_embeddings,
            metadatas=final_metadatas,
            use_ollama=True
        )

            


    





    @async_timer
    async def _add_messages_to_group(self, messages:list[GroupMessage]):
        """添加单条消息到数据库"""

        if not all(messages[0].group_id == message.group_id for message in messages):
            raise ValueError("消息列表中的消息组别不一致")

        memory_data = await self._get_data_from_db(messages[0].group_id)
        
        for message in messages: 
            last_index, single_ids = await self._get_last_single_index_from_db(message.group_id)
            if last_index != len(single_ids):
                raise DatabaseConsistencyError("最后索引与检索到的记录长度不匹配, 可能是单条消息的索引不连续")
            index = last_index + 1
            metadata = {
                "type": "single",
                "index": index,
                "related_user_id": json.dumps([message.user_id]), # chromadb不允许元数据的值为列表, 使用字符串代替
                "related_msg_id": json.dumps([message.msg_id]),
                "earliest_send_time": float(message.send_time),
                "latest_send_time": float(message.send_time),
                "message_count": 1
            }
            if isinstance(message, GroupMessage):
                metadata["message_type"] = "group"
                metadata["group_id"] = message.group_id
            elif isinstance(message, PrivateMessage):
                metadata["message_type"] = "private"
            else:
                raise TypeError("不支持的Message类型")

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
            add_id = await self.olromadb.add_records_to_collection(self.group_collection, documents=[text], metadata=[metadata]) # 此时index=单条消息数量
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
                    "group_id": message.group_id,
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
                add_id = await self.olromadb.add_records_to_collection(self.group_collection, documents=[text], metadata=[metadata])
                chunk_ids.append(add_id[0])
                # 锁定前面第2个消息块
                if len(chunk_ids) == 3:
                    await self._lock_chunk(chunk_ids[0]) 
                # 删除原来的单条消息
                await self.olromadb.delete_records_from_collection(self.group_collection, ids=single_ids)
            
            # TODO: 添加块合并逻辑
