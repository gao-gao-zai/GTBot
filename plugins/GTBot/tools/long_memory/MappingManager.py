import threading
import string
import random
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional, overload

class MappingManager:
    """单进程、内存级 ID 映射管理器。

    支持 (layer, group) 隔离，支持全局 LRU 自动淘汰。
    """
    
    def __init__(self, max_capacity: int = 100000, short_id_length: int = 4) -> None:
        """初始化映射管理器。

        Args:
            max_capacity: 全局最大缓存条目数。推荐 100,000 条（约占用 25MB~30MB 内存）。
            short_id_length: 生成短 ID 的长度，默认 4。
        """
        self.max_capacity: int = max_capacity
        self.short_id_length: int = short_id_length
        
        # 核心存储 (LRU Source of Truth)
        # Key: (layer, group, long_id) -> Value: short_id
        # 使用 OrderedDict 记录插入/访问顺序
        self._forward_map: "OrderedDict[tuple[str, str, str], str]" = OrderedDict()
        
        # 反向索引 (用于快速查重和反查)
        # Key: (layer, group, short_id) -> Value: long_id
        self._reverse_map: dict[tuple[str, str, str], str] = {}
        
        # 线程锁
        self._lock: threading.RLock = threading.RLock()
        
        # 字符集 (去除容易混淆的字符是个好习惯，这里使用标准集)
        self._chars: str = string.ascii_lowercase + string.digits

    @overload
    def get_short_id(self, layer: str, group: str, long_id: str) -> str:
        """获取短 ID（单条）。"""

    @overload
    def get_short_id(self, layer: str, group: str, long_id: list[str] | tuple[str, ...]) -> list[str]:
        """获取短 ID（批量）。"""

    def get_short_id(self, layer: str, group: str, long_id: str | Sequence[str]) -> str | list[str]:
        """获取短 ID。

        如果映射已存在，返回旧值并刷新 LRU；否则生成新值。

        Args:
            layer: 业务层名称，例如 "user_profile"、"chat_history"。
            group: 组名称，例如 "user_123"、"session_abc"。
            long_id: 原始长 ID（单条或列表），例如 UUID。

        Returns:
            - 传入单条 long_id 时：返回映射后的短 ID。
            - 传入 long_id 列表时：返回短 ID 列表（与输入顺序一致）。
        """
        def _get_one(lid: str) -> str:
            lid = str(lid)
            fwd_key = (layer, group, lid)

            # 1) 命中缓存：移动到末尾（最近使用）
            if fwd_key in self._forward_map:
                self._forward_map.move_to_end(fwd_key)
                return self._forward_map[fwd_key]

            # 2) 未命中：生成新的短 ID
            sid = self._generate_unique_short_id(layer, group)

            # 3) 存入映射
            self._forward_map[fwd_key] = sid
            self._reverse_map[(layer, group, sid)] = lid

            # 4) 检查容量并执行淘汰（Eviction）
            if len(self._forward_map) > self.max_capacity:
                self._evict_oldest()
            return sid

        with self._lock:
            if isinstance(long_id, str):
                return _get_one(long_id)

            ids = [str(x) for x in long_id]
            return [_get_one(x) for x in ids]

    @overload
    def get_long_id(self, layer: str, group: str, short_id: str) -> Optional[str]:
        """根据短 ID 还原长 ID（单条）。"""

    @overload
    def get_long_id(self, layer: str, group: str, short_id: list[str] | tuple[str, ...]) -> list[Optional[str]]:
        """根据短 ID 还原长 ID（批量）。"""

    def get_long_id(
        self,
        layer: str,
        group: str,
        short_id: str | Sequence[str],
    ) -> Optional[str] | list[Optional[str]]:
        """根据短 ID 还原长 ID。

        需要提供 layer 和 group 以确保隔离性。

        Args:
            layer: 业务层名称。
            group: 组名称。
            short_id: 短 ID（单条或列表）。

        Returns:
            - 传入单条 short_id 时：对应的长 ID；若不存在则返回 None。
            - 传入 short_id 列表时：长 ID 列表（元素可为 None，与输入顺序一致）。
        """
        def _get_one(sid: str) -> Optional[str]:
            rev_key = (layer, group, str(sid))
            lid = self._reverse_map.get(rev_key)
            if lid:
                # 命中后，需要更新正向映射的活跃度（LRU）
                fwd_key = (layer, group, lid)
                if fwd_key in self._forward_map:
                    self._forward_map.move_to_end(fwd_key)
                return lid
            return None

        with self._lock:
            if isinstance(short_id, str):
                return _get_one(short_id)

            ids = [str(x) for x in short_id]
            return [_get_one(x) for x in ids]

    def _generate_unique_short_id(self, layer: str, group: str) -> str:
        """在当前 (layer, group) 下生成唯一的短 ID。

        Args:
            layer: 业务层名称。
            group: 组名称。

        Returns:
            在 (layer, group) 维度唯一的短 ID。

        Raises:
            RuntimeError: 极端情况下无法生成唯一短 ID。
        """
        # 为了防止死循环（极小概率满载），设置最大重试。
        last_sid: str = ""
        for _ in range(100):
            last_sid = "".join(random.choices(self._chars, k=self.short_id_length))
            rev_key = (layer, group, last_sid)
            if rev_key not in self._reverse_map:
                return last_sid

        # 兜底：如果随机 100 次都撞了（几乎不可能），增加长度并混入时间后再做唯一性校验。
        # 这里仍然要校验一次，避免在高并发/热启动时极端碰撞。
        for extra_len in (1, 2, 3, 4):
            for _ in range(100):
                suffix = str(int(time.time() * 1000) % 1000)
                sid = "".join(random.choices(self._chars, k=self.short_id_length + extra_len)) + suffix
                if (layer, group, sid) not in self._reverse_map:
                    return sid

        raise RuntimeError(f"无法生成唯一短ID: layer={layer}, group={group}")

    def _evict_oldest(self) -> None:
        """淘汰全局最久未使用的记录。"""
        # last=False 表示弹出第一个插入的元素 (FIFO/LRU)
        (oldest_layer, oldest_group, oldest_long_id), oldest_short_id = self._forward_map.popitem(last=False)
        
        # 清理反向索引
        rev_key = (oldest_layer, oldest_group, oldest_short_id)
        if rev_key in self._reverse_map:
            del self._reverse_map[rev_key]

    def clear_group(self, layer: str, group: str) -> None:
        """手动清理某个组的所有缓存。

        注意：该操作在数据量极大时可能有轻微性能损耗，因为需要遍历。
        鉴于 LRU 会自动淘汰，通常不需要频繁调用。

        Args:
            layer: 业务层名称。
            group: 组名称。
        """
        with self._lock:
            # 收集需要删除的 key
            keys_to_remove = [
                k for k in self._forward_map.keys() 
                if k[0] == layer and k[1] == group
            ]
            for k in keys_to_remove:
                # 从 forward 删除
                short_id = self._forward_map.pop(k)
                # 从 reverse 删除
                del self._reverse_map[(k[0], k[1], short_id)]

# --- 单例模式导出 ---
# 全局最大容量 10万条，满载约 30MB 内存
mapping_manager = MappingManager(max_capacity=100000)