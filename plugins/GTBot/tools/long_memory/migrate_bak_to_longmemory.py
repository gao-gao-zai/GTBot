"""
从 bak.db 迁移历史群聊消息到长期记忆库

使用说明：
    python -m plugins.GTBot.services.LongMemory.migrate_bak_to_longmemory
    
    或者：
    python plugins/GTBot/services/LongMemory/migrate_bak_to_longmemory.py

断点续传：
    进度保存在 plugins/GTBot/data/migrate_progress.json
    中断后重新运行会自动从断点继续

配置说明：
    Qdrant 和 Embedding 配置从 LongMemory 服务自动获取
    LLM 配置（用于入库整理）需要在本脚本中配置
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 支持直接运行脚本时的导入
# 文件路径：plugins/GTBot/services/LongMemory/migrate_bak_to_longmemory.py
# 需要添加项目根目录 (d:\QQBOT\nonebot\ggz) 到 sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
# SCRIPT_DIR = .../plugins/GTBot/services/LongMemory
# parent = .../plugins/GTBot/services
# parent.parent = .../plugins/GTBot
# parent.parent.parent.parent = .../ (项目根目录)
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nonebot import logger

from plugins.GTBot.model import GroupMessage
from plugins.GTBot.tools.long_memory.IngestManager import (
    LongMemoryIngestConfig,
    LongMemoryIngestManager,
)
from plugins.GTBot.tools.long_memory import (
    long_memory_manager,
    get_long_memory_ingest_manager,
)


# ============================================================================
# 配置
# ============================================================================

# 数据库路径
# data 目录在 plugins/GTBot/data/
BAK_DB_PATH = PROJECT_ROOT / "plugins" / "GTBot" / "data" / "bak.db"
PROGRESS_FILE = PROJECT_ROOT / "plugins" / "GTBot" / "data" / "migrate_progress.json"

# 迁移配置
BATCH_SIZE = 100  # 每批处理的消息数量
FLUSH_INTERVAL = 0.5  # 批次间暂停时间（秒）
MAX_CONCURRENT_GROUPS = 1  # 同时处理的群数量（建议保持 1，避免并发问题）

# 自动停止阈值：当总体“消息进度”达到该比例时优雅停止。
# - 默认 0.4（40%）
# - 设为 <=0 或 >=1 视为禁用
AUTO_STOP_AT_MESSAGE_PROGRESS = float(os.getenv("GTBOT_MIGRATE_AUTO_STOP_PROGRESS", "0.4"))

# LLM 配置（用于入库整理，需要根据实际环境设置）
# Qdrant 和 Embedding 配置会从 LongMemory 服务自动获取
LLM_CONFIG = {
    "model_id": "deepseek-chat",  # 或其他模型
    "base_url": "https://api.deepseek.com/v1",  # 填入你的 LLM API 地址
    "api_key": "YOUR_API_KEY",  # 填入你的 API Key
    "model_parameters": {"temperature": 0.3},
}

# 调试模式：启用后会输出更详细的日志
DEBUG_MODE = False

# 配置详细日志
if DEBUG_MODE:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 设置 IngestManager 日志级别为 DEBUG
    ingest_logger = logging.getLogger("plugins.GTBot.services.LongMemory.IngestManager")
    if ingest_logger:
        ingest_logger.setLevel(logging.DEBUG)


# ============================================================================
# 优雅退出机制
# ============================================================================

class GracefulShutdown:
    """优雅退出处理器。

    说明：
        - 支持 Ctrl+C / SIGTERM 优雅中断。
        - 也支持在代码内部主动触发停止（例如达到某个迁移进度阈值）。
    """

    def __init__(self) -> None:
        self._shutdown_requested = False
        self._current_group_id: Optional[int] = None
        self._current_batch: Optional[list] = None
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any | None) -> None:  # noqa: ARG002
        """信号处理函数。

        Args:
            signum: 信号编号。
            frame: 调用栈帧（通常不使用）。
        """

        self._shutdown_requested = True
        logger.warning(f"\n收到中断信号 (SIG{signum})，正在保存进度...")
        logger.warning("请勿强制关闭，等待当前批次处理完成...")

    def request_stop(self, *, reason: str) -> None:
        """主动请求停止（会在安全点退出）。

        Args:
            reason: 停止原因（用于日志输出）。
        """

        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        logger.warning(f"\n收到停止请求：{reason}；正在保存进度并停止...")
    
    @property
    def should_stop(self) -> bool:
        """是否应该停止"""
        return self._shutdown_requested
    
    def set_current_group(self, group_id: int) -> None:
        """设置当前处理的群 ID"""
        self._current_group_id = group_id
    
    def set_current_batch(self, batch: list[Any]) -> None:
        """设置当前处理的批次"""
        self._current_batch = batch
    
    def get_status(self) -> str:
        """获取当前状态"""
        if not self._shutdown_requested:
            return "running"
        if self._current_group_id:
            return f"stopping_after_group_{self._current_group_id}"
        return "stopping"


# 全局优雅退出处理器\n_shutdown_handler = GracefulShutdown()\n\n\n# ============================================================================\n# CQ 码清洗功能\n# ============================================================================\n\ndef clean_cq_codes(content: str) -> str:\n    \"\"\"清洗 CQ 码，保留关键参数，去除冗余参数\"\"\"\n    \n    def replace_func(di: dict[str, str]):\n        if di[\"CQ\"] == \"mface\":  # 表情\n            summary = di.get(\"summary\", \"\")\n            return \"[CQ:mface\" + (f\",summary={summary}\" if summary else \"\") + \"]\"\n        elif di[\"CQ\"] == \"record\":  # 语音\n            file = di.get(\"file\", \"\")\n            file_size = di.get(\"file_size\", \"\")\n            return \"[CQ:record\" + (f\",file={file}\" if file else \"\") + \\\n                    (f\",file_size={file_size}\" if file_size else \"\") + \"]\"\n        elif di[\"CQ\"] == \"image\":  # 图片\n            file = di.get(\"file\", \"\")\n            file_size = di.get(\"file_size\", \"\")\n            return \"[CQ:image\" + (f\",file={file}\" if file else \"\") + \\\n                    (f\",file_size={file_size}\" if file_size else \"\") + \"]\"\n        else:\n            # 对于其他类型的 CQ 码，生成标准格式，如果过长则截断\n            parts = [f\"CQ:{di['CQ']}\"]\n            for key, value in di.items():\n                if key == \"CQ\":\n                    continue\n                # 处理值中的特殊字符（右方括号需要转义）\n                escaped_value = str(value).replace(']', '\\\\]')\n                parts.append(f\"{key}={escaped_value}\")\n            \n            text = f\"[{','.join(parts)}]\"\n            if len(text) > 100:\n                return f\"{text[:100]}...]\n            return text\n\n    # 正则表达式匹配 CQ 码模式\n    def replace_cq_codes(text: str, replace_func) -> str:\n        \"\"\"替换文本中的 CQ 码\"\"\"\n        pattern = r'(\\[CQ:[^\\]]+\\])'\n        \n        def replace_match(match):\n            full_match = match.group(0)\n            cq_str = full_match[4:-1]  # 去掉开头的\"[CQ:\"和结尾的\"]\"\n            parts = cq_str.split(',', 1)\n            cq_dict = {\"CQ\": parts[0].strip()}\n            \n            if len(parts) > 1:\n                for segment in re.split(r',\\s*(?=[^=]+=)', parts[1]):\n                    if '=' in segment:\n                        key, value = segment.split('=', 1)\n                        cq_dict[key.strip()] = value.strip().replace('\\\\]', ']')\n            \n            # 调用替换函数，如果返回 None 则保留原始 CQ 码\n            replacement = replace_func(cq_dict)\n            return replacement if replacement is not None else full_match\n        \n        return re.sub(pattern, replace_match, text)\n    \n    # 执行清洗\n    return replace_cq_codes(content, replace_func)\n\n\n# ============================================================================\n# 进度管理器\n# ============================================================================\n

# 全局优雅退出处理器
_shutdown_handler: GracefulShutdown = GracefulShutdown()


def clean_cq_codes(content: str) -> str:
    """清洗文本中的 CQ 码，减少冗余参数。

    说明：
        - 对常见类型（`image` / `record` / `mface`）只保留关键参数，避免向量化时引入噪声。
        - 其他 CQ 码保持结构化格式（会做必要转义），并在过长时截断。

    Args:
        content: 原始消息内容。

    Returns:
        清洗后的消息内容。
    """

    def _format_cq(di: dict[str, str]) -> str:
        cq_type = di.get("CQ", "")
        if cq_type == "mface":
            summary = di.get("summary", "")
            return "[CQ:mface" + (f",summary={summary}" if summary else "") + "]"

        if cq_type == "record":
            file = di.get("file", "")
            file_size = di.get("file_size", "")
            return "[CQ:record" + (f",file={file}" if file else "") + (
                f",file_size={file_size}" if file_size else ""
            ) + "]"

        if cq_type == "image":
            file = di.get("file", "")
            file_size = di.get("file_size", "")
            return "[CQ:image" + (f",file={file}" if file else "") + (
                f",file_size={file_size}" if file_size else ""
            ) + "]"

        parts: list[str] = [f"CQ:{cq_type}"] if cq_type else ["CQ:"]
        for key, value in di.items():
            if key == "CQ":
                continue
            escaped_value = str(value).replace("]", "\\]")
            parts.append(f"{key}={escaped_value}")

        text = f"[{','.join(parts)}]"
        if len(text) > 100:
            return text[:97] + "...]"
        return text

    pattern = r"(\[CQ:[^\]]+\])"

    def _replace_match(match: re.Match[str]) -> str:
        full_match = match.group(0)
        cq_str = full_match[4:-1]
        parts = cq_str.split(",", 1)
        cq_dict: dict[str, str] = {"CQ": parts[0].strip()}

        if len(parts) > 1:
            for segment in re.split(r",\s*(?=[^=]+=)", parts[1]):
                if "=" not in segment:
                    continue
                key, value = segment.split("=", 1)
                cq_dict[key.strip()] = value.strip().replace("\\]", "]")

        return _format_cq(cq_dict)

    return re.sub(pattern, _replace_match, content)


def truncate_message(content: str, *, max_chars: int = 200, ellipsis: str = "...") -> str:
    """截断过长消息内容。

    Args:
        content: 原始消息内容。
        max_chars: 最大保留字符数（不包含省略号）。
        ellipsis: 超长时追加的省略号文本。

    Returns:
        截断后的消息内容；若未超长则返回原内容。
    """

    text = str(content)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if not ellipsis:
        return text[:max_chars]
    if max_chars <= len(ellipsis):
        return ellipsis[:max_chars]
    return text[: max_chars - len(ellipsis)] + ellipsis


class MigrationProgress:
    """迁移进度管理器，支持断点续传"""
    
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.data = self._load_progress()
    
    def _load_progress(self) -> dict[str, Any]:
        """加载进度文件，不存在则创建空结构"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"已加载迁移进度：{self.progress_file}")
                return data
            except Exception as e:
                logger.warning(f"加载进度文件失败：{e}，将创建新文件")
        
        # 创建空结构
        return {
            "last_updated": None,
            "total_groups": 0,
            "completed_groups": 0,
            "total_messages": 0,
            "processed_messages": 0,
            "groups": {}
        }
    
    def save(self):
        """保存进度到文件"""
        self.data["last_updated"] = datetime.now().isoformat()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.debug(f"进度已保存：{self.progress_file}")
    
    def initialize_groups(self, group_stats: list[dict[str, Any]]):
        """初始化群列表
        
        Args:
            group_stats: 群统计信息列表，每项包含：
                - group_id: 群 ID
                - total_messages: 消息总数
                - min_send_time: 最早消息时间
                - max_send_time: 最晚消息时间
        """
        self.data["total_groups"] = len(group_stats)
        self.data["total_messages"] = sum(g["total_messages"] for g in group_stats)
        
        for group in group_stats:
            gid = str(group["group_id"])
            if gid not in self.data["groups"]:
                self.data["groups"][gid] = {
                    "group_id": group["group_id"],
                    "total_messages": group["total_messages"],
                    "processed_count": 0,
                    "last_message_id": 0,
                    "last_send_time": group["min_send_time"],
                    "status": "pending"
                }
        self.save()
    
    def get_incomplete_groups(self) -> list[int]:
        """获取未完成的群 ID 列表"""
        incomplete = []
        for gid, info in self.data["groups"].items():
            if info.get("status") not in ("completed", "failed"):
                incomplete.append(int(gid))
        return incomplete
    
    def is_group_completed(self, group_id: int) -> bool:
        """检查群是否已完成"""
        gid = str(group_id)
        return self.data["groups"].get(gid, {}).get("status") == "completed"
    
    def mark_processed(
        self, 
        group_id: int, 
        processed_count: int, 
        last_message_id: int,
        last_send_time: float,
        status: str = "in_progress"
    ):
        """标记一批消息已处理
        
        Args:
            group_id: 群 ID
            processed_count: 本批处理的消息数
            last_message_id: 最后处理的消息 ID
            last_send_time: 最后处理的消息时间
            status: 状态（in_progress / completed / failed）
        """
        gid = str(group_id)
        if gid not in self.data["groups"]:
            logger.warning(f"未知的群 ID: {group_id}")
            return
        
        group_info = self.data["groups"][gid]
        group_info["processed_count"] = group_info.get("processed_count", 0) + processed_count
        group_info["last_message_id"] = last_message_id
        group_info["last_send_time"] = last_send_time
        group_info["status"] = status
        
        # 更新总体统计
        self.data["processed_messages"] = sum(
            g.get("processed_count", 0) for g in self.data["groups"].values()
        )
        self.data["completed_groups"] = sum(
            1 for g in self.data["groups"].values() if g.get("status") == "completed"
        )
        
        self.save()
    
    def get_progress_summary(self) -> str:
        """获取进度摘要"""
        total = self.data.get("total_groups", 0)
        completed = self.data.get("completed_groups", 0)
        total_msgs = self.data.get("total_messages", 0)
        processed_msgs = self.data.get("processed_messages", 0)
        
        pct_groups = f"{completed}/{total}" if total > 0 else "0/0"
        pct_msgs = f"{processed_msgs}/{total_msgs}" if total_msgs > 0 else "0/0"
        
        return f"进度：群聊 {pct_groups} ({completed*100//max(total,1)}%) | 消息 {pct_msgs} ({processed_msgs*100//max(total_msgs,1)}%)"
    
    def get_last_position(self, group_id: int) -> tuple[int, float]:
        """获取群的上次处理位置
        
        Returns:
            (last_message_id, last_send_time)
        """
        gid = str(group_id)
        info = self.data["groups"].get(gid, {})
        return (
            info.get("last_message_id", 0),
            info.get("last_send_time", 0.0)
        )


# ============================================================================
# 数据读取器
# ============================================================================

class BakDbReader:
    """bak.db 数据读取器"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
    
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn
    
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def get_group_stats(self) -> list[dict[str, Any]]:
        """获取各群的消息统计"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                group_id, 
                COUNT(*) as total_messages,
                MIN(send_time) as min_send_time,
                MAX(send_time) as max_send_time
            FROM group_messages
            GROUP BY group_id
            ORDER BY total_messages DESC
        """)
        
        stats = []
        for row in cursor.fetchall():
            stats.append({
                "group_id": row[0],
                "total_messages": row[1],
                "min_send_time": row[2] or 0.0,
                "max_send_time": row[3] or 0.0
            })
        return stats
    
    def get_messages_batch(
        self,
        group_id: int,
        last_message_id: int,
        last_send_time: float,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """获取一批消息（包含 user_name）
        
        返回字典格式，包含完整的群消息信息，便于构建 GroupMessage 对象。
        
        Args:
            group_id: 群 ID
            last_message_id: 上次处理的消息 ID（从该 ID 之后继续）
            last_send_time: 上次处理的消息时间
            batch_size: 批次大小
        
        Returns:
            字典列表，每项包含：
            - message_id: 消息 ID
            - user_id: 用户 ID
            - user_name: 用户昵称
            - content: 消息内容
            - send_time: 发送时间
            - is_withdrawn: 是否撤回
        """
        cursor = self.conn.cursor()
        
        # 按 send_time 和 message_id 排序，确保顺序一致
        cursor.execute("""
            SELECT message_id, user_id, user_name, content, send_time, is_withdrawn
            FROM group_messages
            WHERE group_id = ? AND (send_time > ? OR (send_time = ? AND message_id > ?))
            ORDER BY send_time ASC, message_id ASC
            LIMIT ?
        """, (group_id, last_send_time, last_send_time, last_message_id, batch_size))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "message_id": row[0],
                "user_id": row[1],
                "user_name": row[2] or "",
                "content": row[3] or "",
                "send_time": row[4] or 0.0,
                "is_withdrawn": bool(row[5]),
            })
        
        return messages


# ============================================================================
# 迁移执行器
# ============================================================================

class MigrationExecutor:
    """迁移执行器"""
    
    def __init__(
        self,
        db_reader: BakDbReader,
        progress: MigrationProgress,
        ingest_manager: LongMemoryIngestManager,
        shutdown_handler: GracefulShutdown,
    ):
        self.db_reader = db_reader
        self.progress = progress
        self.ingest_manager = ingest_manager
        self.shutdown_handler = shutdown_handler
        self._stats = {
            "start_time": 0.0,
            "end_time": 0.0,
            "batches_processed": 0,
            "errors": [],
        }
    
    async def migrate_group(self, group_id: int) -> bool:
        """迁移单个群的消息
        
        Returns:
            是否成功完成
        """
        logger.info(f"开始迁移群 {group_id}")
        self.shutdown_handler.set_current_group(group_id)
        
        last_message_id, last_send_time = self.progress.get_last_position(group_id)
        batch_count = 0
        
        try:
            while True:
                # 检查是否需要停止
                if self.shutdown_handler.should_stop:
                    logger.info(f"群 {group_id}: 收到中断请求，保存进度后停止")
                    self.progress.mark_processed(
                        group_id=group_id,
                        processed_count=0,
                        last_message_id=last_message_id,
                        last_send_time=last_send_time,
                        status="in_progress"  # 保持 in_progress 以便下次继续
                    )
                    return False  # 返回 False 表示未完成
                
                if DEBUG_MODE:
                    logger.debug(f"群 {group_id}: 准备获取消息批次 (last_id={last_message_id}, last_time={last_send_time})")
                
                # 获取一批消息
                msg_data = self.db_reader.get_messages_batch(
                    group_id=group_id,
                    last_message_id=last_message_id,
                    last_send_time=last_send_time,
                    batch_size=BATCH_SIZE,
                )
                
                if not msg_data:
                    # 没有更多消息，标记完成
                    logger.info(f"群 {group_id} 迁移完成，共处理 {batch_count} 批")
                    if DEBUG_MODE:
                        logger.debug(f"群 {group_id}: 最终位置 (last_id={last_message_id}, last_time={last_send_time})")
                    self.progress.mark_processed(
                        group_id=group_id,
                        processed_count=0,
                        last_message_id=last_message_id,
                        last_send_time=last_send_time,
                        status="completed"
                    )
                    return True
                
                # 设置当前批次（用于优雅退出）
                self.shutdown_handler.set_current_batch(msg_data)
                
                if DEBUG_MODE:
                    logger.debug(f"群 {group_id}: 获取到 {len(msg_data)} 条消息，准备处理")
                    for i, d in enumerate(msg_data):
                        logger.debug(f"  [{i}] msg_id={d['message_id']}, user={d['user_name']}, content={d['content'][:50]}...")
                
                # 处理这批消息
                for data in msg_data:
                    # 使用 GroupMessage 模型（包含 user_name 字段）
                    # 先清洗 CQ 码
                    cleaned_content = truncate_message(clean_cq_codes(data["content"]), max_chars=200)
                    
                    msg = GroupMessage(
                        message_id=data["message_id"],
                        user_id=data["user_id"],
                        user_name=data["user_name"],
                        content=cleaned_content,  # 使用清洗后的内容
                        send_time=data["send_time"],
                        is_withdrawn=data["is_withdrawn"],
                        group_id=group_id,
                    )
                    
                    # 添加到 IngestManager
                    session_id = f"group_{group_id}"
                    if DEBUG_MODE:
                        logger.debug(f"群 {group_id}: 添加消息到 IngestManager (msg_id={msg.message_id}, session={session_id})")
                    await self.ingest_manager.add_message(
                        session_id=session_id,
                        message=msg,
                        group_id=group_id,
                        user_id=data["user_id"],
                    )
                
                # 触发整理
                session_id = f"group_{group_id}"
                if DEBUG_MODE:
                    logger.debug(f"群 {group_id}: 触发 flush_session (session={session_id}, reason=migration_batch)")
                await self.ingest_manager.flush_session(
                    session_id=session_id,
                    group_id=group_id,
                    user_id=None,
                    reason="migration_batch"
                )
                if DEBUG_MODE:
                    logger.debug(f"群 {group_id}: flush_session 完成")
                
                # 更新进度
                last_message_id = msg_data[-1]["message_id"]
                last_send_time = msg_data[-1]["send_time"]
                batch_count += 1
                self._stats["batches_processed"] += 1
                
                self.progress.mark_processed(
                    group_id=group_id,
                    processed_count=len(msg_data),
                    last_message_id=last_message_id,
                    last_send_time=last_send_time,
                    status="in_progress"
                )

                # 达到阈值后自动停止（用于分段迁移/成本控制）。
                total_msgs = int(self.progress.data.get("total_messages", 0) or 0)
                processed_msgs = int(self.progress.data.get("processed_messages", 0) or 0)
                threshold = float(AUTO_STOP_AT_MESSAGE_PROGRESS)
                if (
                    total_msgs > 0
                    and 0.0 < threshold < 1.0
                    and (processed_msgs / total_msgs) >= threshold
                    and not self.shutdown_handler.should_stop
                ):
                    self.shutdown_handler.request_stop(
                        reason=(
                            f"消息进度达到 {processed_msgs}/{total_msgs} "
                            f"({processed_msgs*100//max(total_msgs, 1)}%)"
                        )
                    )
                    return False
                
                # 暂停一下，避免过快
                if FLUSH_INTERVAL > 0:
                    await asyncio.sleep(FLUSH_INTERVAL)
                
                # 定期输出进度
                if batch_count % 10 == 0:
                    logger.info(f"群 {group_id}: 已处理 {batch_count} 批 ({len(msg_data) * batch_count} 条消息)")
                    logger.info(self.progress.get_progress_summary())
        
        except Exception as e:
            logger.error(f"群 {group_id} 迁移失败：{e}")
            import traceback
            logger.error(traceback.format_exc())
            self._stats["errors"].append({
                "group_id": group_id,
                "error": str(e),
                "last_position": (last_message_id, last_send_time)
            })
            # 标记为失败，便于重试
            self.progress.mark_processed(
                group_id=group_id,
                processed_count=0,
                last_message_id=last_message_id,
                last_send_time=last_send_time,
                status="failed"
            )
            return False
    
    async def migrate_all(self, group_ids: Optional[list[int]] = None):
        """迁移所有或指定的群
        
        Args:
            group_ids: 指定要迁移的群列表，None 表示迁移所有未完成的群
        """
        self._stats["start_time"] = time.time()
        
        if group_ids is None:
            group_ids = self.progress.get_incomplete_groups()
        
        if not group_ids:
            logger.info("所有群聊已完成迁移")
            return
        
        logger.info(f"开始迁移 {len(group_ids)} 个群聊")
        logger.info(self.progress.get_progress_summary())
        logger.info("按 Ctrl+C 可优雅中断，进度会自动保存")
        
        success_count = 0
        fail_count = 0
        interrupted = False
        
        for gid in group_ids:
            result = await self.migrate_group(gid)
            if result:
                success_count += 1
            else:
                # 检查是否是因为中断而停止
                if self.shutdown_handler.should_stop:
                    interrupted = True
                    logger.info(f"已在群 {gid} 处停止，剩余 {len(group_ids) - success_count - fail_count - 1} 个群待迁移")
                    break
                fail_count += 1
            
            # 输出当前进度
            logger.info(self.progress.get_progress_summary())
        
        self._stats["end_time"] = time.time()
        
        # 输出最终报告
        self._print_final_report(success_count, fail_count, interrupted)
    
    def _print_final_report(self, success_count: int, fail_count: int, interrupted: bool = False):
        """输出最终报告"""
        elapsed = self._stats["end_time"] - self._stats["start_time"]
        
        logger.info("=" * 60)
        if interrupted:
            logger.info("迁移中断报告")
        else:
            logger.info("迁移完成报告")
        logger.info("=" * 60)
        logger.info(f"总耗时：{elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        logger.info(f"成功：{success_count} 个群")
        logger.info(f"失败：{fail_count} 个群")
        logger.info(f"处理批次：{self._stats['batches_processed']}")
        logger.info(self.progress.get_progress_summary())
        
        if interrupted:
            logger.info("恢复迁移：重新运行脚本即可从断点继续")
        
        if self._stats["errors"]:
            logger.warning("错误列表:")
            for err in self._stats["errors"]:
                logger.warning(f"  - 群 {err['group_id']}: {err['error']}")


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始从 bak.db 迁移历史消息到长期记忆库")
    logger.info("=" * 60)
    
    # 检查数据库
    if not BAK_DB_PATH.exists():
        logger.error(f"数据库不存在：{BAK_DB_PATH}")
        return
    
    # 检查长期记忆服务
    if long_memory_manager is None:
        logger.error("LongMemory 服务未初始化")
        logger.error("请确保 LongMemory 服务已正确配置并启动")
        return
    
    # 输出 LongMemory 配置信息
    # 说明：Qdrant 和 Embedding 配置来自 plugins/GTBot/services/LongMemory/__init__.py
    logger.info("LongMemory 配置来源：plugins/GTBot/services/LongMemory/__init__.py")
    logger.info("  - Qdrant 服务：http://172.26.226.57:6333")
    logger.info("  - Embedding 服务：http://172.26.226.57:30020/v1/embeddings")
    logger.info("  - Embedding 模型：qwen3-embedding-0.6b")
    
    # 初始化进度管理器
    progress = MigrationProgress(PROGRESS_FILE)
    
    # 初始化数据库读取器
    db_reader = BakDbReader(BAK_DB_PATH)
    
    try:
        # 获取群统计信息
        group_stats = db_reader.get_group_stats()
        logger.info(f"发现 {len(group_stats)} 个群聊，共 {sum(g['total_messages'] for g in group_stats)} 条消息")
        
        # 初始化进度（如果是首次运行）
        if not progress.data.get("groups"):
            progress.initialize_groups(group_stats)
            logger.info("已初始化迁移进度记录")
        
        # 检查 LLM 配置
        if not LLM_CONFIG["base_url"] or not LLM_CONFIG["api_key"]:
            logger.error("错误：LLM 配置为空")
            logger.error(
                "请设置环境变量 GTBOT_LLM_BASE_URL 与 GTBOT_LLM_API_KEY（以及可选的 GTBOT_LLM_MODEL_ID）"
            )
            return

        if 0.0 < float(AUTO_STOP_AT_MESSAGE_PROGRESS) < 1.0:
            logger.warning(
                f"已启用自动停止：当消息进度达到 {AUTO_STOP_AT_MESSAGE_PROGRESS:.2f} 时停止。"
            )
        
        # 创建 IngestManager 配置
        ingest_config = LongMemoryIngestConfig(
            processed_capacity=500,
            pending_capacity=200,
            flush_pending_threshold=200,
            idle_flush_seconds=30.0,
            flush_max_messages=50,
            max_concurrent_flushes=1,
            model_id=LLM_CONFIG["model_id"],
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
            model_parameters=LLM_CONFIG.get("model_parameters", {}),
        )
        
        # 获取或创建 IngestManager
        if DEBUG_MODE:
            logger.debug(f"LLM 配置：model_id={LLM_CONFIG['model_id']}, base_url={LLM_CONFIG['base_url']}")
            logger.debug(f"IngestConfig: processed_capacity={ingest_config.processed_capacity}, "
                        f"flush_max_messages={ingest_config.flush_max_messages}")
        
        ingest_manager = LongMemoryIngestManager(
            config=ingest_config,
            long_memory=long_memory_manager,
        )
        if DEBUG_MODE:
            logger.debug(f"IngestManager 初始化完成")
        
        # 执行迁移
        executor = MigrationExecutor(db_reader, progress, ingest_manager, _shutdown_handler)
        await executor.migrate_all()
        
    finally:
        db_reader.close()
        logger.info("迁移脚本结束")


def run():
    """入口函数"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("收到 KeyboardInterrupt，已退出")
    except Exception as e:
        logger.error(f"迁移脚本异常退出：{e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    run()