"""GTBot 常量集中定义。

本模块用于集中存放 GTBot 各模块共享的常量，减少重复定义并降低
跨模块引用导致的循环导入风险。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

# GTBot 包目录（用于解析相对路径）。
DIR_PATH: Final[Path] = Path(__file__).parent.resolve()
"""GTBot 包目录的绝对路径。"""


# ------------------------------
# Chat / 消息处理相关常量
# ------------------------------

NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES: Final[int] = 20
"""获取附近消息时额外获取的冗余消息数量，用于保证上下文完整性。"""

DEFAULT_BOT_NAME_PLACEHOLDER: Final[str] = "GTBot"
"""当无法获取机器人名称时使用的占位字符串。"""

SUPPORTED_CQ_CODES: Final[list[str]] = [
    "at",
    "face",
    "image",
    "record",
    "reply",
]
"""允许解析/透传的 CQ 码类型白名单。"""

SEND_MESSAGE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
	r"<msg>(.*?)</msg>",
	re.DOTALL,
)
"""用于匹配 <msg>...</msg> 形式消息块的正则表达式。"""

NOTE_TAG_PATTERN = re.compile(r"<note>(.*?)</note>", flags=re.IGNORECASE | re.DOTALL)
"""用于匹配 <note>...</note> 标签的正则表达式。"""



# ------------------------------
# 数据库 / 持久化相关常量
# ------------------------------

DEFAULT_DB_FILENAME: Final[str] = "data.db"
"""默认数据库文件名。"""

DEFAULT_DB_FALLBACK_PATH: Final[Path] = Path("./data.db")
"""当配置加载失败时使用的默认数据库文件路径（相对于当前工作目录）。"""


__all__ = [
    "DIR_PATH",
    "NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES",
    "DEFAULT_BOT_NAME_PLACEHOLDER",
    "SUPPORTED_CQ_CODES",
    "SEND_MESSAGE_BLOCK_PATTERN",
    "DEFAULT_DB_FILENAME",
    "DEFAULT_DB_FALLBACK_PATH",
]

