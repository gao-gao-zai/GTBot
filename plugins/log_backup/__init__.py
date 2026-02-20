import json
import os
import time
from pathlib import Path
from typing import List, Optional
from collections import deque

from nonebot import get_driver, logger, on_command
from nonebot.rule import to_me
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, Message
from nonebot.params import CommandArg

# 获取当前插件目录
PLUGIN_DIR = Path(__file__).parent.absolute()
CONFIG_FILE = PLUGIN_DIR / "config.json"

def load_config() -> dict:
    """加载并初始化配置文件。

    Returns:
        dict: 包含配置项的字典。
    """
    default_config = {
        "whitelist": [2126979584],  # 预填一个示例 ID
        "log_source_path": "logs/bot.log",
        "backup_dir": "logs/backups"
    }
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        return default_config
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_config = json.load(f)
            # 合并默认配置，确保项完整
            for k, v in default_config.items():
                if k not in user_config:
                    user_config[k] = v
            return user_config
    except Exception as e:
        logger.error(f"加载日志备份插件配置失败: {e}")
        return default_config

# 初始化配置集成
config_data = load_config()
WHITE_LIST: List[int] = config_data.get("whitelist", [])
LOG_SOURCE: str = config_data.get("log_source_path", "logs/bot.log")
BACKUP_DIR: str = config_data.get("backup_dir", "logs/backups")

# 启动时配置日志保存到文件
driver = get_driver()

@driver.on_startup
async def setup_log_file() -> None:
    """在 NoneBot 启动时配置 loguru 文件输出。
    """
    log_path = Path(LOG_SOURCE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加文件 sink，每个文件保存 2 天，保留 10 天
    logger.add(
        LOG_SOURCE,
        rotation="2 days",
        retention="10 days",
        level="DEBUG",
        encoding="utf-8",
        enqueue=True, # 消息队列，提高性能
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    logger.info(f"日志备份服务已启动。源文件: {LOG_SOURCE}")

# 注册指令: #保存日志
save_log = on_command("保存日志", aliases={"backup_log"}, priority=5, block=True, rule=to_me())

@save_log.handle()
async def handle_save_log(bot: Bot, event: MessageEvent, args: Message = CommandArg()) -> None:
    """处理保存日志指令。

    Args:
        bot (Bot): Bot 实例。
        event (MessageEvent): 消息事件对象。
        args (Message): 指令携带的参数。
    """
    user_id = event.user_id
    
    # 鉴权：检查是否为超级用户或在白名单中
    is_whitelist = user_id in WHITE_LIST
    is_superuser = str(user_id) in bot.config.superusers
    
    if not (is_whitelist or is_superuser):
        await save_log.finish("权限不足：只有管理员或白名单用户可以使用此功能。")

    # 参数解析：[注释] [行数]
    params = args.extract_plain_text().strip().split()
    comment = "ManualBackup"
    max_lines = 2000

    if len(params) >= 1:
        comment = params[0]
    if len(params) >= 2:
        try:
            max_lines = int(params[1])
            if max_lines <= 0:
                raise ValueError
        except ValueError:
            await save_log.finish("错误：行数必须是正整数。")

    # 执行备份逻辑
    if not os.path.exists(LOG_SOURCE):
        await save_log.finish(f"错误：日志源文件不存在 ({LOG_SOURCE})。")

    try:
        # 使用 deque 高效读取最后 N 行
        with open(LOG_SOURCE, "r", encoding="utf-8", errors="replace") as f:
            recent_lines = deque(f, maxlen=max_lines)
            log_content = "".join(recent_lines)

        # 生成合规的文件名 (替换 Windows 不支持的字符)
        # 格式：注释 + 当前时间 (月-日 时：分：秒)
        # 考虑到 Windows 兼容性，使用全角或中杠
        timestamp = time.strftime("%m-%d %H-%M-%S")
        filename = f"{comment}_{timestamp}.log"
        
        backup_folder = Path(BACKUP_DIR)
        backup_folder.mkdir(parents=True, exist_ok=True)
        backup_path = backup_folder / filename

        # 写入文件
        with open(backup_path, "w", encoding="utf-8") as bf:
            bf.write(log_content)

        await save_log.finish(
            f"✅ 日志备份成功！\n"
            f"标签: {comment}\n"
            f"数量: {len(recent_lines)} 行\n"
            f"文件名: {filename}"
        )

    except Exception as e:
        logger.exception(f"备份日志时发生异常: {e}")
        await save_log.finish(f"❌ 备份失败: {str(e)}")
