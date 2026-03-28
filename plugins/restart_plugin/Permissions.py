from pathlib import Path
import json

from nonebot import logger
from nonebot.adapters import Event
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from nonebot.permission import Permission
from nonebot.rule import Rule

plugins_dir_path = Path(__file__).parent
Permissions_config_path = plugins_dir_path / "permission_config.json"
Permissions_example_path = plugins_dir_path / "permission_config.json.example"


def _default_permissions() -> dict:
    """返回重启插件默认权限配置。"""

    return {
        "Permissions": {
            "owner": [],
            "admin": [],
            "black_user": [],
            "black_group": [],
        }
    }


def _load_permissions_config() -> dict:
    """加载权限配置，不存在时自动生成本地配置文件。

    Returns:
        dict: 反序列化后的权限配置对象。
    """

    default_config = _default_permissions()

    if Permissions_example_path.exists():
        try:
            with open(Permissions_example_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                default_config = parsed
        except Exception as e:
            logger.warning(f"读取 restart_plugin 示例权限配置失败，将使用内置默认值: {e}")

    if not Permissions_config_path.exists():
        with open(Permissions_config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        return default_config

    try:
        with open(Permissions_config_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.error(f"加载 restart_plugin 权限配置失败: {e}")

    return default_config


config = _load_permissions_config()
logger.debug(f"config: {config}")

Permissions = config.get(
    "Permissions",
    {
        "owner": [],
        "admin": [],
        "black_user": [],
        "black_group": [],
    },
)

logger.info(f"Permissions: {Permissions}")


async def owner_(event: Event):
    user_id = event.get_user_id()
    return user_id in [str(i) for i in Permissions["owner"]]


async def admin_(event: Event):
    user_id = event.get_user_id()
    return user_id in [str(i) for i in Permissions["admin"]]


async def not_black_user_(event: Event):
    user_id = event.get_user_id()
    return user_id not in [str(i) for i in Permissions["black_user"]]


async def not_black_group_(event: Event):
    if isinstance(event, GroupMessageEvent):
        group_id = str(event.group_id)
        return group_id not in [str(i) for i in Permissions["black_group"]]
    return True


owner = Permission(owner_)
admin = Permission(admin_)
not_black_user = Rule(not_black_user_)
not_black_group = Rule(not_black_group_)
