from nonebot.adapters import Event
from nonebot import get_plugin_config, get_driver, logger
from nonebot.permission import Permission
from nonebot.rule import Rule
from pathlib import Path
import json

plugins_dir_path = Path(__file__).parent.parent
Permissions_config_path = plugins_dir_path/"permission_config.json"
with open(Permissions_config_path, "r", encoding="utf-8") as f:
    config = json.load(f)


logger.debug(f"config: {config}")

# 确保配置存在并设置默认值
Permissions = getattr(config, "Permissions", {
    "owner": set(),
    "admin": set(),
    "black_user": set(),
    "black_group": set()
})

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
    group_id = event.get_session_id()
    return group_id not in [str(i) for i in Permissions["black_group"]]

owner = Permission(owner_)
admin = Permission(admin_)
not_black_user = Rule(not_black_user_)
not_black_group = Rule(not_black_group_)
