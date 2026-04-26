from __future__ import annotations

from pathlib import Path
from typing import cast

from nonebot import get_plugin_config
from pydantic import BaseModel, Field


class GTPermissionConfig(BaseModel):
    """描述 GT 权限插件的运行配置。

    该配置只关心 owner 列表、权限数据库位置以及是否启用内置命令。
    配置缺失时会自动回落到安全的默认值，以保证插件在测试环境中也可导入。
    `database_path` 支持相对路径，后续会在权限管理器中解析为绝对路径并按需创建目录。
    """

    owner_user_ids: list[int] = Field(default_factory=list)
    database_path: str = "data/gt_permission/admin_users.db"
    enable_commands: bool = True


def get_permission_config() -> GTPermissionConfig:
    """读取当前插件配置。

    NoneBot 尚未初始化时，`get_plugin_config()` 可能无法取得全局配置。
    这里统一回退到默认配置，保证单元测试和静态导入场景不会失败。

    Returns:
        当前可用的权限插件配置对象。
    """

    try:
        return cast(GTPermissionConfig, get_plugin_config(GTPermissionConfig))
    except Exception:
        return GTPermissionConfig()


def resolve_database_path() -> Path:
    """解析权限数据库的绝对路径并确保父目录存在。

    Returns:
        指向 sqlite 数据库文件的绝对路径。
    """

    raw_path = Path(get_permission_config().database_path).expanduser()
    database_path = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return database_path.resolve()
