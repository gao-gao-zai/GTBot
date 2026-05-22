from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


PLUGIN_DIR = Path(__file__).resolve().parent


class PythonExpressionSolverPluginConfig(BaseModel):
    """描述 Python 表达式求解器插件的运行配置。

    当前版本只提供最小必要配置：总开关，以及“用户可设置的最大返回长度上限”。
    Agent 每次调用仍可通过参数声明更小的本次上限，但不能超过这里配置的硬上限。
    配置文件缺失或损坏时会自动回退到默认值并重写，以保证工具在首次部署或升级后
    可以直接被加载。
    """

    enabled: bool = True
    max_user_result_length_cap: int = Field(default=100, ge=1, le=10_000)


_config_cache: PythonExpressionSolverPluginConfig | None = None


def _config_path() -> Path:
    """返回插件实际配置文件路径。

    Returns:
        插件目录下 `config.json` 的绝对路径。
    """

    return PLUGIN_DIR / "config.json"


def _example_path() -> Path:
    """返回插件示例配置文件路径。

    Returns:
        插件目录下 `config.json.example` 的绝对路径。
    """

    return PLUGIN_DIR / "config.json.example"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """以原子替换方式写入 JSON 文件。

    这里统一先写入临时文件再覆盖正式文件，避免在机器人运行时恰好被其他逻辑读取到
    半写入内容。

    Args:
        path: 目标 JSON 文件路径。
        data: 待写入的 JSON 对象。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _default_config() -> PythonExpressionSolverPluginConfig:
    """构造插件默认配置对象。

    Returns:
        带默认值的插件配置对象。
    """

    return PythonExpressionSolverPluginConfig()


def _ensure_default_files() -> PythonExpressionSolverPluginConfig:
    """确保配置文件与示例配置文件存在。

    Returns:
        默认配置对象，供首次初始化或回退时复用。
    """

    cfg = _default_config()
    payload = cfg.model_dump(mode="json")
    config_path = _config_path()
    example_path = _example_path()

    if not example_path.exists():
        _write_json(example_path, payload)
    if not config_path.exists():
        _write_json(config_path, payload)
    return cfg


def get_python_expression_solver_plugin_config() -> PythonExpressionSolverPluginConfig:
    """读取并缓存 Python 表达式求解器插件配置。

    解析失败时会记录警告、回退到默认值并重写主配置文件。这样做的目标是让 Agent
    工具始终可加载，而不是因为单个 JSON 配置损坏导致整组工具注册中断。

    Returns:
        当前可用的插件配置对象。
    """

    global _config_cache
    if _config_cache is not None:
        return _config_cache

    default_cfg = _ensure_default_files()
    path = _config_path()
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        if not isinstance(parsed, dict):
            raise TypeError("python_expression_solver config.json 必须是 JSON 对象")
        _config_cache = PythonExpressionSolverPluginConfig.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "python_expression_solver config.json 解析失败，将回退默认配置: %s",
            exc,
        )
        _config_cache = default_cfg
        _write_json(path, _config_cache.model_dump(mode="json"))
    return _config_cache


def reload_python_expression_solver_plugin_config() -> PythonExpressionSolverPluginConfig:
    """清空配置缓存并重新读取配置文件。

    Returns:
        重新加载后的插件配置对象。
    """

    global _config_cache
    _config_cache = None
    return get_python_expression_solver_plugin_config()
