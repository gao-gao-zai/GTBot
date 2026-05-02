from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


PLUGIN_DIR = Path(__file__).resolve().parent


class MarkdownImageRenderConfig(BaseModel):
    """描述 Markdown 图片工具的默认渲染参数。

    该配置完全由插件侧控制，供 Agent tool 在未接收额外样式参数时直接复用。
    这样可以把“用户偏好的视觉风格”与“Agent 只负责生成正文内容”分离开，避免
    模型在每次调用时反复拼接宽度、主题和 CSS 这类稳定参数。
    """

    auto_width: bool = True
    width: int | None = None
    min_width: int = Field(default=560, ge=360, le=2000)
    max_width: int = Field(default=1200, ge=360, le=2000)
    padding: int = Field(default=32, ge=0, le=240)
    scale: float = Field(default=2.0, gt=0.0, le=4.0)
    theme: str = "default"
    code_theme: str = "default"
    custom_css: str = ""

    @field_validator("width")
    @classmethod
    def _validate_width(cls, value: int | None) -> int | None:
        """校验固定宽度配置。

        Args:
            value: 待校验的固定宽度。

        Returns:
            通过校验后的固定宽度；`None` 表示启用自动宽度模式。

        Raises:
            ValueError: 当宽度超出允许范围时抛出。
        """

        if value is None:
            return None
        normalized = int(value)
        if normalized < 480 or normalized > 2000:
            raise ValueError("width 必须在 480 到 2000 之间")
        return normalized

    @field_validator("max_width")
    @classmethod
    def _validate_max_width(cls, value: int, info: Any) -> int:
        """校验自动宽度边界关系。

        Args:
            value: 待校验的最大宽度。
            info: Pydantic 字段上下文，用于读取已解析的 `min_width`。

        Returns:
            通过校验后的最大宽度。

        Raises:
            ValueError: 当最大宽度小于最小宽度时抛出。
        """

        min_width = int(info.data.get("min_width") or 0)
        normalized = int(value)
        if min_width > 0 and normalized < min_width:
            raise ValueError("max_width 不能小于 min_width")
        return normalized

    @field_validator("theme")
    @classmethod
    def _validate_theme(cls, value: str) -> str:
        """规范化主题配置值。

        主题字段既可以写内置主题名，也可以写本地 CSS/SCSS 文件路径，因此这里只做
        基础字符串清洗，不在配置层强行限制枚举范围。真正的内置主题匹配和文件解析
        会在渲染阶段完成。

        Args:
            value: 原始主题配置值。

        Returns:
            去除首尾空白后的主题值；为空时回落到 `default`。
        """

        normalized = str(value or "").strip()
        return normalized or "default"


class MarkdownImagePluginConfig(BaseModel):
    """描述 Markdown 图片插件的总配置。

    当前插件只有一个核心工具，因此配置集中在渲染层默认值上，不再额外拆分命令、
    配额或权限结构。若后续需要扩展为命令式插件，可以继续在这个模型上增量加字段。
    """

    render: MarkdownImageRenderConfig = Field(default_factory=MarkdownImageRenderConfig)

    def resolve_path(self, value: str) -> Path:
        """把插件配置中的相对路径解析为绝对路径。

        Args:
            value: 配置文件中声明的路径字符串。

        Returns:
            解析后的绝对路径对象。
        """

        path = Path(value)
        if path.is_absolute():
            return path
        return (PLUGIN_DIR / path).resolve()


_config_cache: MarkdownImagePluginConfig | None = None


def _config_path() -> Path:
    """返回插件主配置文件路径。

    Returns:
        `config.json` 的绝对路径。
    """

    return PLUGIN_DIR / "config.json"


def _config_example_path() -> Path:
    """返回插件示例配置文件路径。

    Returns:
        `config.json.example` 的绝对路径。
    """

    return PLUGIN_DIR / "config.json.example"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """把配置数据写入 JSON 文件。

    Args:
        path: 目标文件路径。
        data: 待写入的 JSON 对象。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_default_files() -> MarkdownImagePluginConfig:
    """确保默认配置文件与示例文件存在。

    Returns:
        使用默认值构造出的配置对象。
    """

    cfg = MarkdownImagePluginConfig()
    payload = cfg.model_dump(mode="json")
    config_path = _config_path()
    example_path = _config_example_path()

    if not example_path.exists():
        _write_json(example_path, payload)
    if not config_path.exists():
        _write_json(config_path, payload)
    return cfg


def get_markdown_image_plugin_config() -> MarkdownImagePluginConfig:
    """读取并缓存 Markdown 图片插件配置。

    当配置文件不存在、为空或内容损坏时，函数会退回默认值，并把当前默认配置
    写回 `config.json`，保证插件仍可正常启动与工作。

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
            raise TypeError("markdown_image config.json must be a JSON object")
        _config_cache = MarkdownImagePluginConfig.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"markdown_image config.json parse failed, fallback to defaults: {exc!s}")
        _config_cache = default_cfg
        _write_json(path, _config_cache.model_dump(mode="json"))
    return _config_cache


def reload_markdown_image_plugin_config() -> MarkdownImagePluginConfig:
    """清空缓存并重新读取 Markdown 图片插件配置。

    Returns:
        重载后的插件配置对象。
    """

    global _config_cache
    _config_cache = None
    return get_markdown_image_plugin_config()
