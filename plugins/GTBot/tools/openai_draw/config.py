from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


PLUGIN_DIR = Path(__file__).resolve().parent
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class PermissionConfig(BaseModel):
    """描述绘图插件命令侧的权限策略。

    当前只区分“提交绘图任务”和“查看任务队列”两类命令权限。Agent tool 不额外做权限
    拦截，仍交由上层对话与工具调用链控制。
    """

    submit: Literal["admin", "all"] = "all"
    query: Literal["admin", "all"] = "all"


class GlobalUsageLimitsConfig(BaseModel):
    """描述全局维度的绘图次数限制。

    当前只支持自然日和自然周两个周期，用于约束整个插件在所有用户上的总绘图提交量。
    字段值为 `0` 时表示该周期不限额。
    """

    per_day: int = Field(default=0, ge=0)
    per_week: int = Field(default=0, ge=0)


class UserUsageLimitsConfig(BaseModel):
    """描述单用户维度的绘图次数限制。

    周期覆盖自然小时、自然日、自然周和自然月，所有边界都按服务端本地时区计算。字段
    值为 `0` 时表示该周期不限额。
    """

    per_hour: int = Field(default=0, ge=0)
    per_day: int = Field(default=0, ge=0)
    per_week: int = Field(default=0, ge=0)
    per_month: int = Field(default=0, ge=0)


class UsageLimitsConfig(BaseModel):
    """描述绘图次数限制功能的总配置。

    该配置同时控制功能开关、全局周期限额、单用户周期限额，以及完全跳过限额检查与
    计数的豁免用户列表。豁免用户不会占用任何全局或个人额度。
    """

    enabled: bool = False
    exempt_user_ids: list[int] = Field(default_factory=list)
    global_limits: GlobalUsageLimitsConfig = Field(default_factory=GlobalUsageLimitsConfig)
    user_limits: UserUsageLimitsConfig = Field(default_factory=UserUsageLimitsConfig)

    @field_validator("exempt_user_ids")
    @classmethod
    def _validate_exempt_user_ids(cls, value: list[int]) -> list[int]:
        """校验豁免用户列表只包含正整数并按出现顺序去重。

        Args:
            value: 待校验的豁免用户 ID 列表。

        Returns:
            去重后的豁免用户 ID 列表。

        Raises:
            ValueError: 当任一用户 ID 不是正整数时抛出。
        """

        normalized: list[int] = []
        seen: set[int] = set()
        for item in value:
            user_id = int(item)
            if user_id <= 0:
                raise ValueError("usage_limits.exempt_user_ids 只能包含正整数")
            if user_id in seen:
                continue
            seen.add(user_id)
            normalized.append(user_id)
        return normalized


class OpenAIDrawPluginConfig(BaseModel):
    """描述 OpenAI 绘图插件的运行配置与默认行为。

    配置文件保存在插件目录下的 `config.json`。缺失或损坏时会自动回退到默认值，并在必要
    时重写配置文件，保证插件在首次启动或配置异常时仍可继续工作。
    """

    enabled: bool = True
    base_url: str = _DEFAULT_BASE_URL
    api_key: str = ""
    model: str = "gpt-image-1"
    timeout_sec: float = Field(default=120.0, ge=1.0, le=600.0)
    worker_concurrency: int = Field(default=1, ge=1, le=8)
    max_queue_size: int = Field(default=10, ge=1, le=200)
    default_size: str = "1024x1024"
    max_width: int = Field(default=3840, ge=1, le=8192)
    max_height: int = Field(default=3840, ge=1, le=8192)
    size_multiple: int = Field(default=16, ge=1, le=512)
    max_aspect_ratio: float = Field(default=3.0, ge=1.0, le=10.0)
    min_pixels: int = Field(default=655_360, ge=1, le=100_000_000)
    max_pixels: int = Field(default=8_294_400, ge=1, le=100_000_000)
    default_quality: str = "auto"
    default_background: str = "auto"
    default_output_format: Literal["png", "jpeg", "webp"] = "png"
    download_dir: str = "./data/openai_draw/images"
    max_input_image_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    max_input_image_count: int = Field(default=16, ge=1, le=16)
    command_prefix: str = "绘图"
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    usage_limits: UsageLimitsConfig = Field(default_factory=UsageLimitsConfig)
    allow_compat_mode: bool = True

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        """规范化绘图网关地址，确保后续能稳定拼接请求路径。

        Args:
            value: 原始配置中的基础地址。

        Returns:
            去除首尾空白和尾部斜杠后的地址。
        """

        normalized = str(value or "").strip().rstrip("/")
        return normalized or _DEFAULT_BASE_URL

    @field_validator("default_size")
    @classmethod
    def _validate_default_size(cls, value: str) -> str:
        """校验默认尺寸必须是合法的 `宽x高` 或 `auto`。

        这里仅验证显式尺寸的基本格式和正整数约束。更完整的尺寸倍数、像素范围和长宽比
        校验放在运行时完成。

        Args:
            value: 待校验的默认尺寸字符串。

        Returns:
            规范化后的尺寸字符串。

        Raises:
            ValueError: 当尺寸既不是 `auto` 也不是合法 `宽x高` 格式时抛出。
        """

        normalized = str(value or "").strip().lower()
        if normalized == "auto":
            return normalized
        width_text, sep, height_text = normalized.partition("x")
        if sep != "x":
            raise ValueError("default_size 必须为 宽x高 格式")
        try:
            width = int(width_text)
            height = int(height_text)
        except ValueError as exc:
            raise ValueError("default_size 必须为 宽x高 格式") from exc
        if width <= 0 or height <= 0:
            raise ValueError("default_size 的宽高必须大于 0")
        return f"{width}x{height}"

    @field_validator("max_height")
    @classmethod
    def _validate_max_height(cls, value: int, info: Any) -> int:
        """校验最大宽高字段组合必须为正整数。

        Args:
            value: 待校验的最大高度。
            info: Pydantic 字段上下文，用于读取已解析的 `max_width`。

        Returns:
            通过校验的最大高度。

        Raises:
            ValueError: 当最大宽度或最大高度不是正整数时抛出。
        """

        max_width = int(info.data.get("max_width") or 0)
        if max_width <= 0 or int(value) <= 0:
            raise ValueError("max_width 和 max_height 必须大于 0")
        return int(value)

    @field_validator("max_pixels")
    @classmethod
    def _validate_max_pixels(cls, value: int, info: Any) -> int:
        """校验像素范围配置，避免最小像素大于最大像素。

        Args:
            value: 待校验的最大像素数。
            info: Pydantic 字段上下文，用于读取已解析的 `min_pixels`。

        Returns:
            通过校验的最大像素数。

        Raises:
            ValueError: 当最大像素数小于最小像素数时抛出。
        """

        min_pixels = int(info.data.get("min_pixels") or 0)
        if min_pixels <= 0 or int(value) < min_pixels:
            raise ValueError("max_pixels 必须大于等于 min_pixels")
        return int(value)

    @property
    def api_base_url(self) -> str:
        """返回用于发起请求的基础地址。"""

        return str(self.base_url).rstrip("/")

    @property
    def image_generation_url(self) -> str:
        """返回标准 Images API 文生图接口地址。"""

        base = self.api_base_url
        if base.endswith("/images/generations"):
            return base
        if base.endswith("/v1"):
            return f"{base}/images/generations"
        return f"{base}/v1/images/generations"

    @property
    def image_edit_url(self) -> str:
        """返回标准 Images API 编辑图接口地址。"""

        base = self.api_base_url
        if base.endswith("/images/edits"):
            return base
        if base.endswith("/v1"):
            return f"{base}/images/edits"
        return f"{base}/v1/images/edits"

    @property
    def download_dir_path(self) -> Path:
        """解析图片保存目录为绝对路径。"""

        path = Path(self.download_dir)
        if path.is_absolute():
            return path
        return (PLUGIN_DIR / path).resolve()

    @property
    def usage_counter_path(self) -> Path:
        """返回绘图次数计数状态文件的绝对路径。"""

        return PLUGIN_DIR / "usage_counters.json"


_config_cache: OpenAIDrawPluginConfig | None = None


def _config_path() -> Path:
    """返回插件主配置文件路径。"""

    return PLUGIN_DIR / "config.json"


def _config_example_path() -> Path:
    """返回插件示例配置文件路径。"""

    return PLUGIN_DIR / "config.json.example"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """将 JSON 数据写入目标文件。

    Args:
        path: 目标文件路径。
        payload: 待写入的 JSON 对象。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_default_files() -> OpenAIDrawPluginConfig:
    """确保默认配置文件和示例配置文件存在。

    Returns:
        基于默认值构造的配置对象。
    """

    cfg = OpenAIDrawPluginConfig()
    payload = cfg.model_dump(mode="json")

    config_path = _config_path()
    example_path = _config_example_path()

    if not example_path.exists():
        _write_json(example_path, payload)
    if not config_path.exists():
        _write_json(config_path, payload)

    cfg.download_dir_path.mkdir(parents=True, exist_ok=True)
    return cfg


def get_openai_draw_plugin_config() -> OpenAIDrawPluginConfig:
    """读取并缓存绘图插件配置。

    配置文件解析失败时会回退到默认配置，并用默认值重写配置文件，保证插件仍然可用。

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
            raise TypeError("openai_draw config.json 必须是 JSON 对象")
        _config_cache = OpenAIDrawPluginConfig.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"openai_draw config.json 解析失败，将回退默认配置: {exc!s}")
        _config_cache = default_cfg
        _write_json(path, _config_cache.model_dump(mode="json"))

    _config_cache.download_dir_path.mkdir(parents=True, exist_ok=True)
    return _config_cache


def reload_openai_draw_plugin_config() -> OpenAIDrawPluginConfig:
    """清空配置缓存并重新加载配置。"""

    global _config_cache
    _config_cache = None
    return get_openai_draw_plugin_config()
