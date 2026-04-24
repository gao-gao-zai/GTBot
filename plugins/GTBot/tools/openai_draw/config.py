from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


PLUGIN_DIR = Path(__file__).resolve().parent
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_ALLOWED_SIZES = ("1024x1024", "1536x1024", "1024x1536")


class PermissionConfig(BaseModel):
    """描述绘图插件命令侧的权限策略。

    当前仅区分“提交绘图任务”和“查询任务队列”两类命令权限。
    Agent tool 不额外做权限拦截，仍交由上层对话与工具调用链控制。
    """

    submit: Literal["admin", "all"] = "all"
    query: Literal["admin", "all"] = "all"


class OpenAIDrawPluginConfig(BaseModel):
    """描述 OpenAI 绘图插件的运行配置与默认行为。

    配置文件保存在插件目录下的 `config.json`，缺失时会自动生成默认值。
    `base_url` 同时支持官方 OpenAI 地址和 OpenAI 兼容协议网关，调用方无需区分。
    """

    enabled: bool = True
    base_url: str = _DEFAULT_BASE_URL
    api_key: str = ""
    model: str = "gpt-image-1"
    timeout_sec: float = Field(default=120.0, ge=1.0, le=600.0)
    worker_concurrency: int = Field(default=1, ge=1, le=8)
    max_queue_size: int = Field(default=10, ge=1, le=200)
    default_size: str = "1024x1024"
    allowed_sizes: list[str] = Field(default_factory=lambda: list(_DEFAULT_ALLOWED_SIZES))
    default_quality: str = "auto"
    default_background: str = "auto"
    default_input_fidelity: Literal["low", "high"] = "low"
    default_output_format: Literal["png", "jpeg", "webp"] = "png"
    download_dir: str = "./data/openai_draw/images"
    max_input_image_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    max_input_image_count: int = Field(default=16, ge=1, le=16)
    command_prefix: str = "绘图"
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
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
        """校验默认尺寸必须是合法的 `宽x高` 形式。

        Args:
            value: 待校验的默认尺寸字符串。

        Returns:
            规范化后的尺寸字符串。

        Raises:
            ValueError: 当尺寸不符合 `宽x高` 格式时抛出。
        """

        normalized = str(value or "").strip().lower()
        if "x" not in normalized:
            raise ValueError("default_size 必须为 宽x高 格式")
        return normalized

    @field_validator("allowed_sizes")
    @classmethod
    def _validate_allowed_sizes(cls, value: list[str]) -> list[str]:
        """规范化允许使用的尺寸列表并去重。

        Args:
            value: 配置中的尺寸列表。

        Returns:
            去重且保留顺序的合法尺寸列表。

        Raises:
            ValueError: 当尺寸列表为空或存在非法项时抛出。
        """

        normalized: list[str] = []
        for item in value:
            text = str(item or "").strip().lower()
            if "x" not in text:
                raise ValueError("allowed_sizes 中存在非法尺寸")
            if text not in normalized:
                normalized.append(text)
        if not normalized:
            raise ValueError("allowed_sizes 不能为空")
        return normalized

    @property
    def api_base_url(self) -> str:
        """返回用于发起请求的基础地址。

        Returns:
            统一后的 OpenAI Images API 基础地址。
        """

        return str(self.base_url).rstrip("/")

    @property
    def image_generation_url(self) -> str:
        """返回标准 Images API 文生图接口地址。

        Returns:
            可直接用于 `POST /images/generations` 的完整 URL。
        """

        base = self.api_base_url
        if base.endswith("/images/generations"):
            return base
        if base.endswith("/v1"):
            return f"{base}/images/generations"
        return f"{base}/v1/images/generations"

    @property
    def image_edit_url(self) -> str:
        """返回标准 Images API 编辑图接口地址。

        Returns:
            可直接用于 `POST /images/edits` 的完整 URL。
        """

        base = self.api_base_url
        if base.endswith("/images/edits"):
            return base
        if base.endswith("/v1"):
            return f"{base}/images/edits"
        return f"{base}/v1/images/edits"

    @property
    def download_dir_path(self) -> Path:
        """解析图片保存目录为绝对路径。

        Returns:
            插件生成图片的本地保存目录。
        """

        path = Path(self.download_dir)
        if path.is_absolute():
            return path
        return (PLUGIN_DIR / path).resolve()


_config_cache: OpenAIDrawPluginConfig | None = None


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


def _write_json(path: Path, payload: dict) -> None:
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

    配置文件解析失败时会回退到默认配置，并用默认值重写配置文件，
    这样可以保证插件在脏配置或首次启动时仍能继续工作。

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
    """清空配置缓存并重新加载配置。

    Returns:
        最新读取到的配置对象。
    """

    global _config_cache
    _config_cache = None
    return get_openai_draw_plugin_config()
