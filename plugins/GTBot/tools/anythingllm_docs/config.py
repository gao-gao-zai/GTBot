from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


PLUGIN_DIR = Path(__file__).resolve().parent


class AnythingLLMChatConfig(BaseModel):
    """AnythingLLM 文档问答配置。

    Attributes:
        mode: 工作区问答模式，首版默认使用 `query`。
        top_n: 检索返回的参考文档数量。
        similarity_threshold: 工作区相似度阈值。
        session_prefix: 发送到 AnythingLLM 的会话前缀。
        reset: 每次提问前是否重置该会话上下文。
    """

    mode: Literal["query", "chat", "automatic"] = "query"
    top_n: int = Field(default=4, ge=1, le=20)
    similarity_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    session_prefix: str = "gtbot-anythingllm-docs"
    reset: bool = True


class AnythingLLMConfig(BaseModel):
    """AnythingLLM API 接入配置。"""

    base_url: str = "http://127.0.0.1:3001"
    api_key: str = ""
    workspace_name: str = "gtbot-global-docs"
    workspace_slug: str = "gtbot-global-docs"
    timeout_sec: float = Field(default=60.0, ge=5.0, le=300.0)
    max_file_size_mb: int = Field(default=20, ge=1, le=200)
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".txt",
            ".md",
            ".pdf",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
            ".csv",
            ".json",
        ]
    )
    chat: AnythingLLMChatConfig = Field(default_factory=AnythingLLMChatConfig)


class StorageConfig(BaseModel):
    """插件本地存储配置。"""

    temp_dir: str = "./data/temp"
    store_file: str = "./data/documents.json"


class AnythingLLMDocsPluginConfig(BaseModel):
    """AnythingLLM 文档插件总配置。"""

    enabled: bool = True
    anythingllm: AnythingLLMConfig = Field(default_factory=AnythingLLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def resolve_path(self, value: str) -> Path:
        """将相对路径解析为插件目录下的绝对路径。

        Args:
            value: 配置文件中的路径字符串。

        Returns:
            解析后的绝对路径对象。
        """

        path = Path(value)
        if path.is_absolute():
            return path
        return (PLUGIN_DIR / path).resolve()

    @property
    def temp_dir_path(self) -> Path:
        """返回上传临时目录绝对路径。"""

        return self.resolve_path(self.storage.temp_dir)

    @property
    def store_file_path(self) -> Path:
        """返回上传记录文件绝对路径。"""

        return self.resolve_path(self.storage.store_file)


_config_cache: AnythingLLMDocsPluginConfig | None = None


def _config_path() -> Path:
    """返回插件实际配置文件路径。"""

    return PLUGIN_DIR / "config.json"


def _example_path() -> Path:
    """返回插件示例配置文件路径。"""

    return PLUGIN_DIR / "config.json.example"


def _write_json(path: Path, data: dict) -> None:
    """安全写入 JSON 文件。

    Args:
        path: 目标文件路径。
        data: 要写入的 JSON 对象。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _default_config() -> AnythingLLMDocsPluginConfig:
    """构造插件默认配置。"""

    return AnythingLLMDocsPluginConfig()


def _ensure_default_files() -> AnythingLLMDocsPluginConfig:
    """确保配置与示例配置文件存在。"""

    cfg = _default_config()
    payload = cfg.model_dump(mode="json")
    config_path = _config_path()
    example_path = _example_path()

    if not example_path.exists():
        _write_json(example_path, payload)
    if not config_path.exists():
        _write_json(config_path, payload)

    cfg.temp_dir_path.mkdir(parents=True, exist_ok=True)
    cfg.store_file_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg


def get_anythingllm_docs_plugin_config() -> AnythingLLMDocsPluginConfig:
    """读取 AnythingLLM 文档插件配置。"""

    global _config_cache
    if _config_cache is not None:
        return _config_cache

    default_cfg = _ensure_default_files()
    path = _config_path()
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        if not isinstance(parsed, dict):
            raise TypeError("anythingllm_docs config.json must be a JSON object")
        _config_cache = AnythingLLMDocsPluginConfig.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"anythingllm_docs config.json parse failed, fallback to defaults: {exc!s}")
        _config_cache = default_cfg
        _write_json(path, _config_cache.model_dump(mode="json"))

    _config_cache.temp_dir_path.mkdir(parents=True, exist_ok=True)
    _config_cache.store_file_path.parent.mkdir(parents=True, exist_ok=True)
    return _config_cache


def reload_anythingllm_docs_plugin_config() -> AnythingLLMDocsPluginConfig:
    """清空缓存并重新读取配置文件。"""

    global _config_cache
    _config_cache = None
    return get_anythingllm_docs_plugin_config()
