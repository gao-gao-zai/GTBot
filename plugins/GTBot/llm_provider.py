from __future__ import annotations

from typing import Any

from .chat_adapter import (
    SUPPORTED_CHAT_PROVIDER_TYPES,
    build_chat_adapter_model,
    normalize_chat_provider_type,
)


def build_chat_model(
    *,
    provider_type: str,
    model_id: str,
    base_url: str,
    api_key: str,
    streaming: bool,
    model_parameters: dict[str, Any] | None = None,
) -> Any:
    """构建聊天模型对象并保持历史调用入口不变。

    该函数作为兼容层的外部入口，保留原有签名，内部统一委托给
    `build_chat_adapter_model()`，避免现有调用点大规模改名。

    Args:
        provider_type: 提供商类型。
        model_id: 上游模型 ID。
        base_url: 提供商基础地址。
        api_key: 提供商 API 密钥。
        streaming: 是否启用流式输出。
        model_parameters: 归一化后的模型参数。

    Returns:
        可直接用于 LangChain agent 与普通调用的模型对象。
    """

    return build_chat_adapter_model(
        provider_type=provider_type,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        streaming=streaming,
        model_parameters=model_parameters,
    )
