from functools import total_ordering
import os
from pydantic import BaseModel, Field, RootModel, field_validator, ValidationInfo
from typing import Any, Optional, TypeAlias, cast
import json
from pathlib import Path
import sys

from .constants import DIR_PATH
from .llm_provider import normalize_chat_provider_type

ConfigParameterValue: TypeAlias = str | bool | int | float | dict[str, Any] | list[Any]


# 导入日志模块
try:
    from .Logger import logger
except ImportError:
    sys.path.insert(0, str(DIR_PATH))
    from Logger import logger




# ============================================================================
# 定义配置结构 - Original 命名空间
# ============================================================================
# Original 命名空间包含从配置文件直接读取的原始数据结构
# 这些结构与配置文件的 JSON 结构一一对应

class Original:
    """原始配置命名空间 - 存储从配置文件直接解析的数据结构"""
    class GeneralConfiguration(BaseModel):
        """通用配置 - 存储主配置文件中的基础信息"""
        api_config_path: str 
        """API配置文件路径（相对或绝对路径）"""
        config_groups_path: str 
        """配置组文件路径（相对或绝对路径）"""
        default_config_group: str 
        """默认使用的配置组名称"""
        prompt_dir_path: str = "."
        """提示词目录路径（相对或绝对路径），默认为当前目录"""
        data_dir_path: str = "./data"
        """数据目录路径（相对或绝对路径），默认为data目录"""
        plugin_dir: str = "./tools"
        """插件目录, 默认为tools目录"""
        user_cache_update_interval_sec: int = 3600
        """用户缓存刷新间隔（秒）"""
        user_cache_expire_sec: int = 604800
        """用户缓存最长保留时长（秒）"""
        owner_user_ids: list[int] = []
        """所有者用户ID列表"""
        admin_user_ids: list[int] = []
        """旧版管理员用户ID列表（兼容字段）"""
    
    class Provider(BaseModel):
        """单个服务提供商的配置"""
        provider_type: str = "openai_compatible"
        process_tool_call_deltas: bool | None = None
        """提供商类型（如 openai_compatible、anthropic、gemini）"""
        class LLMModel(BaseModel):
            """描述单个大语言模型的基础能力与计费配置。

            该模型对象同时承载两类信息：
            1. 与推理调用直接相关的模型标识、上下文长度和能力声明。
            2. 与聊天自动计费相关的模型价格配置。

            这样设计后，模型价格会与模型别名和上游模型 ID 一起维护，
            可避免在 `config_group` 与 `api_config` 之间重复同步单价。
            """

            class ModelPricing(BaseModel):
                """描述单个模型在 API 配置中的价格信息。

                该配置只负责声明“这个模型本身如何计价”，不负责决定是否启用
                自动记账。启用开关和主币种仍由配置组中的 `chat_model.cost`
                控制，便于按场景决定是否记录消费。
                """

                enabled: bool = False
                input_price_per_million: float = Field(default=0.0, ge=0.0)
                output_price_per_million: float = Field(default=0.0, ge=0.0)
                cache_read_price_per_million: float = Field(default=0.0, ge=0.0)
                currency: str = "CNY"

                @field_validator("currency")
                @classmethod
                def _validate_currency(cls, value: str) -> str:
                    """校验模型价格币种必须为 CNY。

                    当前消费台账和聚合统计默认只支持单币种，因此这里显式拒绝
                    非 `CNY` 配置，避免不同来源金额混算后失真。

                    Args:
                        value: 原始币种字符串，允许大小写混用。

                    Returns:
                        归一化后的大写币种字符串。

                    Raises:
                        ValueError: 当币种不是 `CNY` 时抛出。
                    """

                    normalized = str(value or "").strip().upper()
                    if normalized != "CNY":
                        raise ValueError("当前版本仅支持 CNY 作为主币种")
                    return normalized

            model: str
            """上游模型名称/ID（如 gpt-4, claude-3-opus 等）"""
            max_input_tokens: int
            """允许输入的最大token数"""
            supports_vision: bool
            """是否支持视觉输入（图片理解）"""
            supports_audio: bool
            """是否支持音频输入（语音识别）"""
            parameters: dict[str, ConfigParameterValue]
            """自定义模型参数（如 temperature, top_p 等）"""
            pricing: "Original.Provider.LLMModel.ModelPricing" = Field(
                default_factory=lambda: Original.Provider.LLMModel.ModelPricing()
            )
            """模型价格配置，供聊天自动计费功能在运行时生成价格表。"""
        
        base_url: str
        """API 基础 URL（如 https://api.openai.com/v1）"""
        api_key: str
        """API密钥/令牌"""
        llm_models: dict[str, LLMModel] 
        """该提供商下的所有模型配置，key 为模型简称"""
    
    class APIConfiguration(RootModel[dict[str, Provider]]):
        """
        API配置容器 - 存储所有服务提供商的配置
        使用 RootModel 将整个模型等同于一个字典
        key: 提供商名称（如 openai, anthropic）
        value: 提供商配置对象
        """
        def __getitem__(self, key: str) -> "Original.Provider":
            """支持字典式访问: api_config['openai']"""
            return cast("Original.Provider", self.root[key])
        
        def __contains__(self, key: str) -> bool:
            """支持 in 运算符: 'openai' in api_config"""
            return key in self.root
        
        @field_validator('root')
        @classmethod
        def check_keys(cls, v: dict[str, "Original.Provider"]):
            """验证提供商和模型名称不包含 '/'"""
            for provider_name, provider in v.items():
                if '/' in provider_name:
                    raise ValueError(f"提供商名称不能包含 '/': {provider_name}")
                for model_name in provider.llm_models.keys():
                    if '/' in model_name:
                        raise ValueError(f"模型名称不能包含 '/': {model_name}")
            return v
    
    class SingleConfigurationGroup(BaseModel):
        """单个配置组 - 定义一组使用场景的配置"""
        
        class ChatModel(BaseModel):
            class _LegacyUsagePathSet(BaseModel):
                """定义一组可用于提取 token usage 的路径配置。"""

                input_tokens_path: str = ""
                output_tokens_path: str = ""
                cache_read_tokens_path: str = ""
                request_id_path: str = ""
                input_tokens_include_cache_read: bool = False

                @field_validator("input_tokens_path", "output_tokens_path", "cache_read_tokens_path", "request_id_path")
                @classmethod
                def _validate_optional_path(cls, value: str) -> str:
                    """规范化单个路径字段，统一去除首尾空白。"""

                    return str(value or "").strip()
            """聊天模型配置"""

            class _UnusedUsagePathSet(BaseModel):
                """描述一组在运行时生效的 usage 路径配置。"""

            class UsagePathSet(BaseModel):
                """描述一组在运行时生效的 usage 路径配置。"""

                input_tokens_path: str
                output_tokens_path: str
                cache_read_tokens_path: str
                request_id_path: str
                input_tokens_include_cache_read: bool

            class ProviderUsageRule(BaseModel):
                """定义单个供应商响应体中的 usage 提取路径。"""

                input_tokens_path: str
                output_tokens_path: str
                cache_read_tokens_path: str = ""
                request_id_path: str = ""
                input_tokens_include_cache_read: bool = False
                non_streaming: "Original.SingleConfigurationGroup.ChatModel.UsagePathSet | None" = None
                streaming: "Original.SingleConfigurationGroup.ChatModel.UsagePathSet | None" = None

                @field_validator("input_tokens_path", "output_tokens_path")
                @classmethod
                def _validate_required_usage_path(cls, value: str) -> str:
                    """校验必填 usage 路径不为空。"""

                    normalized = str(value or "").strip()
                    if not normalized:
                        raise ValueError("provider_usage_rules 中的 token 路径不能为空")
                    return normalized

                @field_validator("cache_read_tokens_path", "request_id_path")
                @classmethod
                def _validate_optional_usage_path(cls, value: str) -> str:
                    """规范化可选 usage 路径。"""

                    return str(value or "").strip()

            class ModelPricing(BaseModel):
                """定义单个模型的输入、输出和缓存读取价格。"""

                enabled: bool = False
                input_price_per_million: float = Field(default=0.0, ge=0.0)
                output_price_per_million: float = Field(default=0.0, ge=0.0)
                cache_read_price_per_million: float = Field(default=0.0, ge=0.0)
                currency: str = "CNY"

                @field_validator("currency")
                @classmethod
                def _validate_currency(cls, value: str) -> str:
                    """校验模型价格币种必须为 CNY。"""

                    normalized = str(value or "").strip().upper()
                    if normalized != "CNY":
                        raise ValueError("当前版本仅支持 CNY 作为主币种")
                    return normalized

            class Cost(BaseModel):
                """定义聊天自动计费所需的配置组级控制项。

                该配置仅保留与具体使用场景相关的开关、主币种和 usage 提取规则。
                模型价格改由 `api_config` 中对应模型的 `pricing` 字段维护，
                运行时会在配置合并阶段自动生成 `model_pricing` 映射。
                """

                enabled: bool = False
                base_currency: str = "CNY"
                provider_usage_rules: dict[
                    str,
                    "Original.SingleConfigurationGroup.ChatModel.ProviderUsageRule",
                ] = Field(default_factory=dict)

                @field_validator("base_currency")
                @classmethod
                def _validate_base_currency(cls, value: str) -> str:
                    """校验主币种必须为 CNY。"""

                    normalized = str(value or "").strip().upper()
                    if normalized != "CNY":
                        raise ValueError("chat_model.cost.base_currency 仅支持 CNY")
                    return normalized

                @field_validator("provider_usage_rules")
                @classmethod
                def _validate_provider_usage_rules(
                    cls,
                    value: dict[str, "Original.SingleConfigurationGroup.ChatModel.ProviderUsageRule"],
                ) -> dict[str, "Original.SingleConfigurationGroup.ChatModel.ProviderUsageRule"]:
                    """校验供应商 usage 规则键名不为空。"""

                    for provider_name in value.keys():
                        if not str(provider_name or "").strip():
                            raise ValueError("provider_usage_rules 中存在空供应商名称")
                    return value

            class Continuation(BaseModel):
                """群聊续聊窗口配置。"""

                enabled: bool = False
                """是否启用群聊续聊窗口。"""
                window_seconds: float = 30.0
                """续聊窗口持续时长（秒）。"""
                debounce_seconds: float = 2.0
                """收到新消息后的防抖时间（秒）。"""
                scope: str = "all"
                """允许开窗的触发范围。"""
                analyzer_model: str = ""
                """续聊判定小模型，格式为 `provider/model`。"""
                analyzer_parameters: dict[str, ConfigParameterValue] = Field(default_factory=dict)
                """续聊判定小模型参数。"""
                max_pending_messages: int = 8
                """单次分析中最多纳入的新消息条数。"""
                max_accumulated_messages: int = 12
                """窗口期累计消息上限。"""
                pre_history_messages: int = 6
                """判定时额外带上的响应前历史消息条数。"""
                max_analyzer_context_messages: int = 40
                """提供给续聊判定模型的最长上下文消息条数。"""

                @field_validator("window_seconds")
                @classmethod
                def _validate_window_seconds(cls, v: float) -> float:
                    if v <= 0:
                        raise ValueError("continuation.window_seconds 必须大于 0")
                    return float(v)

                @field_validator("debounce_seconds")
                @classmethod
                def _validate_debounce_seconds(cls, v: float) -> float:
                    if v < 0:
                        raise ValueError("continuation.debounce_seconds 不能小于 0")
                    return float(v)

                @field_validator("scope")
                @classmethod
                def _validate_scope(cls, v: str) -> str:
                    normalized = str(v or "").strip()
                    if normalized not in {"all", "explicit_only", "exclude_auto"}:
                        raise ValueError(
                            "continuation.scope 必须是 all / explicit_only / exclude_auto"
                        )
                    return normalized

                @field_validator("analyzer_model")
                @classmethod
                def _validate_analyzer_model(cls, v: str) -> str:
                    normalized = str(v or "").strip()
                    if normalized and "/" not in normalized:
                        raise ValueError(
                            "continuation.analyzer_model 格式必须为 'provider/model'"
                        )
                    return normalized

                @field_validator("max_pending_messages")
                @classmethod
                def _validate_max_pending_messages(cls, v: int) -> int:
                    if int(v) <= 0:
                        raise ValueError("continuation.max_pending_messages 必须大于 0")
                    return int(v)

                @field_validator("max_accumulated_messages")
                @classmethod
                def _validate_max_accumulated_messages(cls, v: int) -> int:
                    if int(v) <= 0:
                        raise ValueError("continuation.max_accumulated_messages 必须大于 0")
                    return int(v)

                @field_validator("max_accumulated_messages")
                @classmethod
                def _validate_max_accumulated_vs_pending(
                    cls,
                    v: int,
                    info: ValidationInfo,
                ) -> int:
                    max_pending = int(info.data.get("max_pending_messages", 0) or 0)
                    if max_pending > 0 and int(v) < max_pending:
                        raise ValueError(
                            "continuation.max_accumulated_messages 不能小于 max_pending_messages"
                        )
                    return int(v)

                @field_validator("pre_history_messages")
                @classmethod
                def _validate_pre_history_messages(cls, v: int) -> int:
                    if int(v) < 0:
                        raise ValueError("continuation.pre_history_messages 不能小于 0")
                    return int(v)

                @field_validator("max_analyzer_context_messages")
                @classmethod
                def _validate_max_analyzer_context_messages(cls, v: int) -> int:
                    if int(v) <= 0:
                        raise ValueError("continuation.max_analyzer_context_messages 必须大于 0")
                    return int(v)

            class SendTiming(BaseModel):
                """描述 Agent 队列消息的发送节奏配置。

                该配置只影响走消息队列的 Agent 正式回复；系统反馈、后台任务通知等
                非 Agent 消息会走直发路径，因此不会读取这里的节奏参数。
                """

                base_interval_seconds: float = 0.2
                """单条消息的基础最小发送间隔（秒）。"""
                per_char_seconds: float = 0.03
                """每个折算字符额外增加的发送间隔（秒）。"""
                jitter_seconds: float = 0.1
                """在最终结果上叠加的对称随机抖动绝对值（秒）。"""
                max_interval_seconds: float = 2.0
                """单条消息发送间隔允许达到的最大上限（秒）。"""
                non_text_equivalent_chars: dict[str, int] = Field(default_factory=dict)
                """不同非文本消息段类型折算成的等效字符数。"""

                @field_validator(
                    "base_interval_seconds",
                    "per_char_seconds",
                    "jitter_seconds",
                    "max_interval_seconds",
                )
                @classmethod
                def _validate_non_negative_seconds(cls, value: float) -> float:
                    """校验发送节奏中的秒数配置均为非负值。"""

                    normalized = float(value)
                    if normalized < 0:
                        raise ValueError("send_timing 秒数配置不能小于 0")
                    return normalized

                @field_validator("max_interval_seconds")
                @classmethod
                def _validate_max_interval_seconds(
                    cls,
                    value: float,
                    info: ValidationInfo,
                ) -> float:
                    """校验最大发送间隔不小于基础发送间隔。"""

                    base_value = float(info.data.get("base_interval_seconds", 0.0) or 0.0)
                    normalized = float(value)
                    if normalized < base_value:
                        raise ValueError(
                            "send_timing.max_interval_seconds 不能小于 base_interval_seconds"
                        )
                    return normalized

                @field_validator("non_text_equivalent_chars")
                @classmethod
                def _validate_non_text_equivalent_chars(
                    cls,
                    value: dict[str, int],
                ) -> dict[str, int]:
                    """校验非文本段折算配置的键和值都合法。"""

                    normalized: dict[str, int] = {}
                    for raw_key, raw_value in value.items():
                        key = str(raw_key or "").strip()
                        if not key:
                            raise ValueError("send_timing.non_text_equivalent_chars 中存在空消息段类型")
                        int_value = int(raw_value)
                        if int_value < 0:
                            raise ValueError(
                                "send_timing.non_text_equivalent_chars 中的折算字数不能小于 0"
                            )
                        normalized[key] = int_value
                    return normalized

            class Memory(BaseModel):
                """记忆配置。

                用于控制会话记事本（中短期记忆）的容量与保留策略。

                Attributes:
                    notepad_max_entries: 记事本最大条目数。
                    notepad_retention_seconds: 记事本保留时间（秒）。
                        表示会话闲置超过该时间后会被清理；可设为 0 表示不自动清理。
                """

                notepad_max_entries: int = 15
                """记事本最大条目数。"""
                notepad_retention_seconds: float = 300.0
                """记事本保留时间（秒）。"""

                @field_validator("notepad_max_entries")
                @classmethod
                def _validate_notepad_max_entries(cls, v: int) -> int:
                    """校验记事本最大条目数。

                    Args:
                        v: 记事本最大条目数。

                    Returns:
                        校验后的条目数。

                    Raises:
                        ValueError: 当条目数小于等于 0 时抛出。
                    """
                    if v <= 0:
                        raise ValueError("notepad_max_entries 必须为正整数")
                    return v

                @field_validator("notepad_retention_seconds")
                @classmethod
                def _validate_notepad_retention_seconds(cls, v: float) -> float:
                    """校验记事本保留时间。

                    Args:
                        v: 保留时间（秒）。

                    Returns:
                        校验后的保留时间。

                    Raises:
                        ValueError: 当保留时间为负数时抛出。
                    """
                    if v < 0:
                        raise ValueError("notepad_retention_seconds 不能为负数")
                    return v

            class ChatOptOutRule(BaseModel):
                """定义单条群关键词免触发规则。

                该规则仅用于“群关键词触发”链路。命中后表示用户显式声明
                “这句虽然包含关键词，但不要和机器人聊天”。

                Attributes:
                    id: 规则唯一标识，用于日志和维护。
                    enabled: 是否启用当前规则。
                    type: 规则类型，仅支持 `keyword`、`suffix`、`expr`。
                    value: 规则内容。对 `keyword`/`suffix` 为匹配文本，
                        对 `expr` 为受限表达式源码。
                """

                id: str
                enabled: bool = True
                type: str = "keyword"
                value: str

                @field_validator("id")
                @classmethod
                def _validate_id(cls, v: str) -> str:
                    """校验规则 ID，避免空标识进入运行时。

                    Args:
                        v: 原始规则 ID。

                    Returns:
                        去除首尾空白后的规则 ID。

                    Raises:
                        ValueError: 当规则 ID 为空时抛出。
                    """

                    normalized = str(v or "").strip()
                    if not normalized:
                        raise ValueError("chat_opt_out.rules[].id 不能为空")
                    return normalized

                @field_validator("type")
                @classmethod
                def _validate_type(cls, v: str) -> str:
                    """校验规则类型，限制在当前实现支持的范围内。

                    Args:
                        v: 原始规则类型。

                    Returns:
                        规范化后的规则类型。

                    Raises:
                        ValueError: 当规则类型不受支持时抛出。
                    """

                    normalized = str(v or "").strip().lower()
                    if normalized not in {"keyword", "suffix", "expr"}:
                        raise ValueError(
                            "chat_opt_out.rules[].type 必须是 keyword / suffix / expr"
                        )
                    return normalized

                @field_validator("value")
                @classmethod
                def _validate_value(cls, v: str, info: ValidationInfo) -> str:
                    """校验规则内容，拦截空规则和超长表达式。

                    Args:
                        v: 原始规则内容。
                        info: 字段校验上下文，用于读取规则类型。

                    Returns:
                        去除首尾空白后的规则内容。

                    Raises:
                        ValueError: 当规则内容为空或表达式过长时抛出。
                    """

                    normalized = str(v or "").strip()
                    if not normalized:
                        raise ValueError("chat_opt_out.rules[].value 不能为空")
                    rule_type = str(info.data.get("type", "") or "").strip().lower()
                    if rule_type == "expr" and len(normalized) > 300:
                        raise ValueError("chat_opt_out.rules[].value 长度不能超过 300 个字符")
                    return normalized

            class ChatOptOut(BaseModel):
                """定义群关键词免触发配置。

                当消息命中群关键词后，系统会按顺序检查这些规则；任一启用规则命中，
                即视为用户显式要求当前消息不要触发 GTBot 的群关键词回复。
                该配置不会影响 `@GTBot` 或 `to_me` 的显式对话。

                Attributes:
                    enabled: 是否启用群关键词免触发能力。
                    rules: 免触发规则列表，按配置顺序依次匹配。
                """

                enabled: bool = False
                rules: list["Original.SingleConfigurationGroup.ChatModel.ChatOptOutRule"] = Field(
                    default_factory=list
                )

                @field_validator("rules")
                @classmethod
                def _validate_rules(
                    cls,
                    v: list["Original.SingleConfigurationGroup.ChatModel.ChatOptOutRule"],
                ) -> list["Original.SingleConfigurationGroup.ChatModel.ChatOptOutRule"]:
                    """校验规则列表，避免重复 ID 干扰运行时日志。

                    Args:
                        v: 已解析的规则列表。

                    Returns:
                        原始规则列表。

                    Raises:
                        ValueError: 当规则数量超限或存在重复 ID 时抛出。
                    """

                    if len(v) > 100:
                        raise ValueError("chat_opt_out.rules 最多允许 100 条规则")
                    ids: set[str] = set()
                    for item in v:
                        if item.id in ids:
                            raise ValueError(f"chat_opt_out.rules 中存在重复 id: {item.id}")
                        ids.add(item.id)
                    return v
            model: str
            """模型标识符，格式为 'provider/model'（如 'openai/gpt-4'）"""
            maximum_number_of_incoming_messages: int
            """最大输入消息数（用于控制上下文长度）"""
            max_message_length: int = 0
            """单条消息最大长度（字符数），超过此长度会被截断并用'...'代替。0 表示不限制"""
            behavioral_prompt: str
            """行为提示词文件路径"""
            character_prompt: str
            """角色提示词文件路径"""
            max_concurrent_responses_per_group: int = 1
            """单个聊群最多允许同时响应的事件数（0 表示不限制）"""
            max_total_concurrent_responses: int = 5
            """全局最多允许同时响应的总事件数（0 表示不限制）"""
            rejection_emoji_id: int = -1
            """被拒绝时的表情贴ID，-1表示不开启表情回应"""
            max_tool_calls_per_turn: int = 10
            recursion_limit: int = 25
            """单回合最多工具回调次数（0 表示不限制）。超过次数后智能体将停止工具调用"""
            processing_emoji_id: int = -1
            """接收请求时的表情贴ID，-1表示不开启表情回应"""
            completion_emoji_id: int = -1
            """完成请求时的表情贴ID，-1表示不开启表情回应"""
            silent_emoji_id: int = -1
            """输出 `<silent>` 时的表情贴ID，-1表示不开启表情回应"""
            api_timeout_sec: float = 120.0
            """API请求超时时间（秒），0表示不设置超时"""

            memory: Memory = Field(default_factory=Memory)
            """记忆配置。"""
            continuation: Continuation = Field(default_factory=Continuation)
            """群聊续聊窗口配置。"""
            send_timing: SendTiming = Field(default_factory=SendTiming)
            """Agent 队列消息的发送节奏配置。"""
            chat_opt_out: ChatOptOut = Field(default_factory=ChatOptOut)
            cost: Cost = Field(default_factory=Cost)
            """群关键词免触发配置。"""
        
        class UserProfile(BaseModel):
            """用户画像配置"""
            max_descriptions: int = 10
            """允许的最大用户画像描述条数"""
            max_description_char_length: int = 50
            """允许的最大单条描述字符长度"""
            
        chat_model: ChatModel
        """聊天模型配置"""
        user_profile: UserProfile = UserProfile()
        """用户画像配置"""
        message_format_placeholder: str
        """消息格式化模板字符串，用于将消息记录格式化为统一的文本格式"""
    
    class ConfigGroups(RootModel[dict[str, SingleConfigurationGroup]]):
        """
        配置组容器 - 存储所有配置组
        key: 配置组名称（如 'default', 'high_performance'）
        value: 配置组对象
        """
        def __getitem__(self, key: str) -> "Original.SingleConfigurationGroup":
            """支持字典式访问: config_groups['default']"""
            return cast("Original.SingleConfigurationGroup", self.root[key])
        
        def __contains__(self, key: str) -> bool:
            """支持 in 运算符: 'default' in config_groups"""
            return key in self.root
        
        def keys(self):
            """返回所有配置组名称"""
            return self.root.keys()

# ============================================================================
# 定义配置结构 - Processed 命名空间
# ============================================================================
# Processed 命名空间包含经过处理和验证后的配置结构
# 这些结构包含了额外的验证逻辑和数据转换

class Processed:
    """处理后的配置命名空间 - 存储经过验证和转换的配置数据"""
    
    class GeneralConfiguration(BaseModel):
        """处理后的通用配置 - 路径已转换为绝对路径并验证存在性"""
        api_config_path: Path
        """API配置文件的绝对路径"""
        config_group_path: Path
        """配置组文件的绝对路径"""
        default_config_group: str
        """默认配置组名称"""
        prompt_dir_path: Path
        """提示词目录的绝对路径"""
        data_dir_path: Path
        """数据目录的绝对路径"""
        plugin_dir: Path
        """插件目录绝对路径"""
        user_cache_update_interval_sec: int
        """用户缓存刷新间隔（秒）"""
        user_cache_expire_sec: int
        """用户缓存最长保留时间（秒）"""
        owner_user_ids: list[int]
        """所有者用户ID列表"""
        
        @classmethod
        def check_path(cls, v: str|Path, base_path: Path = DIR_PATH):
            """
            路径校验器 - 确保路径有效且文件存在
            
            处理流程：
            1. 展开环境变量（如 $HOME）
            2. 展开用户目录符号（如 ~）
            3. 相对路径转换为相对于 base_path 的绝对路径
            4. 检查文件/目录是否存在
            """
            if isinstance(v, Path):
                p = v
            elif isinstance(v, str):
                # 展开环境变量
                v = os.path.expandvars(v)
                # 展开用户目录
                p = Path(v).expanduser()
            else:
                raise TypeError("路径必须是 str 或 Path 类型")
            
            # 相对路径转换为绝对路径
            if not p.is_absolute():
                p = (base_path / p).resolve()
            
            # 检查路径是否存在
            if not p.exists():
                raise FileNotFoundError(f"路径不存在: {p}")
            
            return p
        
        @classmethod
        def check_or_create_dir_path(cls, v: str|Path, base_path: Path = DIR_PATH):
            """
            路径校验器 - 确保目录路径有效，如果不存在则创建
            
            处理流程：
            1. 展开环境变量（如 $HOME）
            2. 展开用户目录符号（如 ~）
            3. 相对路径转换为相对于 base_path 的绝对路径
            4. 如果目录不存在则创建
            """
            if isinstance(v, Path):
                p = v
            elif isinstance(v, str):
                # 展开环境变量
                v = os.path.expandvars(v)
                # 展开用户目录
                p = Path(v).expanduser()
            else:
                raise TypeError("路径必须是 str 或 Path 类型")
            
            # 相对路径转换为绝对路径
            if not p.is_absolute():
                p = (base_path / p).resolve()
            
            # 如果目录不存在则创建
            if not p.exists():
                try:
                    p.mkdir(parents=True, exist_ok=True)
                    logger.info(f"创建数据目录: {p}")
                except Exception as e:
                    raise FileNotFoundError(f"无法创建目录 {p}: {e}")
            elif not p.is_dir():
                raise ValueError(f"路径存在但不是目录: {p}")
            
            return p
        
        @classmethod
        def from_original(cls, original: Original.GeneralConfiguration):
            """
            从原始配置创建处理后的配置
            
            Args:
                original: 原始通用配置对象
            
            Returns:
                处理后的通用配置对象（路径已验证和转换）
            """
            # 首先解析 prompt_dir_path，它是相对于 DIR_PATH 的
            prompt_dir_path = cls.check_path(original.prompt_dir_path, base_path=DIR_PATH)
            
            # 处理 data_dir_path，如果不存在则创建目录
            data_dir_path = cls.check_or_create_dir_path(original.data_dir_path, base_path=DIR_PATH)
            update_interval = max(60, int(original.user_cache_update_interval_sec))
            expire_interval = max(update_interval, int(original.user_cache_expire_sec))
            
            owner_user_ids = original.owner_user_ids or original.admin_user_ids

            # api_config_path 和 config_groups_path 是相对于 DIR_PATH 的
            return cls(
                api_config_path=cls.check_path(original.api_config_path, base_path=DIR_PATH),
                config_group_path=cls.check_path(original.config_groups_path, base_path=DIR_PATH), 
                plugin_dir=cls.check_path(original.plugin_dir, base_path=DIR_PATH),
                default_config_group=original.default_config_group,
                prompt_dir_path=prompt_dir_path,
                data_dir_path=data_dir_path,
                user_cache_update_interval_sec=update_interval,
                user_cache_expire_sec=expire_interval,
                owner_user_ids=owner_user_ids,
            )
    
    class CurrentConfigGroup(BaseModel):
        """
        当前激活的配置组
        将配置组和 API 配置合并，提供完整的运行时配置
        """
        
        class ChatModel(BaseModel):
            """完整的聊天模型配置 - 包含所有运行时所需信息"""
            provider_type: str
            """模型提供商类型"""

            class Continuation(BaseModel):
                """运行时群聊续聊窗口配置。"""

                enabled: bool
                window_seconds: float
                debounce_seconds: float
                scope: str
                analyzer_provider: str
                analyzer_model_id: str
                analyzer_provider_type: str
                analyzer_base_url: str
                analyzer_api_key: str
                analyzer_parameters: dict[str, ConfigParameterValue]
                max_pending_messages: int
                max_accumulated_messages: int
                pre_history_messages: int
                max_analyzer_context_messages: int

            class Memory(BaseModel):
                """记忆配置（运行时）。

                Attributes:
                    notepad_max_entries: 记事本最大条目数。
                    notepad_retention_seconds: 记事本保留时间（秒）；0 表示不自动清理。
                """

                notepad_max_entries: int
                """记事本最大条目数。"""
                notepad_retention_seconds: float
                """记事本保留时间（秒）。"""
            class ChatOptOutRule(BaseModel):
                """描述单条已进入运行时的群关键词免触发规则。

                运行时规则已经过配置校验，因此这里只保留匹配所需的最小字段，
                供触发器和日志直接消费。
                """

                id: str
                enabled: bool
                type: str
                value: str

            class ChatOptOut(BaseModel):
                """描述运行时群关键词免触发配置。

                该配置由原始配置转换而来。若未启用或规则为空，调用方应保持
                现有群关键词触发行为不变。
                """

                enabled: bool
                rules: list["Processed.CurrentConfigGroup.ChatModel.ChatOptOutRule"] = Field(
                    default_factory=list
                )

            class UsagePathSet(BaseModel):
                """描述一组在运行时生效的 usage 路径配置。"""

                input_tokens_path: str
                output_tokens_path: str
                cache_read_tokens_path: str
                request_id_path: str
                input_tokens_include_cache_read: bool

            class ProviderUsageRule(BaseModel):
                """描述单个供应商在运行时生效的 usage 提取规则。"""

                input_tokens_path: str
                output_tokens_path: str
                cache_read_tokens_path: str
                request_id_path: str
                input_tokens_include_cache_read: bool
                non_streaming: "Processed.CurrentConfigGroup.ChatModel.UsagePathSet | None" = None
                streaming: "Processed.CurrentConfigGroup.ChatModel.UsagePathSet | None" = None

            class ModelPricing(BaseModel):
                """描述单个模型在运行时生效的价格配置。"""

                enabled: bool
                input_price_per_million: float
                output_price_per_million: float
                cache_read_price_per_million: float
                currency: str

            class Cost(BaseModel):
                """描述聊天自动计费功能在运行时使用的完整配置。"""

                enabled: bool
                base_currency: str
                provider_usage_rules: dict[
                    str,
                    "Processed.CurrentConfigGroup.ChatModel.ProviderUsageRule",
                ] = Field(default_factory=dict)
                model_pricing: dict[
                    str,
                    dict[str, "Processed.CurrentConfigGroup.ChatModel.ModelPricing"],
                ] = Field(default_factory=dict)

            class SendTiming(BaseModel):
                """描述运行时用于 Agent 队列消息的发送节奏配置。"""

                base_interval_seconds: float
                per_char_seconds: float
                jitter_seconds: float
                max_interval_seconds: float
                non_text_equivalent_chars: dict[str, int] = Field(default_factory=dict)

            provider_name: str
            model_id: str
            """上游模型的实际ID（从 API 配置中提取）"""
            base_url: str
            """API 基础 URL"""
            api_key: str
            """API 密钥"""
            max_input_tokens: int
            """允许输入的最大token数"""
            maximum_number_of_incoming_messages: int
            """最大输入消息数（用于控制上下文长度）"""
            supports_vision: bool
            """是否支持视觉输入"""
            supports_audio: bool
            """是否支持音频输入"""
            parameters: dict[str, ConfigParameterValue]
            """自定义模型参数"""
            behavioral_prompt: str
            """行为提示词内容"""
            character_prompt: str
            """角色提示词内容"""
            prompt: str
            """最终拼接的提示词内容"""
            max_concurrent_responses_per_group: int
            """单个聊群最多允许同时响应的事件数（0 表示不限制）"""
            max_total_concurrent_responses: int
            """全局最多允许同时响应的总事件数（0 表示不限制）"""
            rejection_emoji_id: int
            """被拒绝时的表情贴ID，-1表示不开启表情回应"""
            max_tool_calls_per_turn: int
            recursion_limit: int
            """单回合最多工具回调次数（0 表示不限制）。超过次数后智能体将停止工具调用"""
            processing_emoji_id: int
            """接收请求时的表情贴ID，-1表示不开启表情回应"""
            completion_emoji_id: int
            """完成请求时的表情贴ID，-1表示不开启表情回应"""
            silent_emoji_id: int
            """输出 `<silent>` 时的表情贴ID，-1表示不开启表情回应"""
            api_timeout_sec: float
            """API请求超时时间（秒），0表示不设置超时"""

            memory: Memory
            """记忆配置。"""
            continuation: Continuation
            """群聊续聊窗口配置。"""
            send_timing: SendTiming
            """Agent 队列消息的发送节奏配置。"""
            chat_opt_out: ChatOptOut
            cost: Cost
            """群关键词免触发配置。"""
        
        class UserProfile(BaseModel):
            """用户画像配置"""
            max_descriptions: int
            """允许的最大用户画像描述条数"""
            max_description_char_length: int
            """允许的最大单条描述字符长度"""
        
        chat_model: ChatModel
        """聊天模型的完整配置"""
        user_profile: UserProfile
        """用户画像配置"""
        message_format_placeholder: str
        """消息格式化模板字符串，用于将消息记录格式化为统一的文本格式"""
        group_name: str
        """当前配置组名称"""
        
        @classmethod
        def from_single_configuration_group(
            cls, 
            original: Original.SingleConfigurationGroup, 
            api_config: Original.APIConfiguration,
            group_name: str,
            prompt_dir_path: Path
        ):
            """
            从单个配置组和 API 配置创建当前配置组
            
            处理流程：
            1. 解析模型名称（格式：provider/model）
            2. 验证提供商和模型是否存在
            3. 合并配置组和 API 配置中的信息
            
            Args:
                original: 原始配置组
                api_config: API 配置
                group_name: 配置组名称
                prompt_dir_path: 提示词目录路径
            
            Returns:
                包含完整信息的当前配置组
            
            Raises:
                ValueError: 模型名称格式错误、提供商不存在或模型不存在
            """
            # 解析并验证模型名称格式
            if "/" not in original.chat_model.model:
                raise ValueError(
                    f"模型名称格式必须为 'provider/model', 当前为: {original.chat_model.model}"
                )
            
            provider, model = original.chat_model.model.split("/", 1)
            
            if not provider or not model:
                raise ValueError(
                    f"模型名称格式必须为 'provider/model', 当前为: {original.chat_model.model}"
                )
            
            # 验证提供商是否存在
            if provider not in api_config:
                raise ValueError(f"提供商 {provider} 不存在")
            
            # 验证模型是否存在
            if model not in api_config[provider].llm_models: 
                raise ValueError(f"模型 {model} 不存在于提供商 {provider} 中")
            
            # 提取提示词信息
            behavioral_prompt_path: Path = Path(original.chat_model.behavioral_prompt)
            character_prompt_path: Path = Path(original.chat_model.character_prompt)
            continuation_cfg = original.chat_model.continuation
            send_timing_cfg = original.chat_model.send_timing
            chat_opt_out_cfg = original.chat_model.chat_opt_out
            cost_cfg = original.chat_model.cost

            analyzer_provider = ""
            analyzer_model_alias = ""
            analyzer_provider_type = ""
            analyzer_model_id = ""
            analyzer_base_url = ""
            analyzer_api_key = ""
            analyzer_parameters: dict[str, ConfigParameterValue] = dict(
                continuation_cfg.analyzer_parameters
            )

            if continuation_cfg.analyzer_model:
                analyzer_provider, analyzer_model_alias = continuation_cfg.analyzer_model.split("/", 1)
                if analyzer_provider not in api_config:
                    raise ValueError(
                        f"续聊判定模型 provider 不存在: {continuation_cfg.analyzer_model}"
                    )
                if analyzer_model_alias not in api_config[analyzer_provider].llm_models:
                    raise ValueError(
                        f"续聊判定模型不存在: {continuation_cfg.analyzer_model}"
                    )
                analyzer_provider_type = normalize_chat_provider_type(
                    api_config[analyzer_provider].provider_type
                )
                analyzer_model_id = api_config[analyzer_provider].llm_models[analyzer_model_alias].model
                analyzer_base_url = api_config[analyzer_provider].base_url
                analyzer_api_key = api_config[analyzer_provider].api_key
                analyzer_parameters = dict(
                    api_config[analyzer_provider].llm_models[analyzer_model_alias].parameters
                ) | analyzer_parameters
            
            if not behavioral_prompt_path.is_absolute():
                behavioral_prompt_path = (prompt_dir_path / behavioral_prompt_path).resolve()
            if not character_prompt_path.is_absolute():
                character_prompt_path = (prompt_dir_path / character_prompt_path).resolve()
            
            if not behavioral_prompt_path.exists() or not character_prompt_path.exists():
                raise FileNotFoundError(
                    f"提示词文件不存在: {behavioral_prompt_path} 或 {character_prompt_path}"
                )
            
            behavioral_prompt = behavioral_prompt_path.read_text(encoding="utf-8")
            character_prompt = character_prompt_path.read_text(encoding="utf-8")
            prompt = behavioral_prompt + "\n\n" + character_prompt
            merged_parameters = dict(api_config[provider].llm_models[model].parameters)
            provider_process_tool_call_deltas = api_config[provider].process_tool_call_deltas
            if (
                provider_process_tool_call_deltas is not None
                and "process_tool_call_deltas" not in merged_parameters
            ):
                merged_parameters["process_tool_call_deltas"] = provider_process_tool_call_deltas
            runtime_model_pricing: dict[str, dict[str, "Processed.CurrentConfigGroup.ChatModel.ModelPricing"]] = {}
            for provider_name, provider_cfg in api_config.root.items():
                provider_pricing: dict[str, "Processed.CurrentConfigGroup.ChatModel.ModelPricing"] = {}
                for model_cfg in provider_cfg.llm_models.values():
                    provider_pricing[str(model_cfg.model)] = cls.ChatModel.ModelPricing(
                        enabled=model_cfg.pricing.enabled,
                        input_price_per_million=model_cfg.pricing.input_price_per_million,
                        output_price_per_million=model_cfg.pricing.output_price_per_million,
                        cache_read_price_per_million=model_cfg.pricing.cache_read_price_per_million,
                        currency=model_cfg.pricing.currency,
                    )
                runtime_model_pricing[str(provider_name)] = provider_pricing
            
            # 合并配置信息创建当前配置组
            return cls(
                chat_model=cls.ChatModel(
                    provider_name=provider,
                    provider_type=normalize_chat_provider_type(api_config[provider].provider_type),
                    model_id=api_config[provider].llm_models[model].model,  
                    base_url=api_config[provider].base_url,
                    api_key=api_config[provider].api_key,
                    max_input_tokens=api_config[provider].llm_models[model].max_input_tokens,
                    maximum_number_of_incoming_messages=original.chat_model.maximum_number_of_incoming_messages, 
                    supports_vision=api_config[provider].llm_models[model].supports_vision,  
                    supports_audio=api_config[provider].llm_models[model].supports_audio, 
                    parameters=merged_parameters, 
                    behavioral_prompt=behavioral_prompt,
                    character_prompt=character_prompt,
                    prompt=prompt,
                    max_concurrent_responses_per_group=original.chat_model.max_concurrent_responses_per_group,
                    max_total_concurrent_responses=original.chat_model.max_total_concurrent_responses,
                    rejection_emoji_id=original.chat_model.rejection_emoji_id,
                    max_tool_calls_per_turn=original.chat_model.max_tool_calls_per_turn,
                    recursion_limit=original.chat_model.recursion_limit,
                    processing_emoji_id=original.chat_model.processing_emoji_id,
                    completion_emoji_id=original.chat_model.completion_emoji_id,
                    silent_emoji_id=original.chat_model.silent_emoji_id,
                    api_timeout_sec=original.chat_model.api_timeout_sec,
                    memory=cls.ChatModel.Memory(
                        notepad_max_entries=original.chat_model.memory.notepad_max_entries,
                        notepad_retention_seconds=original.chat_model.memory.notepad_retention_seconds,
                    ),
                    continuation=cls.ChatModel.Continuation(
                        enabled=continuation_cfg.enabled,
                        window_seconds=continuation_cfg.window_seconds,
                        debounce_seconds=continuation_cfg.debounce_seconds,
                        scope=continuation_cfg.scope,
                        analyzer_provider=analyzer_provider,
                        analyzer_model_id=analyzer_model_id,
                        analyzer_provider_type=analyzer_provider_type,
                        analyzer_base_url=analyzer_base_url,
                        analyzer_api_key=analyzer_api_key,
                        analyzer_parameters=analyzer_parameters,
                        max_pending_messages=continuation_cfg.max_pending_messages,
                        max_accumulated_messages=continuation_cfg.max_accumulated_messages,
                        pre_history_messages=continuation_cfg.pre_history_messages,
                        max_analyzer_context_messages=continuation_cfg.max_analyzer_context_messages,
                    ),
                    send_timing=cls.ChatModel.SendTiming(
                        base_interval_seconds=send_timing_cfg.base_interval_seconds,
                        per_char_seconds=send_timing_cfg.per_char_seconds,
                        jitter_seconds=send_timing_cfg.jitter_seconds,
                        max_interval_seconds=send_timing_cfg.max_interval_seconds,
                        non_text_equivalent_chars=dict(send_timing_cfg.non_text_equivalent_chars),
                    ),
                    chat_opt_out=cls.ChatModel.ChatOptOut(
                        enabled=chat_opt_out_cfg.enabled,
                        rules=[
                            cls.ChatModel.ChatOptOutRule(
                                id=item.id,
                                enabled=item.enabled,
                                type=item.type,
                                value=item.value,
                            )
                            for item in chat_opt_out_cfg.rules
                        ],
                    ),
                    cost=cls.ChatModel.Cost(
                        enabled=cost_cfg.enabled,
                        base_currency=cost_cfg.base_currency,
                        provider_usage_rules={
                            str(provider_name): cls.ChatModel.ProviderUsageRule(
                                input_tokens_path=rule.input_tokens_path,
                                output_tokens_path=rule.output_tokens_path,
                                cache_read_tokens_path=rule.cache_read_tokens_path,
                                request_id_path=rule.request_id_path,
                                input_tokens_include_cache_read=rule.input_tokens_include_cache_read,
                                non_streaming=(
                                    cls.ChatModel.UsagePathSet(
                                        input_tokens_path=rule.non_streaming.input_tokens_path,
                                        output_tokens_path=rule.non_streaming.output_tokens_path,
                                        cache_read_tokens_path=rule.non_streaming.cache_read_tokens_path,
                                        request_id_path=rule.non_streaming.request_id_path,
                                        input_tokens_include_cache_read=(
                                            rule.non_streaming.input_tokens_include_cache_read
                                        ),
                                    )
                                    if rule.non_streaming is not None
                                    else None
                                ),
                                streaming=(
                                    cls.ChatModel.UsagePathSet(
                                        input_tokens_path=rule.streaming.input_tokens_path,
                                        output_tokens_path=rule.streaming.output_tokens_path,
                                        cache_read_tokens_path=rule.streaming.cache_read_tokens_path,
                                        request_id_path=rule.streaming.request_id_path,
                                        input_tokens_include_cache_read=(
                                            rule.streaming.input_tokens_include_cache_read
                                        ),
                                    )
                                    if rule.streaming is not None
                                    else None
                                ),
                            )
                            for provider_name, rule in cost_cfg.provider_usage_rules.items()
                        },
                        model_pricing=runtime_model_pricing,
                    ),
                ),
                user_profile=cls.UserProfile(
                    max_descriptions=original.user_profile.max_descriptions,
                    max_description_char_length=original.user_profile.max_description_char_length
                ),
                message_format_placeholder=original.message_format_placeholder,
                group_name=group_name
            )
        
        @classmethod
        def from_original(
            cls, 
            original: "OriginalConfiguration|Original.ConfigGroups", 
            original_api: "Original.APIConfiguration|None" = None, 
            group_name: str|None = None,
            prompt_dir_path: Path|None = None
        ):
            """
            从原始配置创建当前配置组（支持多种输入方式）
            
            使用方式1：传入完整的 OriginalConfiguration
                - 会自动使用默认配置组（如果 group_name 为 None）
                - API 配置从 OriginalConfiguration 中提取
                - 需要提供 prompt_dir_path (如果 original 中没有处理好的)
                  但这里 original 是 OriginalConfiguration，它包含 Original.GeneralConfiguration，
                  其中只有 str 类型的 prompt_dir_path。
                  所以最好还是传入处理好的 prompt_dir_path。
            
            使用方式2：传入 ConfigGroups 和 APIConfiguration
                - 必须指定 group_name
                - 必须提供 original_api
                - 必须提供 prompt_dir_path
            
            Args:
                original: 原始配置对象或配置组容器
                original_api: API 配置（方式2时必需）
                group_name: 配置组名称（方式2时可选，方式1时可选）
                prompt_dir_path: 提示词目录路径（必需）
            
            Returns:
                当前配置组对象
            
            Raises:
                ValueError: 参数缺失或配置组不存在
            """
            if prompt_dir_path is None:
                 # 如果是方式1，我们可以尝试解析，但最好是强制要求传入
                 # 为了简化，我们强制要求传入 prompt_dir_path
                 raise ValueError("必须提供 prompt_dir_path")

            if isinstance(original, OriginalConfiguration):
                # 使用方式1：从完整配置创建
                if group_name is None:
                    # 使用默认配置组
                    group_name = original.config.default_config_group  
                
                # 验证配置组存在
                if group_name not in original.config_groups:
                    raise ValueError(f"配置组 {group_name} 不存在")
                
                original_config_group = original.config_groups[group_name]
                api_config = original.api_config
                
                return cls.from_single_configuration_group(
                    original_config_group, 
                    api_config,
                    group_name,
                    prompt_dir_path
                )  
            
            elif isinstance(original, Original.ConfigGroups):
                # 使用方式2：从配置组容器创建
                if group_name is None:
                    raise ValueError("使用 ConfigGroups 时必须指定配置组名称")
                if original_api is None:
                    raise ValueError("使用 ConfigGroups 时必须指定 API 配置")
                
                # 验证配置组存在
                if group_name not in original:
                    raise ValueError(f"配置组 {group_name} 不存在")
                
                original_config_group = original[group_name]
                api_config = original_api
                
                return cls.from_single_configuration_group(
                    original_config_group, 
                    api_config,
                    group_name,
                    prompt_dir_path
                )  
    
    # 直接复用 Original 中的类型（这些类型不需要额外处理）
    APIConfiguration = Original.APIConfiguration
    ConfigGroups = Original.ConfigGroups

# ============================================================================
# 配置容器类
# ============================================================================

class OriginalConfiguration(BaseModel):
    """
    原始配置容器 - 存储所有从文件读取的原始配置
    这是配置加载的第一阶段，数据与文件内容一致
    """
    config: Original.GeneralConfiguration
    """通用配置（主配置文件内容）"""
    api_config: Original.APIConfiguration
    """API 配置（所有提供商和模型的配置）"""
    config_groups: Original.ConfigGroups
    """配置组集合（所有使用场景的配置）"""


class ProcessedConfiguration(BaseModel):
    """
    处理后的配置容器 - 存储经过验证和转换的配置
    这是配置加载的第二阶段，包含运行时所需的所有信息
    """
    config: Processed.GeneralConfiguration
    """处理后的通用配置（路径已验证）"""
    api_config: Original.APIConfiguration
    """API 配置"""
    config_groups: Original.ConfigGroups
    """配置组集合"""
    current_config_group: Processed.CurrentConfigGroup
    """当前激活的配置组（包含合并后的完整信息）"""
    
    def switch_config_group(self, group_name: str) -> None:
        """
        切换到指定的配置组
        
        功能说明：
        1. 验证目标配置组是否存在
        2. 重新创建 CurrentConfigGroup 对象
        3. 更新 current_config_group 属性
        
        Args:
            group_name: 目标配置组名称
        
        Raises:
            ValueError: 配置组不存在
        
        使用示例：
            ```python
            # 切换到高性能配置组
            processed_config.switch_config_group("high_performance")
            
            # 切换回默认配置组
            processed_config.switch_config_group(
                processed_config.config.default_config_group
            )
            ```
        """
        # 验证配置组是否存在
        if group_name not in self.config_groups:
            available_groups = list(self.config_groups.root.keys())
            raise ValueError(
                f"配置组 '{group_name}' 不存在。"
                f"可用的配置组有: {', '.join(available_groups)}"
            )
        
        # 创建新的当前配置组
        new_current_config_group = Processed.CurrentConfigGroup.from_original(
            self.config_groups,
            self.api_config,
            group_name,
            prompt_dir_path=self.config.prompt_dir_path
        )
        
        # 更新当前配置组
        self.current_config_group = new_current_config_group
        
        logger.info(f"✅ 已切换到配置组: {group_name}")
    
    def get_available_config_groups(self) -> list[str]:
        """
        获取所有可用的配置组名称列表
        
        Returns:
            配置组名称列表
        
        使用示例：
            ```python
            groups = processed_config.get_available_config_groups()
            print(f"可用配置组: {groups}")
            # 输出: 可用配置组: ['default', 'high_performance', 'low_cost']
            ```
        """
        return list(self.config_groups.root.keys())
    
    def get_current_group_name(self) -> str:
        """
        获取当前配置组名称
        
        Returns:
            当前配置组名称
        
        使用示例：
            ```python
            current = processed_config.get_current_group_name()
            print(f"当前配置组: {current}")
            # 输出: 当前配置组: default
            ```
        """
        return self.current_config_group.group_name


class TotalConfiguration(BaseModel):
    """
    总配置类 - 同时保存原始配置和处理后的配置
    
    用途：
    - 原始配置：用于配置文件的编辑和保存
    - 处理后的配置：用于程序运行时使用
    """
    original_configuration: OriginalConfiguration
    """原始配置（未经处理）"""
    processed_configuration: ProcessedConfiguration
    """处理后的配置（可直接使用）"""
    _config_path: Path | None = None
    """配置文件路径（私有字段，用于重载）"""
    
    model_config = {"arbitrary_types_allowed": True}
    
    @classmethod
    def init(cls, config_path: str|Path|None = None): 
        """
        初始化总配置对象
        
        Args:
            config_path: 主配置文件路径，默认为 config/config.json
        
        Returns:
            初始化完成的 TotalConfiguration 对象
        """
        resolved_config_path: Path
        if isinstance(config_path, str):
            resolved_config_path = Path(config_path)
        elif config_path is None:
            resolved_config_path = DIR_PATH / "config" / "config.json"
        else:
            resolved_config_path = config_path
        
        # 确保路径是绝对路径
        if not resolved_config_path.is_absolute():
            resolved_config_path = resolved_config_path.resolve()
        
        # 加载配置文件
        with open(resolved_config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            # 解析为原始配置对象
            original_config = Original.GeneralConfiguration(**config_data)
            # 转换为处理后的配置（验证路径）
            config = Processed.GeneralConfiguration.from_original(original_config)
        
        # 加载 API 配置文件
        with open(config.api_config_path, "r", encoding="utf-8") as f:
            api_config_data = json.load(f)
            # 解析 API 配置
            api_config: Original.APIConfiguration = Original.APIConfiguration(api_config_data)
        
        # 加载配置组文件并创建当前配置组
        with open(config.config_group_path, "r", encoding="utf-8") as f:
            config_groups_data = json.load(f)
            # 解析配置组
            config_groups = Original.ConfigGroups(config_groups_data)
            # 创建当前配置组（使用默认配置组）
            current_config_group = Processed.CurrentConfigGroup.from_original(  
                config_groups,
                api_config,
                config.default_config_group,
                prompt_dir_path=config.prompt_dir_path
            )
        
        # 组装总配置对象
        total_config = cls(
            original_configuration=OriginalConfiguration(
                config=original_config,
                api_config=api_config,
                config_groups=config_groups
            ),
            processed_configuration=ProcessedConfiguration(
                config=config,
                api_config=api_config,
                config_groups=config_groups,
                current_config_group=current_config_group  
            )
        )
        
        # 保存配置文件路径
        total_config._config_path = resolved_config_path
        
        return total_config
    
    def reload(self, keep_current_group: bool = True) -> None:
        """
        重新加载配置文件
        
        功能说明：
        1. 从磁盘重新读取所有配置文件
        2. 重新验证和处理配置
        3. 可选择保持当前配置组或切换回默认配置组
        
        应用场景：
        - 配置文件被外部程序修改后需要刷新
        - 热重载配置而不重启程序
        - API 密钥更新后立即生效
        
        Args:
            keep_current_group: 是否保持当前配置组
                - True: 重载后继续使用当前配置组（如果该组仍存在）
                - False: 重载后切换回默认配置组
        
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误或当前配置组已被删除
        
        使用示例：
            ```python
            # 场景1：保持当前配置组（默认行为）
            total_config.reload()
            
            # 场景2：切换回默认配置组
            total_config.reload(keep_current_group=False)
            
            # 场景3：配置文件被外部修改后
            # 检测到文件变化...
            total_config.reload()
            print("配置已重载")
            ```
        """
        if self._config_path is None:
            raise RuntimeError("无法重载配置：配置文件路径未保存")
        
        # 保存当前配置组名称（如果需要）
        current_group_name = None
        if keep_current_group:
            current_group_name = self.processed_configuration.current_config_group.group_name
        
        print(f"🔄 开始重载配置文件: {self._config_path}")
        
        try:
            # 重新加载主配置文件
            with open(self._config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                original_config = Original.GeneralConfiguration(**config_data)
                config = Processed.GeneralConfiguration.from_original(original_config)
            
            # 重新加载 API 配置文件
            with open(config.api_config_path, "r", encoding="utf-8") as f:
                api_config_data = json.load(f)
                api_config = Original.APIConfiguration(api_config_data)
            
            # 重新加载配置组文件
            with open(config.config_group_path, "r", encoding="utf-8") as f:
                config_groups_data = json.load(f)
                config_groups = Original.ConfigGroups(config_groups_data)
            
            # 确定使用哪个配置组
            target_group_name = config.default_config_group
            if keep_current_group and current_group_name is not None:
                # 检查原配置组是否仍然存在
                if current_group_name in config_groups:
                    target_group_name = current_group_name
                    print(f"ℹ️  保持当前配置组: {current_group_name}")
                else:
                    print(f"⚠️  警告: 配置组 '{current_group_name}' 已不存在，"
                          f"切换到默认配置组 '{config.default_config_group}'")
            
            # 创建当前配置组
            current_config_group = Processed.CurrentConfigGroup.from_original(
                config_groups,
                api_config,
                target_group_name,
                prompt_dir_path=config.prompt_dir_path
            )
            
            # 更新原始配置
            self.original_configuration = OriginalConfiguration(
                config=original_config,
                api_config=api_config,
                config_groups=config_groups
            )
            
            # 更新处理后的配置
            self.processed_configuration = ProcessedConfiguration(
                config=config,
                api_config=api_config,
                config_groups=config_groups,
                current_config_group=current_config_group
            )
            
            print(f"✅ 配置重载成功")
            print(f"   - 当前配置组: {target_group_name}")
            print(f"   - 可用配置组: {', '.join(config_groups.keys())}")
            
        except FileNotFoundError as e:
            print(f"❌ 配置重载失败: 文件不存在 - {e}")
            raise
        except Exception as e:
            print(f"❌ 配置重载失败: {e}")
            raise
    
    def get_config_file_path(self) -> Path | None:
        """
        获取配置文件路径
        
        Returns:
            配置文件的绝对路径，如果未保存则返回 None
        
        使用示例：
            ```python
            path = total_config.get_config_file_path()
            if path:
                print(f"配置文件位置: {path}")
            ```
        """
        return self._config_path
    
    def switch_config_group(self, group_name: str) -> None:
        """
        切换配置组的便捷方法
        
        这是对 ProcessedConfiguration.switch_config_group 的封装
        
        Args:
            group_name: 目标配置组名称
        
        使用示例：
            ```python
            # 直接在 TotalConfiguration 上切换
            total_config.switch_config_group("high_performance")
            ```
        """
        self.processed_configuration.switch_config_group(group_name)
    
    def get_available_config_groups(self) -> list[str]:
        """
        获取可用配置组列表的便捷方法
        
        Returns:
            配置组名称列表
        """
        return self.processed_configuration.get_available_config_groups()
    
    def get_current_group_name(self) -> str:
        """
        获取当前配置组名称的便捷方法
        
        Returns:
            当前配置组名称
        """
        return self.processed_configuration.get_current_group_name()
    
    def get_data_dir_path(self) -> Path:
        """
        获取数据目录路径的便捷方法
        
        Returns:
            数据目录的绝对路径
        
        使用示例：
            ```python
            data_dir = total_config.get_data_dir_path()
            log_file = data_dir / "logs" / "app.log"
            ```
        """
        return self.processed_configuration.config.data_dir_path


# ============================================================================
# 配置加载流程和测试
# ============================================================================
total_config: TotalConfiguration = TotalConfiguration.init()
if __name__ == "__main__":
    from rich import print
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # 初始化配置
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  配置管理系统测试[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    # 1. 加载配置
    console.print("[bold green]1. 初始化配置[/bold green]")
    total_config = TotalConfiguration.init()
    console.print(f"✓ 配置文件路径: {total_config.get_config_file_path()}")
    console.print(f"✓ 当前配置组: {total_config.get_current_group_name()}")
    console.print(f"✓ 可用配置组: {', '.join(total_config.get_available_config_groups())}\n")
    
    # 2. 显示当前配置信息
    console.print("[bold green]2. 当前配置详情[/bold green]")
    current = total_config.processed_configuration.current_config_group
    
    table = Table(title="当前聊天模型配置", show_header=True)
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="yellow")
    
    table.add_row("配置组", current.group_name)
    table.add_row("Provider Type", current.chat_model.provider_type)
    table.add_row("模型ID", current.chat_model.model_id)
    table.add_row("Base URL", current.chat_model.base_url)
    table.add_row("最大输入Token", str(current.chat_model.max_input_tokens))
    table.add_row("支持视觉", "✓" if current.chat_model.supports_vision else "✗")
    table.add_row("支持音频", "✓" if current.chat_model.supports_audio else "✗")
    
    console.print(table)
    console.print()
    
    # 3. 测试配置组切换
    console.print("[bold green]3. 测试配置组切换[/bold green]")
    available_groups = total_config.get_available_config_groups()
    
    if len(available_groups) > 1:
        # 切换到第二个配置组
        target_group = available_groups[1]
        console.print(f"尝试切换到配置组: {target_group}")
        total_config.switch_config_group(target_group)
        console.print(f"✓ 当前配置组: {total_config.get_current_group_name()}")
        
        # 切换回第一个配置组
        console.print(f"\n尝试切换回配置组: {available_groups[0]}")
        total_config.switch_config_group(available_groups[0])
        console.print(f"✓ 当前配置组: {total_config.get_current_group_name()}\n")
    else:
        console.print("[yellow]⚠ 只有一个配置组，跳过切换测试[/yellow]\n")
    
    # 4. 测试重载配置
    console.print("[bold green]4. 测试配置重载[/bold green]")
    console.print("保持当前配置组重载...")
    total_config.reload(keep_current_group=True)
    
    console.print("\n切换到默认配置组重载...")
    total_config.reload(keep_current_group=False)
    console.print()
    
    # 5. 测试错误处理
    console.print("[bold green]5. 测试错误处理[/bold green]")
    try:
        total_config.switch_config_group("non_existent_group")
    except ValueError as e:
        console.print(f"✓ 正确捕获错误: [red]{e}[/red]\n")
    
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  测试完成[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

