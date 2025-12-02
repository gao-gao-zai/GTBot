from functools import total_ordering
import os
from pydantic import BaseModel, Field, RootModel, field_validator, ValidationInfo
from typing import Optional
import json
from pathlib import Path
import sys

# 获取当前文件所在目录的绝对路径
DIR_PATH = Path(__file__).parent.resolve()

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
        user_cache_update_interval_sec: int = 3600
        """用户缓存刷新间隔（秒）"""
        user_cache_expire_sec: int = 604800
        """用户缓存最长保留时长（秒）"""
    
    class Provider(BaseModel):
        """单个服务提供商的配置"""
        
        class LLMModel(BaseModel):
            """单个大语言模型的配置"""
            model: str
            """上游模型名称/ID（如 gpt-4, claude-3-opus 等）"""
            max_input_tokens: int
            """允许输入的最大token数"""
            supports_vision: bool
            """是否支持视觉输入（图片理解）"""
            supports_audio: bool
            """是否支持音频输入（语音识别）"""
            parameters: dict[str, str | dict | list | bool | int | float]
            """自定义模型参数（如 temperature, top_p 等）"""
        
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
            return self.root[key]
        
        def __contains__(self, key: str) -> bool:
            """支持 in 运算符: 'openai' in api_config"""
            return key in self.root
        
        @field_validator('root')
        @classmethod
        def check_keys(cls, v: dict):
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
            """聊天模型配置"""
            model: str
            """模型标识符，格式为 'provider/model'（如 'openai/gpt-4'）"""
            maximum_number_of_incoming_messages: int
            """最大输入消息数（用于控制上下文长度）"""
            behavioral_prompt: str
            """行为提示词文件路径"""
            character_prompt: str
            """角色提示词文件路径"""
            
        chat_model: ChatModel
        """聊天模型配置"""
    
    class ConfigGroups(RootModel[dict[str, SingleConfigurationGroup]]):
        """
        配置组容器 - 存储所有配置组
        key: 配置组名称（如 'default', 'high_performance'）
        value: 配置组对象
        """
        def __getitem__(self, key: str) -> "Original.SingleConfigurationGroup":
            """支持字典式访问: config_groups['default']"""
            return self.root[key]
        
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
        user_cache_update_interval_sec: int
        """用户缓存刷新间隔（秒）"""
        user_cache_expire_sec: int
        """用户缓存最长保留时间（秒）"""
        
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
            
            # api_config_path 和 config_groups_path 是相对于 DIR_PATH 的
            return cls(
                api_config_path=cls.check_path(original.api_config_path, base_path=DIR_PATH),
                config_group_path=cls.check_path(original.config_groups_path, base_path=DIR_PATH), 
                default_config_group=original.default_config_group,
                prompt_dir_path=prompt_dir_path,
                data_dir_path=data_dir_path,
                user_cache_update_interval_sec=update_interval,
                user_cache_expire_sec=expire_interval,
            )
    
    class CurrentConfigGroup(BaseModel):
        """
        当前激活的配置组
        将配置组和 API 配置合并，提供完整的运行时配置
        """
        
        class ChatModel(BaseModel):
            """完整的聊天模型配置 - 包含所有运行时所需信息"""
            model_id: str
            """上游模型的实际ID（从 API 配置中提取）"""
            base_url: str
            """API 基础 URL"""
            api_key: str
            """API 密钥"""
            max_input_tokens: int
            """允许输入的最大token数"""
            supports_vision: bool
            """是否支持视觉输入"""
            supports_audio: bool
            """是否支持音频输入"""
            parameters: dict[str, str | dict | list | bool | int | float]
            """自定义模型参数"""
            behavioral_prompt: str
            """行为提示词内容"""
            character_prompt: str
            """角色提示词内容"""
            prompt: str
            """最终拼接的提示词内容"""
        
        chat_model: ChatModel
        """聊天模型的完整配置"""
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
            behavioral_prompt_path = original.chat_model.behavioral_prompt
            character_prompt_path = original.chat_model.character_prompt
            
            if not isinstance(behavioral_prompt_path, Path):
                behavioral_prompt_path = Path(behavioral_prompt_path)
            if not isinstance(character_prompt_path, Path):
                character_prompt_path = Path(character_prompt_path)
            
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
            
            # 合并配置信息创建当前配置组
            return cls(
                chat_model=cls.ChatModel(
                    model_id=api_config[provider].llm_models[model].model,  
                    base_url=api_config[provider].base_url,
                    api_key=api_config[provider].api_key,
                    max_input_tokens=api_config[provider].llm_models[model].max_input_tokens, 
                    supports_vision=api_config[provider].llm_models[model].supports_vision,  
                    supports_audio=api_config[provider].llm_models[model].supports_audio, 
                    parameters=api_config[provider].llm_models[model].parameters, 
                    behavioral_prompt=behavioral_prompt,
                    character_prompt=character_prompt,
                    prompt=prompt
                ),
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
    api_config: Processed.APIConfiguration
    """API 配置"""
    config_groups: Processed.ConfigGroups
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
        if isinstance(config_path, str):
            config_path = Path(config_path)
        if config_path is None:
            config_path = DIR_PATH / "config" / "config.json"
        
        # 确保路径是绝对路径
        if not config_path.is_absolute():
            config_path = config_path.resolve()
        
        # 加载配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            # 解析为原始配置对象
            original_config = Original.GeneralConfiguration(**config_data)
            # 转换为处理后的配置（验证路径）
            config = Processed.GeneralConfiguration.from_original(original_config)
        
        # 加载 API 配置文件
        with open(config.api_config_path, "r", encoding="utf-8") as f:
            api_config_data = json.load(f)
            # 解析 API 配置
            api_config: Processed.APIConfiguration = Original.APIConfiguration(api_config_data)
        
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
        total_config._config_path = config_path
        
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
else:
    total_config: TotalConfiguration = TotalConfiguration.init()