

import asyncio
import time
import sys
import os
import json
import aiofiles
import tomli
from pathlib import Path
from datetime import datetime
from typing import Any, TypeVar, Optional, Union
from nonebot import get_driver, logger, on_command
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, Message, MessageSegment, Event
from nonebot.rule import to_me

dir_path = Path(__file__).parent
sys.path.append(str(dir_path))
from Permissions import owner, admin

# 默认提示词
default_character_prompt = """# 角色设定
你是一只具有猫和人属性的猫娘, 名字叫发情猫娘.
你的回答十分可爱且简洁. 常使用emoji和颜文字."""
default_output_format_prompt = """
# 关于 CQ 码
CQ 码基本格式: [CQ:类型,参数=值,参数=值]
@某人: [CQ:at, qq= 用户QQ号]
回复或引用消息: [CQ:reply, id= 消息ID]
戳一戳或rua或捏: [CQ:poke,qq= QQ号]
熟练使用CQ码可以更好的与用户进行互动, 

# 关于输入

对于聊天记录的输入一般格式为: [时间] [用户名](用户QQ号, 消息ID) [消息内容]
如果用户名为QQ_system且QQ号和消息ID为-1, 代表这是QQ系统消息, 如新人加群, 戳一戳等, 此类消息一般无法引用.
# 输出格式规范

你的每次回复必须包含以下 2 个部分：

## 1. 思考部分（[THINKING]）

- 分析用户请求的核心需求
- 规划问题解决路径

## 2. 输出部分（[RESPONSE]）

- 直接返回给用户的最终内容
- 内容要求自然流畅的对话风格

# 示例输出

```input
你好
```

```output
[THINKING]
用户请求问候，我应当以简洁的形式回复问候语
[RESPONSE]
你好！有什么可以帮助你的吗？
```

"""

class ImageRecognitionConfig:
    """图片识别配置类"""
    def __init__(
        self,
        api_config_data: dict = {},
        prompt_dir: Path|str = "prompts",
        enable: bool = False,
        ai_model: str = "openrouter_proxy/qwen2.5-vl-32b-instruct",
        prompt_path: str|Path = "识图模型提示词/default.txt",
        max_concurrent_tasks: int = 2,
        max_api_calls_per_recognition: int = 2,
        max_image_size_mb: float = 5,
        cache_db_path: Path|str = "data/image_cache.db",
        processing_emoji_id: int = 10025
    ):
        self.api_config_data = api_config_data
        self.prompt_dir = prompt_dir
        self.enable = enable
        self.ai_model = ai_model
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_api_calls_per_recognition = max_api_calls_per_recognition
        self.max_image_size_mb = max_image_size_mb
        self.cache_db_path = cache_db_path
        self.processing_emoji_id = processing_emoji_id
        self.prompt_path = prompt_path
        self.prompt: str = ""
        self.model_provider:str = ""
        self.model_name = ""

    def _process_configs(self):
        # 路径处理
        if isinstance(self.prompt_path, str):
            self.prompt_path = Path(self.prompt_path)
        if isinstance(self.cache_db_path, str):
            self.cache_db_path = Path(self.cache_db_path)
        if isinstance(self.prompt_dir, str):
            self.prompt_dir = Path(self.prompt_dir)
        if not self.cache_db_path.is_absolute():
            self.cache_db_path = dir_path/self.cache_db_path
        if not self.prompt_dir.is_absolute():
            self.prompt_dir = dir_path/self.prompt_dir
        logger.debug(f"识图缓存数据库路径: {self.cache_db_path}")
        if not self.prompt_path.is_absolute():
            self.prompt_path = self.prompt_dir/self.prompt_path

        # 解析模型配置
        try:
            self.model_provider, self.model_name = self.ai_model.split("/", 1)
            logger.info(f"使用模型提供者: {self.model_provider}, 模型名: {self.model_name}")
        except ValueError:
            logger.error(f"模型配置格式错误: {self.ai_model}，应为'provider/model'格式")
            self.model_provider = "openrouter_proxy"
            self.model_name = "qwen2.5-vl-32b-instruct"
        
        # API设置
        self.chat_ai_model = "deepseek/deepseek-chat-v3-0324:free"
        self.chat_ai_url = "http://127.0.0.1:30001/openrouter"
        self.chat_ai_key = ""
        
        # 从API配置文件中获取设置
        try:
            if self.model_provider in self.api_config_data:
                self.provider_config = self.api_config_data[self.model_provider]
                self.chat_ai_url = self.provider_config.get("base_url", self.chat_ai_url)
                self.chat_ai_key = self.provider_config.get("api_key", self.chat_ai_key)

                if "models" in self.provider_config and self.model_name in self.provider_config["models"]:
                    self.model_mapping = self.provider_config["models"][self.model_name]
                    self.chat_ai_model = self.model_mapping.get("model", self.chat_ai_model)

                logger.info(f"成功从API配置文件加载API信息: URL={self.chat_ai_url}, MODEL={self.chat_ai_model}")
            else:
                logger.warning(f"API配置文件中不存在提供者: {self.model_provider}，将使用默认配置")
        except Exception as e:
            logger.error(f"读取API配置失败: {str(e)}，将使用默认配置")

        # 仅读取图像识别专用提示词
        if self.prompt_path.exists():
            try:
                with open(self.prompt_path, "r", encoding="utf-8") as f:
                    self.prompt = f.read()
                logger.info(f"成功加载图像识别提示词: {self.prompt_path}")
            except Exception as e:
                logger.error(f"读取图像提示词失败: {str(e)}")
                self.prompt = ""
        else:
            logger.warning(f"图像识别提示词文件不存在: {self.prompt_path}")
            self.prompt = ""

class MainAIConfig:
    """主AI配置类"""
    def __init__(self):
        self.main_model: str = "openrouter_proxy/deepseek-v3"
        self.temperature: float = 1.0
        self.main_output_prompt: str = "输出格式提示词/default.txt"
        self.main_character_prompt: str = "角色提示词/天天.txt"
        self.model_provider: str = ""
        self.model_name: str = ""
        self.chat_ai_model: str = ""
        self.chat_ai_url: str = ""
        self.chat_ai_key: str = ""
        
    def process_config(self, api_config_data: dict):
        """处理AI配置"""
        # 解析模型配置
        try:
            self.model_provider, self.model_name = self.main_model.split("/", 1)
            logger.info(f"使用模型提供者: {self.model_provider}, 模型名: {self.model_name}")
        except ValueError:
            logger.error(f"模型配置格式错误: {self.main_model}，应为'provider/model'格式")
            self.model_provider = "openrouter_proxy"
            self.model_name = "deepseek-v3"
            
        # 默认API设置
        self.chat_ai_model = "deepseek/deepseek-chat-v3-0324:free"
        self.chat_ai_url = "http://127.0.0.1:30001/openrouter"
        self.chat_ai_key = ""
        
        # 从API配置文件中获取设置
        try:
            if self.model_provider in api_config_data:
                provider_config = api_config_data[self.model_provider]
                self.chat_ai_url = provider_config.get("base_url", self.chat_ai_url)
                self.chat_ai_key = provider_config.get("api_key", self.chat_ai_key)

                if "models" in provider_config and self.model_name in provider_config["models"]:
                    model_mapping = provider_config["models"][self.model_name]
                    self.chat_ai_model = model_mapping.get("model", self.chat_ai_model)

                logger.info(f"成功加载API信息: URL={self.chat_ai_url}, MODEL={self.chat_ai_model}")
            else:
                logger.warning(f"API配置文件中不存在提供者: {self.model_provider}，将使用默认配置")
        except Exception as e:
            logger.error(f"读取API配置失败: {str(e)}，将使用默认配置")

class EmojiResponseConfig:
    """表情回应配置类"""
    def __init__(self):
        self.enable: bool = True
        self.processing_emoji_id: int = 10024
        self.processing_success_emoji_id: int = 320
        self.processing_rejected_emoji_id: int = 10060
        self.remove_after_complete: bool = True

class MessageHandlingConfig:
    """消息处理配置类"""
    def __init__(self):
        self.chat_record_db_path: str|Path = "data/data.db"
        self.enable_streaming: bool = True
        self.max_chat_history_limit: int = 15
        self.max_single_message_length: int = 300
        self.max_extra_ai_messages: int = 10
        self.max_extra_qq_system_messages: int = 5
        self.replace_my_name_with_me: bool = True
        self.min_send_interval: float = 0.4
        self.send_interval_random_range: float = 0.2
        self.QQ_system_name: str = "QQ_system"
        self.QQ_system_message_id: int = -1
        self.inject_group_metadata = False
        self.inject_user_metadata = False
        
    def process_paths(self):
        """处理路径配置"""
        self.chat_record_db_path = Path(self.chat_record_db_path)
        if not self.chat_record_db_path.is_absolute():
            self.chat_record_db_path = dir_path / self.chat_record_db_path
        logger.debug(f"数据库路径: {self.chat_record_db_path}")

class PromptConfig:
    """提示词配置类"""
    def __init__(self):
        self.prompt_dir: Path = Path("prompts")
        self.character_prompt: str = default_character_prompt
        self.output_format_prompt: str = default_output_format_prompt
        self.full_prompt: str = self.character_prompt + self.output_format_prompt
        
    def load_prompts(self, character_file: str, output_file: str):
        """加载提示词文件"""
        # 处理路径
        if not self.prompt_dir.is_absolute():
            self.prompt_dir = dir_path / self.prompt_dir
        logger.info(f"提示词目录: {self.prompt_dir}")
        
        # 读取角色提示词
        char_path = self.prompt_dir / character_file
        if char_path.exists():
            try:
                with open(char_path, "r", encoding="utf-8") as f:
                    self.character_prompt = f.read()
                logger.info(f"成功加载角色提示词: {char_path}")
            except Exception as e:
                logger.error(f"读取角色提示词失败: {str(e)}")
        else:
            logger.warning(f"角色提示词文件不存在: {char_path}")
        
        # 读取输出格式提示词
        out_path = self.prompt_dir / output_file
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    self.output_format_prompt = f.read()
                logger.info(f"成功加载输出格式提示词: {out_path}")
            except Exception as e:
                logger.error(f"读取输出格式提示词失败: {str(e)}")
        else:
            logger.warning(f"输出格式提示词文件不存在: {out_path}")
        
        # 组合提示词
        self.full_prompt = self.character_prompt + self.output_format_prompt

class ConfigGroupManager:
    """配置文件管理器"""
    def __init__(
        self, 
        config_group: str = "default", 
        config_group_path: Path = dir_path/"config_group.toml", 
        api_config_path: Path = dir_path/"api_config.json",
    ):
        self.config_group = config_group
        self.config_group_path = config_group_path 
        self.api_config_path = api_config_path
        
        # 初始化配置对象
        self.main_ai = MainAIConfig()
        self.image_recognition = ImageRecognitionConfig()
        self.emoji_response = EmojiResponseConfig()
        self.message_handling = MessageHandlingConfig()
        self.prompt = PromptConfig()
        
        # 其他配置
        self.api_config_data: dict = {}

    def load_toml_config_sync(self):
        """同步加载TOML配置文件"""
        try:
            if self.config_group_path.exists():
                with open(self.config_group_path, "rb") as f:
                    self.config_data = tomli.load(f)
                logger.info(f"成功加载TOML配置文件: {self.config_group_path}")
            else:
                logger.warning(f"TOML配置文件不存在: {self.config_group_path}，将使用默认配置")
                self.config_data = {}
        except Exception as e:
            logger.error(f"读取TOML配置文件失败: {str(e)}，将使用默认配置")
            self.config_data = {}

    def load_api_config_sync(self):
        """同步加载API配置文件"""
        try:
            if self.api_config_path.exists():
                with open(self.api_config_path, "r", encoding="utf-8") as f:
                    self.api_config_data = json.load(f)
                logger.info(f"成功加载API配置文件: {self.api_config_path}")
            else:
                logger.warning(f"API配置文件不存在: {self.api_config_path}，将使用默认配置")
                self.api_config_data = {}
        except Exception as e:
            logger.error(f"读取API配置文件失败: {str(e)}，将使用默认配置")
            self.api_config_data = {}

    def _get_config_value(self, path: str, default: Any = None) -> Any:
        """通过点路径获取嵌套配置值"""
        keys = path.split('.')
        current = self.config_data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def load_sync(self):
        """同步加载配置"""
        self.load_toml_config_sync()
        self.load_api_config_sync()
        self._process_loaded_configs()

        self.api_config_data: dict = {}

    def apply_config_with_logging(self, config_dict: dict, key: str, target_obj: object, target_attr: str, config_path: str):
        """应用配置项并记录缺失情况
        
        从配置字典中读取指定键的值，并将其设置到目标对象的指定属性上。
        如果配置项不存在，则保持目标对象的默认值不变，并记录错误日志。
        
        Args:
            config_dict (dict): 包含配置项的字典
            key (str): 要查找的配置项键名
            target_obj (object): 目标对象，其属性将被设置
            target_attr (str): 目标对象上要设置的属性名
            config_path (str): 配置路径，用于日志记录中标识配置项位置
            
        Returns:
            None
            
        Raises:
            AttributeError: 当target_obj不存在target_attr属性时抛出
            
        Example:
            >>> config = {"timeout": 30, "retries": 3}
            >>> obj = SomeClass()
            >>> self.apply_config_with_logging(
            ...     config_dict=config,
            ...     key="timeout", 
            ...     target_obj=obj,
            ...     target_attr="connection_timeout",
            ...     config_path="network.connection"
            ... )
            # 将config["timeout"]的值30设置给obj.connection_timeout
            
        Note:
            - 如果配置项存在，直接设置属性值
            - 如果配置项不存在，保持原有默认值并记录错误日志
            - 错误日志格式：'配置项 {config_path}.{key} 未找到，使用默认值：{default_value}'
        """
        # 检查配置字典中是否存在指定键
        if key in config_dict:
            # 如果存在，则将配置值赋给目标对象的属性
            setattr(target_obj, target_attr, config_dict[key])
        else:
            # 如果不存在，则获取目标对象的默认值
            default_value = getattr(target_obj, target_attr)
            # 记录错误日志，提示配置项未找到，并使用默认值
            logger.error(f"配置项 {config_path}.{key} 未找到，使用默认值：{default_value}")

    def _process_loaded_configs(self):
        """处理已加载的配置"""
        # 获取当前配置组的数据
        group_data = self._get_config_value(self.config_group, {})
        
        # 主AI配置
        ai_data = self._get_config_value(f"{self.config_group}.main_ai", {})
        if not ai_data:
            logger.error(f"配置组 {self.config_group}.main_ai 未找到，所有主AI配置将使用默认值")
        else:
            self.apply_config_with_logging(ai_data, "main_model", self.main_ai, "main_model", f"{self.config_group}.main_ai")
            self.apply_config_with_logging(ai_data, "temperature", self.main_ai, "temperature", f"{self.config_group}.main_ai")
            self.apply_config_with_logging(ai_data, "main_output_prompt", self.main_ai, "main_output_prompt", f"{self.config_group}.main_ai")
            self.apply_config_with_logging(ai_data, "main_character_prompt", self.main_ai, "main_character_prompt", f"{self.config_group}.main_ai")
        self.main_ai.process_config(self.api_config_data)
        
        # 图像识别配置
        img_data = self._get_config_value(f"{self.config_group}.image_recognition", {})
        if not img_data:
            logger.error(f"配置组 {self.config_group}.image_recognition 未找到，所有图像识别配置将使用默认值")
        else:
            self.apply_config_with_logging(img_data, "enable", self.image_recognition, "enable", f"{self.config_group}.image_recognition")
            self.apply_config_with_logging(img_data, "ai_model", self.image_recognition, "ai_model", f"{self.config_group}.image_recognition")
            self.apply_config_with_logging(img_data, "prompt_path", self.image_recognition, "prompt_path", f"{self.config_group}.image_recognition")
            self.apply_config_with_logging(img_data, "max_concurrent_tasks", self.image_recognition, "max_concurrent_tasks", f"{self.config_group}.image_recognition")
            self.apply_config_with_logging(img_data, "max_image_size_mb", self.image_recognition, "max_image_size_mb", f"{self.config_group}.image_recognition")
            self.apply_config_with_logging(img_data, "cache_db_path", self.image_recognition, "cache_db_path", f"{self.config_group}.image_recognition")
            self.apply_config_with_logging(img_data, "processing_emoji_id", self.image_recognition, "processing_emoji_id", f"{self.config_group}.image_recognition")
            self.apply_config_with_logging(img_data, "max_api_calls_per_recognition", self.image_recognition, "max_api_calls_per_recognition", f"{self.config_group}.image_recognition")
        self.image_recognition.api_config_data = self.api_config_data
        self.image_recognition.prompt_dir = self.prompt.prompt_dir
        self.image_recognition._process_configs()
        
        # 表情回应配置
        emoji_data = self._get_config_value(f"{self.config_group}.emoji_response", {})
        if not emoji_data:
            logger.error(f"配置组 {self.config_group}.emoji_response 未找到，所有表情回应配置将使用默认值")
        else:
            self.apply_config_with_logging(emoji_data, "enable_emoji_response", self.emoji_response, "enable", f"{self.config_group}.emoji_response")
            self.apply_config_with_logging(emoji_data, "processing_emoji_id", self.emoji_response, "processing_emoji_id", f"{self.config_group}.emoji_response")
            self.apply_config_with_logging(emoji_data, "processing_success_emoji_id", self.emoji_response, "processing_success_emoji_id", f"{self.config_group}.emoji_response")
            self.apply_config_with_logging(emoji_data, "processing_rejected_emoji_id", self.emoji_response, "processing_rejected_emoji_id", f"{self.config_group}.emoji_response")
            self.apply_config_with_logging(emoji_data, "remove_processing_emoji_after_complete", self.emoji_response, "remove_after_complete", f"{self.config_group}.emoji_response")
        
        # 消息处理配置
        msg_data = self._get_config_value(f"{self.config_group}.message_handling", {})
        if not msg_data:
            logger.error(f"配置组 {self.config_group}.message_handling 未找到，所有消息处理配置将使用默认值")
        else:
            self.apply_config_with_logging(msg_data, "chat_record_db_path", self.message_handling, "chat_record_db_path", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "enable_streaming", self.message_handling, "enable_streaming", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "max_chat_history_limit", self.message_handling, "max_chat_history_limit", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "max_single_message_length", self.message_handling, "max_single_message_length", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "max_extra_ai_messages", self.message_handling, "max_extra_ai_messages", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "max_extra_qq_system_messages", self.message_handling, "max_extra_qq_system_messages", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "replace_my_name_with_me", self.message_handling, "replace_my_name_with_me", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "min_send_interval", self.message_handling, "min_send_interval", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "send_interval_random_range", self.message_handling, "send_interval_random_range", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "QQ_system_name", self.message_handling, "QQ_system_name", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "QQ_system_message_id", self.message_handling, "QQ_system_message_id", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "inject_group_metadata", self.message_handling, "inject_group_metadata", f"{self.config_group}.message_handling")
            self.apply_config_with_logging(msg_data, "inject_user_metadata", self.message_handling, "inject_user_metadata", f"{self.config_group}.message_handling")
        self.message_handling.process_paths()
        
        # 提示词配置
        # 1. 提示词目录
        if "prompt_dir" in group_data:
            self.prompt.prompt_dir = Path(group_data["prompt_dir"])
            logger.info(f"提示词目录配置为: {self.prompt.prompt_dir}")
        else:
            logger.error(f"配置项 {self.config_group}.prompt_dir 未找到，使用默认值: {self.prompt.prompt_dir}")
        
        # 2. 角色提示词文件路径和输出格式提示词文件路径
        character_file = self.main_ai.main_character_prompt
        output_file = self.main_ai.main_output_prompt
        
        # 加载提示词
        self.prompt.load_prompts(character_file, output_file)

CONFIG_PATH = dir_path/"config.json"

def read_config(config_path: Path = CONFIG_PATH) -> dict:
    """读取基础配置文件"""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def write_config(config: dict, config_path: Path = CONFIG_PATH) -> None:
    """写入基础配置文件"""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

# 初始化全局配置管理器
config = read_config()
config_group = config.get("default_group", "default")
config_group_path = Path(config.get("config_group_path", "config_group.toml"))
if not config_group_path.is_absolute():
    config_group_path = dir_path/config_group_path
api_config_path = Path(config.get("api_config_path", "api_config.json"))
if not api_config_path.is_absolute():
    api_config_path = dir_path/api_config_path

logger.info(f"加载配置文件成功，当前配置组为：{config_group}")

# 创建配置管理器并加载配置
global_config_manager = ConfigGroupManager(
    config_group, 
    config_group_path, 
    api_config_path, 
)
global_config_manager.load_sync()
logger.info(f"配置组数据已加载: {global_config_manager.config_group}")

# 命令处理器
reload_config = on_command("reload_config", 
                          aliases={"重载配置", "重载配置文件"}, 
                          priority=-5, block=True)

@reload_config.handle()
async def handle_reload_config(bot: Bot, event: Event):
    global global_config_manager
    if not (owner(bot, event) or admin(bot, event)):
        await reload_config.finish("权限不足，无法重载配置文件")
    
    # 重载配置
    global_config_manager.load_sync()
    await reload_config.finish(f"重载配置文件成功，当前配置组为：{global_config_manager.config_group}")
