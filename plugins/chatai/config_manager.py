import json
from pathlib import Path
import tomllib
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Union, get_type_hints
import inspect
import toml
from pydantic import BaseModel, Field
import threading


try:
    from nonebot import get_driver
    get_driver()
    del get_driver
    from nonebot import logger as logging
    NONE_BOT_ENV = True
except ValueError:
    import logging
    NONE_BOT_ENV = False



import sys


dir_path = Path(__file__).parent
sys.path.append(str(dir_path))


ROOT_CONFIG_PATH = dir_path / "config.toml"
"""根配置文件路径"""

def load_toml_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到 {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return toml.load(f)

def load_text_file(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"文件未找到 {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()



def convert_to_absolute_path(path: Union[str, Path], base_path: Union[str, Path, None] = dir_path):
    if isinstance(path, str):
        path = Path(path)
    if isinstance(base_path, str):
        base_path = Path(base_path)
    if not base_path:
        base_path = dir_path
    if not path.is_absolute():
        path = base_path / path
    return path


ROOT_CONFIG_DATA = load_toml_config(ROOT_CONFIG_PATH)
"""根配置数据"""


DEFAULT_CONFIG_GROUP_NAME = ROOT_CONFIG_DATA["default_config_group"]
"""默认配置组名称"""
config_group_file_path: Path = Path(ROOT_CONFIG_DATA["config_group_path"])
CONFIG_GROUP_FILE_PATH = convert_to_absolute_path(config_group_file_path)
"""配置组文件路径"""
api_config_file_path = Path(ROOT_CONFIG_DATA["api_config_path"])
API_CONFIG_FILE_PATH = convert_to_absolute_path(api_config_file_path)
"""API配置文件路径"""
defailt_config_group_file_path = Path(ROOT_CONFIG_DATA["default_config_group_path"])
DEFAULT_CONFIG_GROUP_FILE_PATH = convert_to_absolute_path(defailt_config_group_file_path)
"""默认配置文件路径（包含所有配置项及其默认值）"""








class DotDict(dict):
    """支持点号访问的字典类"""
  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_nested()
  
    def _convert_nested(self):
        """递归转换嵌套的字典为DotDict"""
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, DotDict):
                self[key] = DotDict(value)
            elif isinstance(value, list):
                self[key] = self._convert_list(value)
  
    def _convert_list(self, lst):
        """转换列表中的字典元素"""
        new_list = []
        for item in lst:
            if isinstance(item, dict) and not isinstance(item, DotDict):
                new_list.append(DotDict(item))
            elif isinstance(item, list):
                new_list.append(self._convert_list(item))
            else:
                new_list.append(item)
        return new_list
  
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
  
    def __setattr__(self, key, value):
        self[key] = value
  
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
  
    def to_dict(self):
        """转换回普通字典"""
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = self._list_to_dict(value)
            else:
                result[key] = value
        return result
  
    def _list_to_dict(self, lst):
        """将列表中的DotDict转换回字典"""
        result = []
        for item in lst:
            if isinstance(item, DotDict):
                result.append(item.to_dict())
            elif isinstance(item, list):
                result.append(self._list_to_dict(item))
            else:
                result.append(item)
        return result




# 加载默认配置数据
try:
    DEFAULT_CONFIG_DATA = DotDict(load_toml_config(DEFAULT_CONFIG_GROUP_FILE_PATH).get("default", {}))
    if DEFAULT_CONFIG_DATA == {}:
        raise RuntimeError("默认配置文件为空")
except FileNotFoundError:
    raise RuntimeError(f"默认配置文件未找到: {DEFAULT_CONFIG_GROUP_FILE_PATH}") from None



class _MainAiConfig:
    main_model: str
    temperature: float
    main_output_prompt: str
    main_character_prompt: str
    is_reasoning_model: bool

class _ImageRecognitionConfig:
    enable: bool
    ai_model: str
    is_reasoning_model: bool
    prompt_path: str
    max_concurrent_tasks: int
    max_api_calls_per_recognition: int
    max_image_size_mb: float
    cache_db_path: Path
    processing_emoji_id: int
  
class _EmojiResponseConfig:
    enable_emoji_response: bool
    processing_emoji_id: int
    processing_success_emoji_id: int
    processing_rejected_emoji_id: int
    remove_processing_emoji_after_complete: bool

class _MessageHandlingConfig:
    chat_record_db_path: Path
    enable_streaming: bool
    max_chat_history_limit: int
    max_single_message_length: int
    max_extra_ai_messages: int
    max_extra_qq_system_messages: int
    replace_my_name_with_me: bool
    min_send_interval: float
    send_interval_random_range: float
    QQ_system_message_id: int
    QQ_system_name: str
    inject_group_metadata: bool
    inject_user_metadata: bool

class _APIConfig:
    chat_model_provider: str
    chat_model_name: str
    chat_ai_url: str
    chat_ai_key: str
    chat_ai_model: str
    chat_model_is_reasoning: bool
    """实际模型ID"""
    image_model_provider: str
    image_model_name: str
    image_ai_url: str
    image_ai_key: str
    image_ai_model: str
    image_model_is_reasoning: bool

class _PromptConfig:
    prompt_dir: Path
    chat_character_prompt: str
    chat_character_prompt_file: Path
    chat_output_format_prompt: str
    chat_output_format_prompt_file: Path
    chat_full_prompt: str
    """完整聊天AI提示词"""
    image_recognition_prompt_file: Path
    image_recognition_prompt: str


class ConfigGroup(DotDict):
    """配置类"""
    main_ai: _MainAiConfig
    image_recognition: _ImageRecognitionConfig
    emoji_response: _EmojiResponseConfig
    message_handling: _MessageHandlingConfig
    prompt: _PromptConfig
    api: _APIConfig
    prompt_dir: Path

    def __init__(
        self, 
        config_group_name: str = DEFAULT_CONFIG_GROUP_NAME, 
        config_group_file_path: Path = CONFIG_GROUP_FILE_PATH,
        api_config_file_path: Path = API_CONFIG_FILE_PATH,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config_group_name = config_group_name
        self.config_group_file_path = config_group_file_path
        self.api_config_file_path = api_config_file_path
        self.api = _APIConfig()
        self.prompt = _PromptConfig()

            
        self.check_configuration_item_integrity()
        self.path_configuration_handling()
        self.API_Configuration_Handling()
        self.Prompt_Configuration_Handling()

    def check_configuration_item_integrity(self):
        """检查配置项的完整性，使用默认配置文件填充缺失项"""
        self._merge_defaults(self, DEFAULT_CONFIG_DATA, prefix="")
    
    def _merge_defaults(self, current, defaults, prefix):
        """
        递归合并默认配置到当前配置
        :param current: 当前配置（DotDict）
        :param defaults: 默认配置（DotDict）
        :param prefix: 当前路径前缀（用于日志）
        """
        for key, default_value in defaults.items():
            full_path = f"{prefix}.{key}" if prefix else key
            
            if key not in current:
                # 直接设置默认值
                current[key] = default_value
                logging.warning(f"配置项 {full_path} 缺失，已自动添加默认值: {default_value}")
            else:
                # 检查嵌套字典
                if isinstance(default_value, dict) and isinstance(current[key], dict):
                    self._merge_defaults(
                        current[key], 
                        default_value, 
                        prefix=full_path
                    )
                # 处理列表中的字典（如果需要）
                elif isinstance(default_value, list) and isinstance(current[key], list):
                    self._merge_list_defaults(current[key], default_value, full_path)
    
    def _merge_list_defaults(self, current_list, default_list, prefix):
        """处理列表中的默认值（仅处理字典元素）"""
        # 只处理第一个元素作为模板（假设列表结构一致）
        if default_list and isinstance(default_list[0], dict):
            for i, item in enumerate(current_list):
                if i < len(default_list) and isinstance(item, dict):
                    self._merge_defaults(
                        item, 
                        default_list[i], 
                        prefix=f"{prefix}[{i}]"
                    )
    
    @classmethod
    def from_config_group_name(cls, config_group_name: str = DEFAULT_CONFIG_GROUP_NAME):
        """
        从配置组名称构建Config对象
        """
        # 加载配置组文件
        group_config_data = load_toml_config(CONFIG_GROUP_FILE_PATH)
      
        # 获取指定配置组的数据
        if config_group_name in group_config_data:
            config_data = group_config_data[config_group_name]
        else:
            logging.warning(f"配置组 '{config_group_name}' 未找到，使用空配置")
            config_data = {}
      
        # 创建Config实例
        config = cls(
            config_group_name=config_group_name,
            config_group_file_path=CONFIG_GROUP_FILE_PATH,
            api_config_file_path=API_CONFIG_FILE_PATH,
            **config_data
        )
      
        return config




    def API_Configuration_Handling(self, api_config_file_path: Path = API_CONFIG_FILE_PATH):
        """处理API相关的配置"""


        self.api.chat_model_is_reasoning = self.main_ai.is_reasoning_model
        self.api.image_model_is_reasoning = self.image_recognition.is_reasoning_model

        
        # 读取api配置
        if not api_config_file_path.exists():
            raise FileNotFoundError(f"API配置文件未找到: {api_config_file_path}")
        with open(api_config_file_path, "r", encoding="utf-8") as f:
            self.api_config_data: dict = json.load(f)
        

        
        try:
            self.api.chat_model_provider, self.api.chat_model_name = self.main_ai.main_model.split("/", 1)
        except ValueError:
            logging.error(f"模型配置格式错误: {self.main_ai.main_model}，应为'provider/model'格式")
            self.api.chat_model_provider = "openrouter_proxy"
            self.api.chat_model_name = "deepseek-v3"


        if self.api.chat_model_provider in self.api_config_data:
            provider_config = self.api_config_data[self.api.chat_model_provider]
            self.api.chat_ai_url = provider_config.get("base_url", "")
            self.api.chat_ai_key = provider_config.get("api_key", "")
            if "models" in provider_config and self.api.chat_model_name in provider_config["models"]:
                self.api.chat_ai_model = provider_config["models"][self.api.chat_model_name].get("model", "")
                logging.info(f"API配置加载完成: {self.api.chat_ai_url}, {self.api.chat_ai_key}, {self.api.chat_ai_model}")
            else:
                logging.error(f"模型 {self.api.chat_model_name} 未在API配置中找到")
        else:
            logging.error(f"模型提供者 {self.api.chat_model_provider} 未在API配置中找到")

        try:
            self.api.image_model_provider, self.api.image_model_name = self.image_recognition.ai_model.split("/", 1)
        except ValueError:
            logging.error(f"模型配置格式错误: {self.image_recognition.ai_model}，应为'provider/model'格式")
            self.api.image_model_provider = "openrouter_proxy"
            self.api.image_model_name = "qwen2.5-vl"


        if self.api.image_model_provider in self.api_config_data:
            provider_config = self.api_config_data[self.api.image_model_provider]
            self.api.image_ai_url = provider_config.get("base_url", "")
            self.api.image_ai_key = provider_config.get("api_key", "")
            if "models" in provider_config and self.api.image_model_name in provider_config["models"]:
                self.api.image_ai_model = provider_config["models"][self.api.image_model_name].get("model", "")
                logging.info(f"API配置加载完成: {self.api.image_ai_url}, {self.api.image_ai_key}, {self.api.image_ai_model}")
            else:
                logging.error(f"模型 {self.api.image_model_name} 未在API配置中找到")
        else:
            logging.error(f"模型提供者 {self.api.image_model_provider} 未在API配置中找到")

    def Prompt_Configuration_Handling(self):
        """处理提示词相关的配置"""

        # 加载聊天AI角色提示词文件
        self.prompt.chat_character_prompt = load_text_file(self.prompt.chat_character_prompt_file)
        # 加载聊天AI输出格式提示词文件
        self.prompt.chat_output_format_prompt = load_text_file(self.prompt.chat_output_format_prompt_file)
        # 整合角色提示词和输出格式提示词
        self.prompt.chat_full_prompt = self.prompt.chat_output_format_prompt + "\n" + self.prompt.chat_character_prompt
        # 加载图像识别提示词文件
        self.prompt.image_recognition_prompt = load_text_file(self.prompt.image_recognition_prompt_file)


    def path_configuration_handling(self):
        """处理路径相关的配置"""

        def convert_path_and_check(path: Path|str, base_path: Path|None = None) -> Path:
            """转换路径并检查其存在"""
            converted_path = convert_to_absolute_path(path, base_path)
            if not converted_path.exists():
                logging.error(f"路径不存在: {converted_path}")
            return converted_path
        
        self.prompt_dir = convert_path_and_check(self.prompt_dir)

        self.prompt.prompt_dir = self.prompt_dir
        self.prompt.chat_character_prompt_file = convert_path_and_check(self.main_ai.main_character_prompt, self.prompt_dir)
        self.prompt.chat_output_format_prompt_file = convert_path_and_check(self.main_ai.main_output_prompt, self.prompt_dir)
        self.prompt.image_recognition_prompt_file = convert_path_and_check(self.image_recognition.prompt_path, self.prompt_dir)

        self.image_recognition.cache_db_path = convert_path_and_check(self.image_recognition.cache_db_path)
        self.message_handling.chat_record_db_path = convert_path_and_check(self.message_handling.chat_record_db_path)


class ConfigGroupManager:
    """
    管理和切换多个配置组的管理器。
    - list_groups(): 列出可用配置组名称
    - switch_group(name, persist=False): 切换活动配置组（可选持久化为默认）
    - reload_active(): 重载当前活动配置组
    - get_active_group_name(): 获取当前活动配置组名
    - get_active_config(): 获取当前活动 ConfigGroup
    """
    def __init__(self, initial_group_name: str = DEFAULT_CONFIG_GROUP_NAME):
        self._lock = threading.RLock()
        self._cache: Dict[str, ConfigGroup] = {}
        self._active_group_name: str | None = None
        # 初始化时切换到默认组（不会立即持久化）
        self.switch_group(initial_group_name, persist=False)

    def list_groups(self) -> List[str]:
        """列出可用配置组（来自配置组文件的顶层键）。"""
        data = load_toml_config(CONFIG_GROUP_FILE_PATH)
        # 只收集 value 为 dict 的顶层键，视为一个组
        return [name for name, val in data.items() if isinstance(val, dict)]

    def get_active_group_name(self) -> str:
        with self._lock:
            if not self._active_group_name:
                raise RuntimeError("尚未初始化活动配置组")
            return self._active_group_name

    def get_active_config(self) -> "ConfigGroup":
        with self._lock:
            if not self._active_group_name:
                raise RuntimeError("尚未初始化活动配置组")
            return self._cache[self._active_group_name]

    def load_group(self, name: str) -> "ConfigGroup":
        """从文件加载指定配置组，并放入缓存。"""
        cfg = ConfigGroup.from_config_group_name(name)
        self._cache[name] = cfg
        return cfg

    def reload_active(self) -> "ConfigGroup":
        """重载当前活动配置组（重新读取配置文件、API配置、提示词与路径校验）。"""
        with self._lock:
            if not self._active_group_name:
                raise RuntimeError("尚未初始化活动配置组")
            name = self._active_group_name
            cfg = ConfigGroup.from_config_group_name(name)
            self._cache[name] = cfg
            self._update_global(cfg)
            logging.info(f"已重载配置组: {name}")
            return cfg

    def switch_group(self, name: str, persist: bool = False) -> "ConfigGroup":
        """
        切换活动配置组。
        - persist=True 则写回根配置文件 default_config_group，并更新内存中的默认值。
        """
        with self._lock:
            available = self.list_groups()
            if name not in available:
                raise ValueError(f"配置组 '{name}' 不存在，可用: {available}")

            cfg = self.load_group(name)
            self._active_group_name = name
            self._update_global(cfg)

            if persist:
                self._persist_default_group(name)

            logging.info(f"已切换配置组为: {name}")
            return cfg

    def _persist_default_group(self, name: str):
        """将默认配置组写回根配置文件，并更新内存中的 DEFAULT_CONFIG_GROUP_NAME/ROOT_CONFIG_DATA。"""
        try:
            data = load_toml_config(ROOT_CONFIG_PATH)
        except FileNotFoundError:
            logging.error(f"无法持久化默认配置组，根配置文件未找到: {ROOT_CONFIG_PATH}")
            return

        if data.get("default_config_group") != name:
            data["default_config_group"] = name
            with open(ROOT_CONFIG_PATH, "w", encoding="utf-8") as f:
                toml.dump(data, f)

            # 同步内存的根配置数据
            global ROOT_CONFIG_DATA, DEFAULT_CONFIG_GROUP_NAME
            ROOT_CONFIG_DATA = data
            DEFAULT_CONFIG_GROUP_NAME = name
            logging.info(f"已持久化默认配置组到根配置文件: {name}")

    def _update_global(self, cfg: "ConfigGroup"):
        """保持与既有全局变量兼容。"""
        global config_group_data
        config_group_data = cfg


# 加载默认配置组
config_group_manager = ConfigGroupManager(initial_group_name=DEFAULT_CONFIG_GROUP_NAME)
config_group_data = config_group_manager.get_active_config()


# 添加全局函数来获取当前配置组
def get_config_group() -> ConfigGroup:
    """获取当前配置组实例"""
    global config_group_data
    return config_group_data



if NONE_BOT_ENV:
    from nonebot.adapters.onebot.v11 import MessageEvent, GroupMessageEvent, PrivateMessageEvent
    from nonebot import on_command
    from nonebot.params import CommandArg
    from nonebot.adapters import Message
    from Permissions import admin, owner

    reload_config = on_command("reload_config", aliases={"重载配置", "重载配置文件"},
                               priority=5, block=True, permission=owner | admin)

    @reload_config.handle()
    async def _(
        event: MessageEvent,
        args: Message = CommandArg()
    ):
        text = args.extract_plain_text().strip()
        try:
            if text:
                # 带参数：切换配置组并持久化为默认
                config_group_manager.switch_group(text, persist=True)
                await reload_config.finish(f"已切换配置组为: {config_group_manager.get_active_group_name()}")
            else:
                # 无参数：重载当前配置组
                config_group_manager.reload_active()
                await reload_config.finish(f"已重载当前配置组: {config_group_manager.get_active_group_name()}")
        except Exception as e:
            logging.exception("重载/切换配置组失败")
            await reload_config.finish(f"操作失败: {e}")
elif __name__ == "__main__":
    print(config_group_data)
