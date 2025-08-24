import json
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Union, get_type_hints
import toml
from pydantic import BaseModel, Field
import threading
import os
import re

# --- Regex for Expression Parsing ---
# Matches expressions that take up the entire string, e.g., "<{[$ config.some_value ]}>"
_expr_re = re.compile(r'^\<\{\[\$(.*)\]\}\>$', re.DOTALL)
# Matches expressions embedded within a larger string, e.g., "Hello, <{[$ config.user.name ]}>!"
_inline_expr_re = re.compile(r'\<\{\[\$(.*?)\\]\}>', re.DOTALL)

try:
    from nonebot import get_driver
    get_driver()
    del get_driver
    from nonebot import logger as logging
    NONE_BOT_ENV = True
except Exception:
    import logging
    NONE_BOT_ENV = False

import sys

dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

# --- Global Configuration Paths and Data ---
ROOT_CONFIG_PATH = dir_path / "config.toml"
"""根配置文件路径"""

def load_toml_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到 {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return toml.load(f)

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

# --- DotDict Class ---
class DotDict(dict):
    """支持点号访问的字典类"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_nested()
    
    def _convert_nested(self):
        """递归转换嵌套的字典为DotDict"""
        for key, value in list(self.items()):
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

# --- Load Default Configuration ---
try:
    DEFAULT_CONFIG_DATA = DotDict(load_toml_config(DEFAULT_CONFIG_GROUP_FILE_PATH).get("default", {}))
    if DEFAULT_CONFIG_DATA == {}:
        raise RuntimeError("默认配置文件为空")
except FileNotFoundError:
    raise RuntimeError(f"默认配置文件未找到: {DEFAULT_CONFIG_GROUP_FILE_PATH}") from None

# --- Configuration Classes ---
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
    max_recognition_failures: int
    failure_reset_window_seconds: int

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

class _RetrievalAugmentedGenerationConfig:
    enable: bool
    chroma_service_url: str
    ollama_service_url: str
    embedding_model: str
    reranker_model: str



class ConfigGroup(DotDict):
    """配置类"""
    main_ai: _MainAiConfig
    image_recognition: _ImageRecognitionConfig
    emoji_response: _EmojiResponseConfig
    message_handling: _MessageHandlingConfig
    Retrieval_Augmented_Generation: _RetrievalAugmentedGenerationConfig
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
        # 初始化一个空的求值上下文，将在后面填充
        self.eval_context = {}

        self.check_configuration_item_integrity()
        self._evaluate_expressions()
        self.path_configuration_handling()
        self.API_Configuration_Handling()
        self.Prompt_Configuration_Handling()
    
    def _evaluate_expressions(self):
        """递归地解析配置字典中标记的 Python 表达式。"""
        # 创建并存储安全的执行上下文，供后续使用
        self.eval_context = self._create_safe_context()
        self._recursive_eval(self, self.eval_context)

    def _recursive_eval(self, obj, context: dict):
        """
        递归地遍历配置对象并评估其中的表达式。
        :param obj: 要处理的对象（可能是字典、列表或字符串）
        :param context: 表达式评估的上下文
        """
        if isinstance(obj, dict):
            # 如果是字典，递归处理每个值
            for key, value in obj.items():
                obj[key] = self._recursive_eval(value, context)
        elif isinstance(obj, list):
            # 如果是列表，递归处理每个元素
            for i, item in enumerate(obj):
                obj[i] = self._recursive_eval(item, context)
        elif isinstance(obj, str):
            # 如果是字符串，检查是否包含表达式
            match = _expr_re.match(obj)
            if match:
                # 完整表达式，直接求值
                expr = match.group(1).strip()
                try:
                    return eval(expr, context)
                except Exception as e:
                    logging.error(f"评估表达式 '{expr}' 失败: {e}")
                    return obj
            else:
                # 内联表达式，使用替换方法
                return self._evaluate_string_with_context(obj, context)
        
        return obj

    def _evaluate_string_with_context(self, text: str, context: dict) -> str:
        """
        查找并评估字符串中所有 <{[$...]>} 格式的内联表达式。
        """
        def replacer(match):
            expr = match.group(1).strip()
            try:
                # 在安全上下文中评估表达式
                result = eval(expr, context)
                return str(result)
            except Exception as e:
                logging.error(f"评估表达式 '{expr}' 失败: {e}")
                # 如果出错，返回原始表达式字符串，便于调试
                return match.group(0)

        return _inline_expr_re.sub(replacer, text)

    def _create_safe_context(self, source_file_path: Path|None = None) -> dict:
        """
        创建一个严格控制且安全的上下文（沙箱），供 eval() 使用。
        """
        # 1. 创建白名单的内置函数集合（已扩展更多无害方法）
        # 仅包含不会进行I/O操作、不会访问系统资源、不会执行代码的纯函数
        safe_builtins = {
            # 基础类型转换
            'abs': abs, 'bool': bool, 'bytes': bytes, 'complex': complex,
            'dict': dict, 'float': float, 'frozenset': frozenset, 'int': int,
            'list': list, 'set': set, 'str': str, 'tuple': tuple,
            
            # 集合操作
            'all': all, 'any': any, 'enumerate': enumerate, 'filter': filter,
            'map': map, 'max': max, 'min': min, 'reversed': reversed,
            'sorted': sorted, 'zip': zip,
            
            # 数学运算
            'divmod': divmod, 'pow': pow, 'round': round, 'sum': sum,
            
            # 对象检查
            'callable': callable, 'dir': dir, 'hasattr': hasattr, 'getattr': getattr,
            'isinstance': isinstance, 'issubclass': issubclass, 'len': len, 'type': type,
            
            # 字符串处理
            'ascii': ascii, 'bin': bin, 'chr': chr, 'format': format,
            'hex': hex, 'oct': oct, 'ord': ord, 'repr': repr,
            
            # 常量
            'True': True, 'False': False, 'None': None,
            'Ellipsis': Ellipsis, 'NotImplemented': NotImplemented,
        }

        # 2. 创建安全的文件访问函数（增强版）
        def get_file_content(file_path: str) -> str:
            """安全获取文件内容，相对路径基于源文件的目录"""
            # 确定基准路径：如果有源文件路径，使用其目录；否则使用项目根目录
            base_path = source_file_path.parent if source_file_path else dir_path
            full_path = convert_to_absolute_path(file_path, base_path)
            if not full_path.resolve().is_relative_to(dir_path):
                raise PermissionError(f"路径遍历攻击检测：禁止访问 '{full_path}'")
            if not full_path.is_file():
                raise FileNotFoundError(f"文件不存在：{full_path}")
            if full_path.stat().st_size > 1024 * 1024:  # 1MB limit
                raise ValueError(f"文件过大（>{1}MB）：{full_path}")
            return full_path.read_text(encoding="utf-8", errors="replace")

        # 3. 创建安全的数据处理工具函数
        def safe_json_load(json_str: str) -> Any:
            """安全解析JSON字符串"""
            import json
            try:
                return json.loads(json_str, parse_float=float, parse_int=int)
            except Exception as e:
                raise ValueError(f"JSON解析失败: {str(e)}") from None

        # 4. 环境变量白名单（更严格的筛选）
        safe_env = {
            key: value for key, value in os.environ.items()
            if key in {"HOME", "LANG", "LC_ALL", "USER", "TERM"}
        }

        # 5. NoneBot配置安全处理
        nonebot_config = {}
        if NONE_BOT_ENV:
            try:
                from nonebot import get_driver
                nb_config = get_driver().config.dict()
                safe_config = {}
                for k, v in nb_config.items():
                    if not callable(v) and not hasattr(v, '__module__'):
                        safe_config[k] = v
                nonebot_config = safe_config
            except Exception:
                pass

        # 6. 添加实用工具函数
        def safe_get(data: dict, key: str, default: Any = None) -> Any:
            """安全获取字典值，支持多级路径（用.分隔）"""
            current = data
            for part in key.split('.'):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current

        def safe_merge(*dicts: dict) -> dict:
            """安全合并多个字典（浅拷贝）"""
            result = {}
            for d in dicts:
                if isinstance(d, dict):
                    result.update(d)
            return result

        # 7. 汇总最终的安全上下文
        context = {
            "__builtins__": safe_builtins,
            "config": self.to_dict(),
            "env": safe_env,
            "nb_config": nonebot_config,
            "get_file_content": get_file_content,
            "safe_json_load": safe_json_load,
            "safe_get": safe_get,
            "safe_merge": safe_merge,
            "PI": 3.141592653589793,
            "E": 2.718281828459045,
            "is_number": lambda x: isinstance(x, (int, float)),
            "is_string": lambda x: isinstance(x, str),
            "to_upper": str.upper,
            "to_lower": str.lower,
        }
        
        return context

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
                current[key] = default_value
                logging.warning(f"配置项 {full_path} 缺失，已自动添加默认值: {default_value}")
            else:
                if isinstance(default_value, dict) and isinstance(current[key], dict):
                    self._merge_defaults(
                        current[key], 
                        default_value, 
                        prefix=full_path
                    )
                elif isinstance(default_value, list) and isinstance(current[key], list):
                    self._merge_list_defaults(current[key], default_value, full_path)
    
    def _merge_list_defaults(self, current_list, default_list, prefix):
        """处理列表中的默认值（仅处理字典元素）"""
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
        group_config_data = load_toml_config(CONFIG_GROUP_FILE_PATH)
        
        if config_group_name in group_config_data:
            config_data = group_config_data[config_group_name]
        else:
            logging.warning(f"配置组 '{config_group_name}' 未找到，使用空配置")
            config_data = {}
        
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
                raise KeyError(f"模型 '{self.api.chat_model_name}' 未在API配置文件中为 provider '{self.api.chat_model_provider}' 定义")
        else:
            raise KeyError(f"模型提供者 '{self.api.chat_model_provider}' 未在API配置文件中找到")

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
        """处理提示词相关的配置，并解析其中的表达式"""
        # 读取角色设定提示词文件，并解析其中的表达式
        char_prompt_file = Path(self.prompt.chat_character_prompt_file)
        raw_char_prompt = char_prompt_file.read_text(encoding="utf-8")
        char_context = self._create_safe_context(char_prompt_file)
        self.prompt.chat_character_prompt = self._evaluate_string_with_context(raw_char_prompt, char_context)

        # 读取输出格式提示词文件，并解析其中的表达式
        output_format_file = Path(self.prompt.chat_output_format_prompt_file)
        raw_output_format_prompt = output_format_file.read_text(encoding="utf-8")
        output_format_context = self._create_safe_context(output_format_file)
        self.prompt.chat_output_format_prompt = self._evaluate_string_with_context(raw_output_format_prompt, output_format_context)
        
        # 组合成完整的聊天提示词
        self.prompt.chat_full_prompt = self.prompt.chat_output_format_prompt + "\n" + self.prompt.chat_character_prompt
        
        # 读取图像识别提示词文件，并解析其中的表达式
        image_recognition_file = Path(self.prompt.image_recognition_prompt_file)
        raw_image_recognition_prompt = image_recognition_file.read_text(encoding="utf-8")
        image_recognition_context = self._create_safe_context(image_recognition_file)
        self.prompt.image_recognition_prompt = self._evaluate_string_with_context(raw_image_recognition_prompt, image_recognition_context)


    def path_configuration_handling(self):
        """处理路径相关的配置"""
        def convert_path_and_check(path: Path|str, base_path: Path|None = None) -> Path:
            """转换路径并检查其存在"""
            converted_path = convert_to_absolute_path(path, base_path)
            if not converted_path.exists():
                raise FileNotFoundError(f"必要的配置文件或路径不存在: {converted_path}")
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
    """
    def __init__(self, initial_group_name: str = DEFAULT_CONFIG_GROUP_NAME):
        self._lock = threading.RLock()
        self._cache: Dict[str, ConfigGroup] = {}
        self._active_group_name: str | None = None
        self.switch_group(initial_group_name, persist=False)

    def list_groups(self) -> List[str]:
        """列出可用配置组（来自配置组文件的顶层键）。"""
        data = load_toml_config(CONFIG_GROUP_FILE_PATH)
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
                config_group_manager.switch_group(text, persist=True)
                await reload_config.finish(f"已切换配置组为: {config_group_manager.get_active_group_name()}")
            else:
                config_group_manager.reload_active()
                await reload_config.finish(f"已重载当前配置组: {config_group_manager.get_active_group_name()}")
        except Exception as e:
            logging.exception("重载/切换配置组失败")
            await reload_config.finish(f"操作失败: {e}")
elif __name__ == "__main__":
    print(config_group_data)