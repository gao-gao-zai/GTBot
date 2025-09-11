import copy
import json
from pathlib import Path, WindowsPath
import tomllib
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Union, get_type_hints
import inspect
import toml
from pydantic import BaseModel, Field
import threading
import re
import os


try:
    from nonebot import get_driver
    get_driver()
    del get_driver
    from nonebot import logger as logging
    NONE_BOT_ENV = True
except (ValueError, ImportError):
    import logging
    # Simple fallback logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    NONE_BOT_ENV = False



import sys


dir_path = Path(__file__).parent
sys.path.append(str(dir_path))


ROOT_CONFIG_PATH = dir_path / "config.toml"
"""根配置文件路径"""

def load_toml_config(config_path: Path|str):
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到 {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)

def load_text_file(file_path: Path|str):
    if isinstance(file_path, str):
        file_path = Path(file_path)
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


# 全局变量将被移动到ConfigGroupManager类中








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
  
    def _convert_to_basic_type(self, value: Any) -> Any:
        """递归将值转换为Python基本类型"""
        if value is None:
            return None
        # 注意这里 isinstance 检查的是 DotDict，而不是 ConfigGroup
        elif isinstance(value, DotDict):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._convert_to_basic_type(item) for item in value]
        # 关键：处理 Path 对象
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, (int, float, str, bool)):
            return value
        # 处理其他可序列化或有 to_dict 方法的对象
        elif hasattr(value, 'to_dict') and callable(value.to_dict):
            return value.to_dict()
        else:
            try:
                # 最后的尝试，使用json序列化技巧
                return json.loads(json.dumps(value, default=str))
            except (TypeError, ValueError):
                # 如果失败，则返回其字符串表示形式
                return str(value)

    def to_dict(self):
        """转换回普通字典"""
        result = {}
        for key, value in self.items():
            # 使用辅助函数来处理所有值
            result[key] = self._convert_to_basic_type(value)
        return result
  
    def _list_to_dict(self, lst):
        """将列表中的DotDict转换回字典"""
        result = []
        for item in lst:
            # 这里也应该使用统一的转换方法
            result.append(self._convert_to_basic_type(item))
        return result




# 默认配置数据将在ConfigGroupManager初始化时加载



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
    # 最多允许失败的次数，超过该次数后图像将被跳过
    max_recognition_failures: int
    # 图像失败计数重置的时间（以秒为单位）
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
    """实际模型ID"""
    chat_model_is_reasoning: bool
    image_model_provider: str
    image_model_name: str
    image_ai_url: str
    image_ai_key: str
    image_ai_model: str
    image_model_is_reasoning: bool
    embedding_model_provider: str
    embedding_model_name: str
    embedding_ai_url: str
    embedding_ai_key: str
    embedding_ai_model: str
    reranker_model_provider: str
    reranker_model_name: str
    reranker_ai_url: str
    reranker_ai_key: str
    reranker_ai_model: str
    knowledge_base_management_ai_model_provider: str
    knowledge_base_management_ai_model_name: str
    knowledge_base_management_ai_url: str
    knowledge_base_management_ai_key: str
    knowledge_base_management_ai_model: str

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

    knowledge_base_management_prompt_file: Path
    knowledge_base_management_prompt: str

class _RetrievalAugmentedGenerationConfig:
    enable: bool
    chroma_service_url: str
    tenant: str
    embedding_model: str
    reranker_model: str
    enable_record_chat_history: bool
    enable_chat_history_retrieval: bool
    enable_knowledge_base: bool
    knowledge_base_management_ai: str
    knowledge_base_management_prompt_path: Path



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

    # regex for expression parsing
    _EXPRESSION_REGEX = re.compile(r"<{\[\$(.*?)\]}>", re.DOTALL)
    _TYPE_PREFIX_REGEX = re.compile(r"^<{\[(str|int|float|bool|list|dict|None)\$\]}>(.*)", re.DOTALL)


    def __init__(
        self, 
        config_group_name: str,
        config_group_file_path: Path,
        api_config_file_path: Path,
        default_config_group_file_path: Path,
        default_config_data: DotDict,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config_group_name = config_group_name
        self.config_group_file_path = config_group_file_path
        self.api_config_file_path = api_config_file_path
        self.default_config_group_file_path = default_config_group_file_path
        self.default_config_data = default_config_data
        self.api = _APIConfig()
        self.prompt = _PromptConfig()

            
        self.check_configuration_item_integrity()
        
        # --- NEW: Process dynamic expressions before further handling ---
        self._process_all_dynamic_values()

        self.path_configuration_handling()
        self.API_Configuration_Handling()
        self.Prompt_Configuration_Handling()



    def check_configuration_item_integrity(self):
        """检查配置项的完整性，使用默认配置文件填充缺失项"""
        self._merge_defaults(self, self.default_config_data, prefix="")
    
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
    
    # --- NEW: Methods for Dynamic Expression Evaluation ---

    def _validate_file_path(self, path: str) -> Path:
        """验证文件路径是否在安全范围内且符合大小限制"""
        # 转换为绝对路径
        file_path = convert_to_absolute_path(path, base_path=dir_path)
        
        # 确保路径在配置目录下
        try:
            # 尝试获取相对路径，如果失败说明不在dir_path下
            _ = file_path.relative_to(dir_path)
        except ValueError:
            raise ValueError(f"非法路径: {path}, 只能访问配置目录下的文件")
        
        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件未找到: {path}")
        
        # 检查文件大小 (1MB限制)
        file_size = file_path.stat().st_size
        if file_size > 1024 * 1024:  # 1MB
            raise ValueError(f"文件过大: {path} ({file_size}字节), 限制为1MB")
        
        return file_path

    def get_file_content(self, path: str) -> str:
        """获取文件中的文本内容（UTF-8编码）"""
        file_path = self._validate_file_path(path)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_toml_content(self, path: str) -> dict:
        """获取TOML文件内容"""
        file_path = self._validate_file_path(path)
        with open(file_path, 'rb') as f:
            return tomllib.load(f)

    def get_json_content(self, path: str) -> dict:
        """获取JSON文件内容"""
        file_path = self._validate_file_path(path)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_yaml_content(self, path: str) -> dict:
        """获取YAML文件内容"""
        file_path = self._validate_file_path(path)
        try:
            import yaml
        except ImportError:
            raise ImportError("需要安装PyYAML库来解析YAML文件: pip install pyyaml")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_safe_eval_context(self, config_snapshot) -> Dict[str, Any]:
        """Creates a secure context for expression evaluation using a config snapshot."""
        # 创建文件访问命名空间
        class FileAccessor:
            def __init__(self, config_instance):
                self.config = config_instance
                
            def get_file_content(self, path: str) -> str:
                return self.config.get_file_content(path)
                
            def get_toml_content(self, path: str) -> dict:
                return self.config.get_toml_content(path)
                
            def get_json_content(self, path: str) -> dict:
                return self.config.get_json_content(path)
                
            def get_yaml_content(self, path: str) -> dict:
                return self.config.get_yaml_content(path)
        
        file_accessor = FileAccessor(self)
        
        context = {
            'config': config_snapshot,  # 指向不可变快照
            'env': os.environ,
            'file': file_accessor,      # 安全的文件访问接口
            # 保留原始的快捷方式以保持向后兼容
            'get_file_content': lambda path: self.get_file_content(path),
            'get_toml_content': lambda path: self.get_toml_content(path),
            'get_json_content': lambda path: self.get_json_content(path),
            'get_yaml_content': lambda path: self.get_yaml_content(path)
        }
        return context

    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> Any:
        """Executes eval() in a restricted environment."""
        safe_builtins = {
            "str": str, "int": int, "float": float, "bool": bool, "list": list,
            "dict": dict, "tuple": tuple, "set": set, "len": len, "abs": abs,
            "round": round, "min": min, "max": max, "sum": sum, "any": any,
            "all": all, "None": None, "True": True, "False": False,
        }
        try:
            return eval(expression, {"__builtins__": safe_builtins}, context)
        except Exception as e:
            logging.error(f"表达式求值失败 '{expression}': {e}")
            return f"EVAL_ERROR: {e}"

    def _evaluate_string(self, value: str, context: Dict[str, Any]) -> Any:
        """
        Recursively evaluates expressions in a string and handles type casting.
        """
        history = set()
        while match := self._EXPRESSION_REGEX.search(value):
            expression = match.group(1)
            
            # Circular dependency check
            if expression in history:
                logging.error(f"在解析字符串时检测到循环依赖: {expression}")
                break
            history.add(expression)

            result = self._safe_eval(expression, context)

            # Convert result to string for substitution, unless it's None
            if result is None:
                result_str = ""
            elif isinstance(result, (int, float, str, bool, list, tuple, dict, set)):
                 result_str = str(result)
            else:
                 logging.warning(f"表达式 '{expression}' 返回了不支持的类型 {type(result)}，将被视为空字符串。")
                 result_str = ""

            value = value[:match.start()] + result_str + value[match.end():]
        
        # Type casting after all expressions are resolved
        if type_match := self._TYPE_PREFIX_REGEX.match(value):
            type_str = type_match.group(1)
            content = type_match.group(2)
            try:
                if type_str == "int":
                    return int(content)
                elif type_str == "float":
                    return float(content)
                elif type_str == "bool":
                    return content.lower() in ('true', '1', 't', 'y', 'yes')
                elif type_str == "list" or type_str == "dict":
                    return json.loads(content)
                elif type_str == "None":
                    return None
                else: # str
                    return content
            except (ValueError, json.JSONDecodeError) as e:
                logging.error(f"无法将 '{content}' 转换为类型 '{type_str}': {e}")
                return content # Return original string on failure
        
        return value

    def _process_dynamic_values(self, item: Any, context: Dict[str, Any]) -> Any:
        """Recursively traverses the config to process dynamic values."""
        if isinstance(item, dict):
            return DotDict({k: self._process_dynamic_values(v, context) for k, v in item.items()})
        elif isinstance(item, list):
            return [self._process_dynamic_values(v, context) for v in item]
        elif isinstance(item, str):
            return self._evaluate_string(item, context)
        return item

    def _process_all_dynamic_values(self):
        """Initializes the dynamic value processing for the entire configuration."""
        logging.info("开始解析配置中的动态表达式...")
        # 创建当前配置的深拷贝作为不可变快照
        import copy
        config_snapshot = copy.deepcopy(self)
        
        # 使用快照创建安全上下文
        context = self._create_safe_eval_context(config_snapshot)
        processed_config = self._process_dynamic_values(self, context)
        self.clear()
        self.update(processed_config)
        logging.info("动态表达式解析完成。")

    # --- End of New Methods ---

    def _get_model_info(self, model_name: str, api_config_data: dict) -> dict:
        """获取模型信息"""
        try:
            provider, model_name = model_name.split("/", 1)
        except ValueError:
            logging.error(f"模型配置格式错误: {model_name}，应为'provider/model'格式")
            raise ValueError(f"模型配置格式错误: {model_name}，应为'provider/model'格式")

        key: str = ""
        url: str = ""

        if provider in api_config_data:
            provider_config = api_config_data[provider]
            key = provider_config.get("api_key", "")
            url = provider_config.get("base_url", "")
            if "models" in provider_config and model_name in provider_config["models"]:
                model = provider_config["models"][model_name]
                if "model" not in model:
                    raise ValueError(f"模型 {model_name} 未在API配置中找到")
                model_id = model["model"]
                return {
                    "provider": provider,
                    "model_name": model_name,
                    "model_id": model_id,
                    "url": url,
                    "key": key,
                }
            else:
                logging.error(f"模型 {model_name} 未在API配置中找到")
                raise ValueError(f"模型 {model_name} 未在API配置中找到")
        else:
            logging.error(f"模型提供者 {provider} 未在API配置中找到")
            raise ValueError(f"模型提供者 {provider} 未在API配置中找到")
    

    def API_Configuration_Handling(self, api_config_file_path: Path|None = None):
        """处理API相关的配置"""
        if api_config_file_path is None:
            api_config_file_path = self.api_config_file_path

        self.api.chat_model_is_reasoning = self.main_ai.is_reasoning_model
        self.api.image_model_is_reasoning = self.image_recognition.is_reasoning_model

        
        # 读取api配置
        if not api_config_file_path.exists():
            raise FileNotFoundError(f"API配置文件未找到: {api_config_file_path}")
        with open(api_config_file_path, "r", encoding="utf-8") as f:
            self.api_config_data: dict = json.load(f)
        

        
        # try:
        #     self.api.chat_model_provider, self.api.chat_model_name = self.main_ai.main_model.split("/", 1)
        # except ValueError:
        #     logging.error(f"模型配置格式错误: {self.main_ai.main_model}，应为'provider/model'格式")
        #     raise ValueError(f"模型配置格式错误: {self.main_ai.main_model}，应为'provider/model'格式")
        #     # self.api.chat_model_provider = "openrouter_proxy"
        #     # self.api.chat_model_name = "deepseek-v3"


        # if self.api.chat_model_provider in self.api_config_data:
        #     provider_config = self.api_config_data[self.api.chat_model_provider]
        #     self.api.chat_ai_url = provider_config.get("base_url", "")
        #     self.api.chat_ai_key = provider_config.get("api_key", "")
        #     if "models" in provider_config and self.api.chat_model_name in provider_config["models"]:
        #         self.api.chat_ai_model = provider_config["models"][self.api.chat_model_name].get("model", "")
        #         logging.info(f"API配置加载完成: {self.api.chat_ai_url}, {self.api.chat_ai_key}, {self.api.chat_ai_model}")
        #     else:
        #         logging.error(f"模型 {self.api.chat_model_name} 未在API配置中找到")
        # else:
        #     logging.error(f"模型提供者 {self.api.chat_model_provider} 未在API配置中找到")


        chat_model_info = self._get_model_info(self.main_ai.main_model, self.api_config_data)
        self.api.chat_model_provider = chat_model_info["provider"]
        self.api.chat_model_name = chat_model_info["model_name"]
        self.api.chat_ai_model = chat_model_info["model_id"]
        self.api.chat_ai_url = chat_model_info["url"]
        self.api.chat_ai_key = chat_model_info["key"]
        
        image_model_info = self._get_model_info(self.image_recognition.ai_model, self.api_config_data)
        self.api.image_model_provider = image_model_info["provider"]
        self.api.image_model_name = image_model_info["model_name"]
        self.api.image_ai_model = image_model_info["model_id"]
        self.api.image_ai_url = image_model_info["url"]
        self.api.image_ai_key = image_model_info["key"]

        embedding_model_info = self._get_model_info(self.Retrieval_Augmented_Generation.embedding_model, self.api_config_data)
        self.api.embedding_model_provider = embedding_model_info["provider"]
        self.api.embedding_model_name = embedding_model_info["model_name"]
        self.api.embedding_ai_model = embedding_model_info["model_id"]
        self.api.embedding_ai_url = embedding_model_info["url"]
        self.api.embedding_ai_key = embedding_model_info["key"]

        reranker_model_info = self._get_model_info(self.Retrieval_Augmented_Generation.reranker_model, self.api_config_data)
        self.api.reranker_model_provider = reranker_model_info["provider"]
        self.api.reranker_model_name = reranker_model_info["model_name"]
        self.api.reranker_ai_model = reranker_model_info["model_id"]
        self.api.reranker_ai_url = reranker_model_info["url"]
        self.api.reranker_ai_key = reranker_model_info["key"]

        knowledge_base_management_ai_model_info = self._get_model_info(self.Retrieval_Augmented_Generation.knowledge_base_management_ai, self.api_config_data)
        self.api.knowledge_base_management_ai_model_provider = knowledge_base_management_ai_model_info["provider"]
        self.api.knowledge_base_management_ai_model_name = knowledge_base_management_ai_model_info["model_name"]
        self.api.knowledge_base_management_ai_model = knowledge_base_management_ai_model_info["model_id"]
        self.api.knowledge_base_management_ai_url = knowledge_base_management_ai_model_info["url"]
        self.api.knowledge_base_management_ai_key = knowledge_base_management_ai_model_info["key"]
        
        

    def Prompt_Configuration_Handling(self):
        """处理提示词相关的配置"""

        import copy
        config_snapshot = copy.deepcopy(self)
    
        context = self._create_safe_eval_context(config_snapshot)

        # 加载并解析聊天AI角色提示词文件
        raw_char_prompt = load_text_file(self.prompt.chat_character_prompt_file)
        self.prompt.chat_character_prompt = self._evaluate_string(raw_char_prompt, context)
        
        # 加载并解析聊天AI输出格式提示词文件
        raw_output_prompt = load_text_file(self.prompt.chat_output_format_prompt_file)
        self.prompt.chat_output_format_prompt = self._evaluate_string(raw_output_prompt, context)

        # 整合角色提示词和输出格式提示词
        self.prompt.chat_full_prompt = self.prompt.chat_output_format_prompt + "\n" + self.prompt.chat_character_prompt
        
        # 加载并解析图像识别提示词文件
        raw_img_prompt = load_text_file(self.prompt.image_recognition_prompt_file)
        self.prompt.image_recognition_prompt = self._evaluate_string(raw_img_prompt, context)

        # 加载并解析知识库管理提示词文件
        raw_knowledge_base_management_prompt = load_text_file(self.prompt.knowledge_base_management_prompt_file)
        self.prompt.knowledge_base_management_prompt = self._evaluate_string(raw_knowledge_base_management_prompt, context)


    def path_configuration_handling(self):
        """处理路径相关的配置"""

        def convert_path_and_check(path: Path|str, base_path: Path|None = None) -> Path:
            """转换路径并检查其存在"""
            # If path is already a Path object, no conversion needed
            if isinstance(path, Path):
                converted_path = path
            else:
                converted_path = convert_to_absolute_path(path, base_path)

            if not converted_path.exists():
                # Allow parent directory creation for db files
                if '.db' in converted_path.suffixes:
                    logging.warning(f"路径不存在: {converted_path}，将尝试创建其父目录。")
                    converted_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    logging.error(f"路径不存在: {converted_path}")
            return converted_path
        
        self.prompt_dir = convert_path_and_check(self.prompt_dir)

        self.prompt.prompt_dir = self.prompt_dir
        self.prompt.chat_character_prompt_file = convert_path_and_check(self.main_ai.main_character_prompt, self.prompt_dir)
        self.prompt.chat_output_format_prompt_file = convert_path_and_check(self.main_ai.main_output_prompt, self.prompt_dir)
        self.prompt.image_recognition_prompt_file = convert_path_and_check(self.image_recognition.prompt_path, self.prompt_dir)
        self.prompt.knowledge_base_management_prompt_file = convert_path_and_check(self.Retrieval_Augmented_Generation.knowledge_base_management_prompt_path, self.prompt_dir)


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
    def __init__(self, initial_group_name: str|None = None):
        self._lock = threading.RLock()
        self._cache: Dict[str, ConfigGroup] = {}
        self._active_group_name: str | None = None
        
        # 初始化配置路径和数据
        self.root_config_path = dir_path / "config.toml"
        self.root_config_data = load_toml_config(self.root_config_path)
        
        self.default_config_group_name = self.root_config_data["default_config_group"]
        
        
        config_group_file_path: Path = Path(self.root_config_data["config_group_path"])
        self.config_group_file_path = convert_to_absolute_path(config_group_file_path)
        logging.info(f"配置组文件路径: {self.config_group_file_path}")
        
        api_config_file_path = Path(self.root_config_data["api_config_path"])
        self.api_config_file_path = convert_to_absolute_path(api_config_file_path)
        logging.info(f"API配置文件路径: {self.api_config_file_path}")
        
        default_config_group_file_path = Path(self.root_config_data["default_config_group_path"])
        self.default_config_group_file_path = convert_to_absolute_path(default_config_group_file_path)
        logging.info(f"默认配置组文件路径: {self.default_config_group_file_path}")
        
        # 加载默认配置数据
        try:
            # 直接加载整个文件作为默认配置
            self.default_config_data = DotDict(load_toml_config(self.default_config_group_file_path))
            if not self.default_config_data: # 检查字典是否为空
                raise RuntimeError("默认配置文件为空")
        except FileNotFoundError:
            raise RuntimeError(f"默认配置文件未找到: {self.default_config_group_file_path}") from None
        
        # 初始化时切换到默认组（不会立即持久化）
        if initial_group_name is None:
            initial_group_name = self.default_config_group_name
        self.switch_group(initial_group_name, persist=False)

    @staticmethod
    def convert_to_absolute_path(path: Path|str, base_path: Path|None = None) -> Path:
        """将相对路径转换为绝对路径"""
        if isinstance(path, str):
            path = Path(path)
        if path.is_absolute():
            return path
        if base_path is None:
            base_path = dir_path
        return (base_path / path).resolve()

    def list_groups(self) -> List[str]:
        """列出可用配置组（来自配置组文件的顶层键）。"""
        data = load_toml_config(self.config_group_file_path)
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
        cfg = self._create_config_group(name)
        self._cache[name] = cfg
        return cfg

    def reload_active(self) -> "ConfigGroup":
        """重载当前活动配置组（重新读取配置文件、API配置、提示词与路径校验）。"""
        with self._lock:
            if not self._active_group_name:
                raise RuntimeError("尚未初始化活动配置组")
            name = self._active_group_name
            # Re-load default config data in case it changed
            self.default_config_data = DotDict(load_toml_config(self.default_config_group_file_path).get("default", {}))
            cfg = self._create_config_group(name)
            self._cache[name] = cfg
            self._update_global(cfg)
            logging.info(f"已重载配置组: {name}")
            return cfg

    def switch_group(self, name: str|None, persist: bool = False) -> "ConfigGroup":
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

    def _create_config_group(self, name: str) -> "ConfigGroup":
        """创建ConfigGroup实例"""
        # 加载配置组文件
        group_config_data = load_toml_config(self.config_group_file_path)
        
        # 获取指定配置组的数据
        if name in group_config_data:
            config_data = group_config_data[name]
        else:
            logging.warning(f"配置组 '{name}' 未找到，使用空配置")
            config_data = {}
        
        # 创建Config实例
        config = ConfigGroup(
            config_group_name=name,
            config_group_file_path=self.config_group_file_path,
            api_config_file_path=self.api_config_file_path,
            default_config_group_file_path=self.default_config_group_file_path,
            default_config_data=self.default_config_data,
            **config_data
        )
        
        return config

    def _persist_default_group(self, name: str):
        """将默认配置组写回根配置文件，并更新内存中的默认值。"""
        try:
            # Using tomllib to read and toml to write
            with open(self.root_config_path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            logging.error(f"无法持久化默认配置组，根配置文件未找到: {self.root_config_path}")
            return

        if data.get("default_config_group") != name:
            data["default_config_group"] = name
            with open(self.root_config_path, "w", encoding="utf-8") as f:
                toml.dump(data, f)

            # 同步内存的配置数据
            self.root_config_data = data
            self.default_config_group_name = name
            logging.info(f"已持久化默认配置组到根配置文件: {name}")

    def _update_global(self, cfg: "ConfigGroup"):
        """保持与既有全局变量兼容。"""
        global config_group_data
        config_group_data = cfg


# 加载默认配置组
config_group_manager = ConfigGroupManager()
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
    
    # Assuming Permissions.py exists and defines these
    # from Permissions import admin, owner
    def owner(): return lambda: True # Placeholder
    def admin(): return lambda: True # Placeholder

    reload_config = on_command("reload_config", aliases={"重载配置", "重载配置文件"},
                               priority=5, block=True)

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
                await reload_config.finish(f"已切换并持久化配置组为: {config_group_manager.get_active_group_name()}")
            else:
                # 无参数：重载当前配置组
                config_group_manager.reload_active()
                await reload_config.finish(f"已重载当前配置组: {config_group_manager.get_active_group_name()}")
        except Exception as e:
            logging.exception("重载/切换配置组失败")
            await reload_config.finish(f"操作失败: {e}")
elif __name__ == "__main__":
    # Example usage for standalone execution
    print("--- Active Configuration ---")
    print(json.dumps(config_group_data.to_dict(), indent=2, ensure_ascii=False))
    print("\n--- Full Character Prompt ---")
    print(config_group_data.prompt.chat_full_prompt)
