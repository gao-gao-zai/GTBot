import os
import json
import aiofiles
import httpx
import requests
from typing import List, Dict, Any, Optional

# 配置日志记录器
try:
    from nonebot import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
class GPTSoVITSAPIError(Exception):
    """自定义异常，用于表示API请求失败。"""
    pass

class ConfigurationError(Exception):
    """自定义异常，用于表示配置文件错误。"""
    pass

class ManagerStateError(Exception):
    """自定义异常，用于表示管理器状态不一致。"""
    pass

class GPTSoVITSManager:
    """
    一个用于安全、可靠地管理GPT-SoVITS语音合成服务的Python库。
    
    本管理器通过配置文件驱动，提供清晰的API接口，确保每次TTS调用
    都满足服务端的参数要求，并为多说话人场景设计，防止声音特征混杂（"窜声"）。
    """

    def __init__(self, config_path: str, host: str = "127.0.0.1", port: int = 9880, timeout: int = 30):
        """
        初始化管理器。

        :param config_path: 配置文件的路径。
        :param host: GPT-SoVITS WebAPI 服务的主机地址。
        :param port: GPT-SoVITS WebAPI 服务的端口。
        :param timeout: 网络请求的超时时间（秒）。
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        
        self._config = self._load_and_validate_config(config_path)
        
        # 内部状态跟踪
        self.current_speaker: Optional[Dict[str, Any]] = None
        self.current_gpt_path: Optional[str] = None
        self.current_sovits_path: Optional[str] = None

        logger.info("管理器初始化成功，正在将服务状态与基础模型同步...")
        self._initialize_server_state()
        logger.info("服务状态同步完成。")

    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """加载并验证配置文件。"""
        if not os.path.exists(config_path):
            raise ConfigurationError(f"配置文件不存在: {config_path}")

        config_dir = os.path.dirname(os.path.abspath(config_path))
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # --- 结构验证 ---
        if "base_models" not in config or "sovits_path" not in config["base_models"] or "gpt_path" not in config["base_models"]:
            raise ConfigurationError("配置文件缺少 'base_models' 或其必要的 'sovits_path', 'gpt_path' 键。")
        
        if "speakers" not in config or not isinstance(config["speakers"], list):
            raise ConfigurationError("配置文件缺少 'speakers' 列表。")

        # --- 路径解析和文件存在性验证 ---
        def resolve_path(p):
            if not p or not isinstance(p, str): return None
            abs_path = os.path.join(config_dir, p) if not os.path.isabs(p) else p
            if not os.path.exists(abs_path):
                raise ConfigurationError(f"配置文件中指定的文件不存在: {abs_path} (原始路径: {p})")
            return abs_path

        config["base_models"]["sovits_path"] = resolve_path(config["base_models"]["sovits_path"])
        config["base_models"]["gpt_path"] = resolve_path(config["base_models"]["gpt_path"])

        for speaker in config["speakers"]:
            required_keys = ["name", "is_trained", "default_ref_audio", "default_prompt_text", "default_prompt_lang", "styles"]
            if not all(key in speaker for key in required_keys):
                raise ConfigurationError(f"说话人 '{speaker.get('name', '未知')}' 配置不完整，缺少键。")
            
            if not speaker["styles"]:
                 raise ConfigurationError(f"说话人 '{speaker['name']}' 必须至少有一个风格配置。")

            if speaker["is_trained"]:
                if "sovits_path" not in speaker or "gpt_path" not in speaker:
                    raise ConfigurationError(f"训练说话人 '{speaker['name']}' 必须提供 'sovits_path' 和 'gpt_path'。")
                speaker["sovits_path"] = resolve_path(speaker["sovits_path"])
                speaker["gpt_path"] = resolve_path(speaker["gpt_path"])
            
            speaker["default_ref_audio"] = resolve_path(speaker["default_ref_audio"])
            
            for style in speaker["styles"]:
                if not all(key in style for key in ["name", "ref_audio", "prompt_text", "prompt_lang"]):
                    raise ConfigurationError(f"说话人 '{speaker['name']}' 的风格 '{style.get('name', '未知')}' 配置不完整。")
                style["ref_audio"] = resolve_path(style["ref_audio"])

        return config

    def _initialize_server_state(self):
        """使用基础模型初始化服务器状态。"""
        base_gpt = self._config["base_models"]["gpt_path"]
        base_sovits = self._config["base_models"]["sovits_path"]
        self.set_gpt_weights(base_gpt)
        self.set_sovits_weights(base_sovits)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """统一的网络请求函数。"""
        url = self.base_url + endpoint
        
        # 如果是GET请求，记录完整的URL
        if method.upper() == 'GET':
            # 处理查询参数
            params = kwargs.get('params', {})
            if params:
                # 构建带参数的完整URL
                import urllib.parse
                query_string = urllib.parse.urlencode(params, doseq=True)
                full_url = f"{url}?{query_string}" if query_string else url
                logger.info(f"GET Request URL: {full_url}")
            else:
                logger.info(f"GET Request URL: {url}")
        
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            if response.status_code != 200:
                error_info = response.json() if response.headers.get('Content-Type') == 'application/json' else response.text
                raise GPTSoVITSAPIError(f"API请求失败: {response.status_code} - {error_info}")
            return response
        except requests.exceptions.RequestException as e:
            raise GPTSoVITSAPIError(f"网络请求错误: {e}")

    def _find_speaker(self, speaker_name: str) -> Dict[str, Any]:
        """根据名称查找说话人配置。"""
        for speaker in self._config["speakers"]:
            if speaker["name"] == speaker_name:
                return speaker
        raise ValueError(f"未找到名为 '{speaker_name}' 的说话人。")

    # --- 服务控制 ---
    
    def set_gpt_weights(self, weights_path: str) -> None:
        """切换GPT模型。"""
        logger.info(f"正在切换GPT模型到: {os.path.basename(weights_path)}")
        self._make_request('GET', '/set_gpt_weights', params={'weights_path': weights_path})
        self.current_gpt_path = weights_path
        logger.info("GPT模型切换成功。")

    def set_sovits_weights(self, weights_path: str) -> None:
        """切换Sovits模型。"""
        logger.info(f"正在切换SoVITS模型到: {os.path.basename(weights_path)}")
        self._make_request('GET', '/set_sovits_weights', params={'weights_path': weights_path})
        self.current_sovits_path = weights_path
        logger.info("SoVITS模型切换成功。")

    def restart_service(self) -> None:
        """重启GPT-SoVITS服务。"""
        logger.info("正在发送重启命令...")
        self._make_request('GET', '/control', params={'command': 'restart'})
        logger.info("重启命令已发送。")

    def shutdown_service(self) -> None:
        """关闭GPT-SoVITS服务。"""
        logger.info("正在发送关闭命令...")
        self._make_request('GET', '/control', params={'command': 'exit'})
        logger.info("关闭命令已发送。")
        
    # --- 说话人与风格管理 ---

    def list_speakers(self) -> List[str]:
        """列出所有可用的说话人名称。"""
        return [s["name"] for s in self._config["speakers"]]

    def list_speaker_styles(self, speaker_name: str) -> List[str]:
        """列出指定说话人的所有风格名称。"""
        speaker = self._find_speaker(speaker_name)
        return [style["name"] for style in speaker["styles"]]

    def list_current_speaker_styles(self) -> List[str]:
        """列出当前活动说话人的所有风格名称。"""
        if not self.current_speaker:
            raise ManagerStateError("当前没有活动的说话人，无法列出风格。")
        return self.list_speaker_styles(self.current_speaker["name"])
    
    def switch_speaker(self, speaker_name: str) -> None:
        """
        安全地切换当前活动说话人。
        此操作会按正确顺序（先GPT后Sovits）切换模型，并在失败时自动回滚。
        """
        if self.current_speaker and self.current_speaker["name"] == speaker_name:
            logger.info(f"说话人 '{speaker_name}' 已经是当前活动说话人。")
            return
            
        logger.info(f"--- 开始切换说话人到: {speaker_name} ---")
        target_speaker = self._find_speaker(speaker_name)

        original_gpt = self.current_gpt_path
        original_sovits = self.current_sovits_path

        if target_speaker["is_trained"]:
            target_gpt = target_speaker["gpt_path"]
            target_sovits = target_speaker["sovits_path"]
        else:
            target_gpt = self._config["base_models"]["gpt_path"]
            target_sovits = self._config["base_models"]["sovits_path"]
            
        try:
            # 1. 确保先切换GPT模型
            if self.current_gpt_path != target_gpt:
                self.set_gpt_weights(target_gpt)
            else:
                logger.info("目标GPT模型已加载，无需切换。")
            
            # 2. 再切换Sovits模型
            if self.current_sovits_path != target_sovits:
                self.set_sovits_weights(target_sovits)
            else:
                logger.info("目标SoVITS模型已加载，无需切换。")
            
            # 3. 更新内部状态
            self.current_speaker = target_speaker
            logger.info(f"--- 成功切换到说话人: {speaker_name} ---")

        except Exception as e:
            logger.error(f"!! 切换到 '{speaker_name}' 失败: {e}。正在尝试回滚到之前的状态...")
            try:
                if self.current_gpt_path != original_gpt:
                    self.set_gpt_weights(original_gpt)
                if self.current_sovits_path != original_sovits:
                    self.set_sovits_weights(original_sovits)
                logger.warning("!! 状态回滚成功。")
            except Exception as rollback_e:
                raise ManagerStateError(
                    f"!! 关键错误：切换失败后，回滚也失败了！"
                    f"管理器状态可能与服务器状态不一致。请考虑重启服务。回滚错误: {rollback_e}"
                )
            raise e # 重新抛出原始异常

    # --- 语音合成 ---

    def tts(self, text: str, text_lang: str, ref_audio_path: str, prompt_text: str, prompt_lang: str, **kwargs) -> bytes:
        """
        基础TTS合成接口，强制要求所有核心参数。

        :param text: 要合成的文本。
        :param text_lang: 合成文本的语言。
        :param ref_audio_path: 参考音频的路径。
        :param prompt_text: 参考音频的提示文本。
        :param prompt_lang: 提示文本的语言。
        :param kwargs: 其他所有WebAPI支持的可选参数 (e.g., top_k, top_p, temperature, etc.)。
        :return: 合成后的WAV音频流 (bytes)。
        """
        payload = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang.lower(),
            **kwargs
        }
        logger.info(f"发起TTS请求: text='{text[:20]}...', ref_audio='{os.path.basename(ref_audio_path)}'")
        response = self._make_request('POST', '/tts', json=payload)
        return response.content
        
    def tts_with_current_speaker(self, text: str, text_lang: str, style_name: Optional[str] = None, **kwargs) -> bytes:
        """
        使用当前活动的说话人配置进行语音合成。

        :param text: 要合成的文本。
        :param text_lang: 合成文本的语言。
        :param style_name: （可选）要使用的风格名称。如果为None，则使用当前说话人的默认风格。
        :param kwargs: 其他所有WebAPI支持的可选参数。
        :return: 合成后的WAV音频流 (bytes)。
        """
        if not self.current_speaker:
            raise ManagerStateError("没有活动的说话人。请先调用 switch_speaker()。")

        if style_name:
            style = next((s for s in self.current_speaker["styles"] if s["name"] == style_name), None)
            if not style:
                raise ValueError(f"当前说话人 '{self.current_speaker['name']}' 不存在名为 '{style_name}' 的风格。")
            ref_audio = style["ref_audio"]
            prompt_text = style["prompt_text"]
            prompt_lang = style["prompt_lang"]
            logger.info(f"使用当前说话人 '{self.current_speaker['name']}' 的风格 '{style_name}'。")
        else:
            ref_audio = self.current_speaker["default_ref_audio"]
            prompt_text = self.current_speaker["default_prompt_text"]
            prompt_lang = self.current_speaker["default_prompt_lang"]
            logger.info(f"使用当前说话人 '{self.current_speaker['name']}' 的默认风格。")

        return self.tts(text, text_lang, ref_audio, prompt_text, prompt_lang, **kwargs)

    def tts_with_style(self, speaker_name: str, style_name: str, text: str, text_lang: str, **kwargs) -> bytes:
        """
        使用指定说话人的指定风格进行语音合成。
        如果当前说话人不是目标说话人，会自动进行切换。

        :param speaker_name: 目标说话人的名称。
        :param style_name: 目标风格的名称。
        :param text: 要合成的文本。
        :param text_lang: 合成文本的语言。
        :param kwargs: 其他所有WebAPI支持的可选参数。
        :return: 合成后的WAV音频流 (bytes)。
        """
        if not self.current_speaker or self.current_speaker["name"] != speaker_name:
            self.switch_speaker(speaker_name)
        
        return self.tts_with_current_speaker(text, text_lang, style_name, **kwargs)

    def tts_to_file(self, text: str, text_lang: str, ref_audio_path: str, prompt_text: str, prompt_lang: str, 
                   output_path: str, **kwargs) -> str:
        """
        基础TTS合成接口，强制要求所有核心参数，并将结果保存为文件。

        :param text: 要合成的文本。
        :param text_lang: 合成文本的语言。
        :param ref_audio_path: 参考音频的路径。
        :param prompt_text: 参考音频的提示文本。
        :param prompt_lang: 提示文本的语言。
        :param output_path: 输出文件的路径。
        :param kwargs: 其他所有WebAPI支持的可选参数 (e.g., top_k, top_p, temperature
        :return: 输出文件的路径。
        """
        audio_data = self.tts(text, text_lang, ref_audio_path, prompt_text, prompt_lang, **kwargs)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"音频已保存到: {output_path}")
        return output_path

    def tts_with_current_speaker_to_file(self, text: str, text_lang: str, output_path: str, 
                                       style_name: Optional[str] = None, **kwargs) -> str:
        """
        使用当前活动的说话人配置进行语音合成，并将结果保存为文件。

        :param text: 要合成的文本。
        :param text_lang: 合成文本的语言。
        :param output_path: 输出文件的路径。
        :param style_name: （可选）要使用的风格名称。如果为None，则使用当前说话人的默认风格。
        :param kwargs: 其他所有WebAPI支持的可选参数。
        :return: 输出文件的路径。
        """
        if not self.current_speaker:
            raise ManagerStateError("没有活动的说话人。请先调用 switch_speaker()。")

        if style_name:
            style = next((s for s in self.current_speaker["styles"] if s["name"] == style_name), None)
            if not style:
                raise ValueError(f"当前说话人 '{self.current_speaker['name']}' 不存在名为 '{style_name}' 的风格。")
            ref_audio = style["ref_audio"]
            prompt_text = style["prompt_text"]
            prompt_lang = style["prompt_lang"]
            logger.info(f"使用当前说话人 '{self.current_speaker['name']}' 的风格 '{style_name}'。")
        else:
            ref_audio = self.current_speaker["default_ref_audio"]
            prompt_text = self.current_speaker["default_prompt_text"]
            prompt_lang = self.current_speaker["default_prompt_lang"]
            logger.info(f"使用当前说话人 '{self.current_speaker['name']}' 的默认风格。")

        return self.tts_to_file(text, text_lang, ref_audio, prompt_text, prompt_lang, output_path, **kwargs)

    def tts_with_style_to_file(self, speaker_name: str, style_name: str, text: str, text_lang: str, 
                             output_path: str, **kwargs) -> str:
        """
        使用指定说话人的指定风格进行语音合成，并将结果保存为文件。
        如果当前说话人不是目标说话人，会自动进行切换。

        :param speaker_name: 目标说话人的名称。
        :param style_name: 目标风格的名称。
        :param text: 要合成的文本。
        :param text_lang: 合成文本的语言。
        :param output_path: 输出文件的路径。
        :param kwargs: 其他所有WebAPI支持的可选参数。
        :return: 输出文件的路径。
        """
        if not self.current_speaker or self.current_speaker["name"] != speaker_name:
            self.switch_speaker(speaker_name)

        return self.tts_with_current_speaker_to_file(text, text_lang, output_path, style_name, **kwargs)


class AsyncGPTSoVITSManager:
    """
    一个用于安全、可靠地管理GPT-SoVITS语音合成服务的异步Python库。
    
    本管理器通过配置文件驱动，提供清晰的API接口，确保每次TTS调用
    都满足服务端的参数要求，并为多说话人场景设计，防止声音特征混杂（"窜声"）。
    所有网络和文件操作均为异步。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9880, timeout: int = 30):
        """
        私有构造函数。请使用异步类方法 `create` 来实例化。
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        
        self._config: Optional[Dict[str, Any]] = None
        self.current_speaker: Optional[Dict[str, Any]] = None
        self.current_gpt_path: Optional[str] = None
        self.current_sovits_path: Optional[str] = None

    @classmethod
    async def create(cls, config_path: str, host: str = "127.0.0.1", port: int = 9880, timeout: int = 30):
        """
        异步创建并初始化管理器实例。

        :param config_path: 配置文件的路径。
        :param host: GPT-SoVITS WebAPI 服务的主机地址。
        :param port: GPT-SoVITS WebAPI 服务的端口。
        :param timeout: 网络请求的超时时间（秒）。
        :return: 一个初始化完成的 AsyncGPTSoVITSManager 实例。
        """
        instance = cls(host, port, timeout)
        instance._config = await instance._load_and_validate_config(config_path)
        
        logger.info("管理器初始化成功，正在将服务状态与基础模型同步...")
        await instance._initialize_server_state()
        logger.info("服务状态同步完成。")
        return instance

    async def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """异步加载并验证配置文件。"""
        if not os.path.exists(config_path):
            raise ConfigurationError(f"配置文件不存在: {config_path}")

        config_dir = os.path.dirname(os.path.abspath(config_path))
        
        async with aiofiles.open(config_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            config = json.loads(content)

        # --- 结构验证 (与同步版本相同) ---
        if "base_models" not in config or "sovits_path" not in config["base_models"] or "gpt_path" not in config["base_models"]:
            raise ConfigurationError("配置文件缺少 'base_models' 或其必要的 'sovits_path', 'gpt_path' 键。")
        
        if "speakers" not in config or not isinstance(config["speakers"], list):
            raise ConfigurationError("配置文件缺少 'speakers' 列表。")

        # --- 路径解析和文件存在性验证 (与同步版本相同) ---
        def resolve_path(p):
            if not p or not isinstance(p, str): return None
            abs_path = os.path.join(config_dir, p) if not os.path.isabs(p) else p
            if not os.path.exists(abs_path):
                raise ConfigurationError(f"配置文件中指定的文件不存在: {abs_path} (原始路径: {p})")
            return abs_path

        config["base_models"]["sovits_path"] = resolve_path(config["base_models"]["sovits_path"])
        config["base_models"]["gpt_path"] = resolve_path(config["base_models"]["gpt_path"])

        for speaker in config["speakers"]:
            required_keys = ["name", "is_trained", "default_ref_audio", "default_prompt_text", "default_prompt_lang", "styles"]
            if not all(key in speaker for key in required_keys):
                raise ConfigurationError(f"说话人 '{speaker.get('name', '未知')}' 配置不完整，缺少键。")
            
            if not speaker["styles"]:
                raise ConfigurationError(f"说话人 '{speaker['name']}' 必须至少有一个风格配置。")

            if speaker["is_trained"]:
                if "sovits_path" not in speaker or "gpt_path" not in speaker:
                    raise ConfigurationError(f"训练说话人 '{speaker['name']}' 必须提供 'sovits_path' 和 'gpt_path'。")
                speaker["sovits_path"] = resolve_path(speaker["sovits_path"])
                speaker["gpt_path"] = resolve_path(speaker["gpt_path"])
            
            speaker["default_ref_audio"] = resolve_path(speaker["default_ref_audio"])
            
            for style in speaker["styles"]:
                if not all(key in style for key in ["name", "ref_audio", "prompt_text", "prompt_lang"]):
                    raise ConfigurationError(f"说话人 '{speaker['name']}' 的风格 '{style.get('name', '未知')}' 配置不完整。")
                style["ref_audio"] = resolve_path(style["ref_audio"])
        
        return config

    async def _initialize_server_state(self):
        """使用基础模型异步初始化服务器状态。"""
        if not self._config:
            raise ManagerStateError("配置未加载，无法初始化服务器。")
        base_gpt = self._config["base_models"]["gpt_path"]
        base_sovits = self._config["base_models"]["sovits_path"]
        await self.set_gpt_weights(base_gpt)
        await self.set_sovits_weights(base_sovits)
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """统一的异步网络请求函数。"""
        log_message = f"{method.upper()} Request to {self.base_url}{endpoint}"
        if 'params' in kwargs:
            log_message += f" with params {kwargs['params']}"
        if 'json' in kwargs:
             log_message += f" with json data" # Avoid logging large data
        logger.info(log_message)
        
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            if response.status_code != 200:
                try:
                    error_info = response.json()
                except json.JSONDecodeError:
                    error_info = response.text
                raise GPTSoVITSAPIError(f"API请求失败: {response.status_code} - {error_info}")
            return response
        except httpx.RequestError as e:
            raise GPTSoVITSAPIError(f"网络请求错误: {e}")

    def _find_speaker(self, speaker_name: str) -> Dict[str, Any]:
        """根据名称查找说话人配置。"""
        if not self._config:
            raise ManagerStateError("配置未加载，无法查找说话人。")
        for speaker in self._config["speakers"]:
            if speaker["name"] == speaker_name:
                return speaker
        raise ValueError(f"未找到名为 '{speaker_name}' 的说话人。")

    async def close(self):
        """关闭底层的httpx客户端。"""
        await self.client.aclose()
        logger.info("HTTP客户端已关闭。")

    # --- 服务控制 ---
    
    async def set_gpt_weights(self, weights_path: str) -> None:
        """异步切换GPT模型。"""
        logger.info(f"正在切换GPT模型到: {os.path.basename(weights_path)}")
        await self._make_request('GET', '/set_gpt_weights', params={'weights_path': weights_path})
        self.current_gpt_path = weights_path
        logger.info("GPT模型切换成功。")

    async def set_sovits_weights(self, weights_path: str) -> None:
        """异步切换Sovits模型。"""
        logger.info(f"正在切换SoVITS模型到: {os.path.basename(weights_path)}")
        await self._make_request('GET', '/set_sovits_weights', params={'weights_path': weights_path})
        self.current_sovits_path = weights_path
        logger.info("SoVITS模型切换成功。")

    async def restart_service(self) -> None:
        """异步重启GPT-SoVITS服务。"""
        logger.info("正在发送重启命令...")
        await self._make_request('GET', '/control', params={'command': 'restart'})
        logger.info("重启命令已发送。")

    async def shutdown_service(self) -> None:
        """异步关闭GPT-SoVITS服务。"""
        logger.info("正在发送关闭命令...")
        await self._make_request('GET', '/control', params={'command': 'exit'})
        logger.info("关闭命令已发送。")
        
    # --- 说话人与风格管理 ---

    def list_speakers(self) -> List[str]:
        """列出所有可用的说话人名称。"""
        if not self._config:
            raise ManagerStateError("配置未加载。")
        return [s["name"] for s in self._config["speakers"]]

    def list_speaker_styles(self, speaker_name: str) -> List[str]:
        """列出指定说话人的所有风格名称。"""
        speaker = self._find_speaker(speaker_name)
        return [style["name"] for style in speaker["styles"]]

    def list_current_speaker_styles(self) -> List[str]:
        """列出当前活动说话人的所有风格名称。"""
        if not self.current_speaker:
            raise ManagerStateError("当前没有活动的说话人，无法列出风格。")
        return self.list_speaker_styles(self.current_speaker["name"])
    
    async def switch_speaker(self, speaker_name: str) -> None:
        """
        安全地异步切换当前活动说话人。
        此操作会按正确顺序（先GPT后Sovits）切换模型，并在失败时自动回滚。
        """
        if self.current_speaker and self.current_speaker["name"] == speaker_name:
            logger.info(f"说话人 '{speaker_name}' 已经是当前活动说话人。")
            return
        
        if not self._config:
            raise ManagerStateError("配置未加载。")

        logger.info(f"--- 开始切换说话人到: {speaker_name} ---")
        target_speaker = self._find_speaker(speaker_name)

        original_gpt = self.current_gpt_path
        original_sovits = self.current_sovits_path

        if target_speaker["is_trained"]:
            target_gpt = target_speaker["gpt_path"]
            target_sovits = target_speaker["sovits_path"]
        else:
            target_gpt = self._config["base_models"]["gpt_path"]
            target_sovits = self._config["base_models"]["sovits_path"]
            
        try:
            if self.current_gpt_path != target_gpt:
                await self.set_gpt_weights(target_gpt)
            else:
                logger.info("目标GPT模型已加载，无需切换。")
            
            if self.current_sovits_path != target_sovits:
                await self.set_sovits_weights(target_sovits)
            else:
                logger.info("目标SoVITS模型已加载，无需切换。")
            
            self.current_speaker = target_speaker
            logger.info(f"--- 成功切换到说话人: {speaker_name} ---")

        except Exception as e:
            logger.error(f"!! 切换到 '{speaker_name}' 失败: {e}。正在尝试回滚到之前的状态...")
            try:
                if self.current_gpt_path != original_gpt and original_gpt is not None:
                    await self.set_gpt_weights(original_gpt)
                if self.current_sovits_path != original_sovits and original_sovits is not None:
                    await self.set_sovits_weights(original_sovits)
                logger.warning("!! 状态回滚成功。")
            except Exception as rollback_e:
                raise ManagerStateError(
                    f"!! 关键错误：切换失败后，回滚也失败了！"
                    f"管理器状态可能与服务器状态不一致。请考虑重启服务。回滚错误: {rollback_e}"
                )
            raise e

    # --- 语音合成 ---

    async def tts(self, text: str, text_lang: str, ref_audio_path: str, prompt_text: str, prompt_lang: str, **kwargs) -> bytes:
        """
        基础TTS异步合成接口，强制要求所有核心参数。
        """
        payload = {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang.lower(),
            **kwargs
        }
        logger.info(f"发起TTS请求: text='{text[:20]}...', ref_audio='{os.path.basename(ref_audio_path)}'")
        response = await self._make_request('POST', '/tts', json=payload)
        return response.content
        
    async def tts_with_current_speaker(self, text: str, text_lang: str, style_name: Optional[str] = None, **kwargs) -> bytes:
        """
        使用当前活动的说话人配置进行异步语音合成。
        """
        if not self.current_speaker:
            raise ManagerStateError("没有活动的说话人。请先调用 switch_speaker()。")

        if style_name:
            style = next((s for s in self.current_speaker["styles"] if s["name"] == style_name), None)
            if not style:
                raise ValueError(f"当前说话人 '{self.current_speaker['name']}' 不存在名为 '{style_name}' 的风格。")
            ref_audio = style["ref_audio"]
            prompt_text = style["prompt_text"]
            prompt_lang = style["prompt_lang"]
            logger.info(f"使用当前说话人 '{self.current_speaker['name']}' 的风格 '{style_name}'。")
        else:
            ref_audio = self.current_speaker["default_ref_audio"]
            prompt_text = self.current_speaker["default_prompt_text"]
            prompt_lang = self.current_speaker["default_prompt_lang"]
            logger.info(f"使用当前说话人 '{self.current_speaker['name']}' 的默认风格。")

        return await self.tts(text, text_lang, ref_audio, prompt_text, prompt_lang, **kwargs)

    async def tts_with_style(self, speaker_name: str, style_name: str, text: str, text_lang: str, **kwargs) -> bytes:
        """
        使用指定说话人的指定风格进行异步语音合成。
        如果当前说话人不是目标说话人，会自动进行切换。
        """
        if not self.current_speaker or self.current_speaker["name"] != speaker_name:
            await self.switch_speaker(speaker_name)
        
        return await self.tts_with_current_speaker(text, text_lang, style_name, **kwargs)

    async def tts_to_file(self, text: str, text_lang: str, ref_audio_path: str, prompt_text: str, prompt_lang: str, 
                      output_path: str, **kwargs) -> str:
        """
        基础TTS异步合成接口，并将结果异步保存为文件。
        """
        audio_data = await self.tts(text, text_lang, ref_audio_path, prompt_text, prompt_lang, **kwargs)
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        async with aiofiles.open(output_path, "wb") as f:
            await f.write(audio_data)
        
        logger.info(f"音频已保存到: {output_path}")
        return output_path

    async def tts_with_current_speaker_to_file(self, text: str, text_lang: str, output_path: str, 
                                           style_name: Optional[str] = None, **kwargs) -> str:
        """
        使用当前活动的说话人配置进行异步语音合成，并将结果异步保存为文件。
        """
        if not self.current_speaker:
            raise ManagerStateError("没有活动的说话人。请先调用 switch_speaker()。")

        if style_name:
            style = next((s for s in self.current_speaker["styles"] if s["name"] == style_name), None)
            if not style:
                raise ValueError(f"当前说话人 '{self.current_speaker['name']}' 不存在名为 '{style_name}' 的风格。")
            ref_audio = style["ref_audio"]
            prompt_text = style["prompt_text"]
            prompt_lang = style["prompt_lang"]
            logger.info(f"使用当前说话人 '{self.current_speaker['name']}' 的风格 '{style_name}'。")
        else:
            ref_audio = self.current_speaker["default_ref_audio"]
            prompt_text = self.current_speaker["default_prompt_text"]
            prompt_lang = self.current_speaker["default_prompt_lang"]
            logger.info(f"使用当前说话人 '{self.current_speaker['name']}' 的默认风格。")

        return await self.tts_to_file(text, text_lang, ref_audio, prompt_text, prompt_lang, output_path, **kwargs)

    async def tts_with_style_to_file(self, speaker_name: str, style_name: str, text: str, text_lang: str, 
                                 output_path: str, **kwargs) -> str:
        """
        使用指定说话人的指定风格进行异步语音合成，并将结果异步保存为文件。
        如果当前说话人不是目标说话人，会自动进行切换。
        """
        if not self.current_speaker or self.current_speaker["name"] != speaker_name:
            await self.switch_speaker(speaker_name)

        return await self.tts_with_current_speaker_to_file(text, text_lang, output_path, style_name, **kwargs)
