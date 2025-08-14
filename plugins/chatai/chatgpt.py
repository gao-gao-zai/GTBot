import json
import time
import base64
import os
import mimetypes
from typing import List, Dict, Optional, AsyncGenerator, Union, Any, Literal
import tiktoken
import copy
from rich import print
import asyncio
import aiohttp
from openai import AsyncOpenAI

# 定义 OneMessage 类封装消息属性
class OneMessage:
    def __init__(
        self,
        role: str,
        content: Union[str, List[Dict[str, Any]]],
        token_count: int = 0,
        lock: bool = False
    ):
        self.role = role
        self.content = content
        self.token_count = token_count
        self.lock = lock

    def to_api_format(self) -> Dict[str, Any]:
        """转换为 OpenAI API 需要的消息格式"""
        return {
            "role": self.role,
            "content": self.content
        }

    @property
    def text_content(self) -> str:
        """返回原始文本内容"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # 提取所有 text 类型的内容并拼接
            text_parts = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts)
        else:
            return ""

class StreamReturn:
    def __init__(self, is_main_text: bool, is_reasoning: bool, main_text: str = "", reasoning_text: str = ""):
        self.is_main_text: bool = is_main_text
        self.is_reasoning: bool = is_reasoning
        self.main_text: str = main_text
        self.reasoning_text: str = reasoning_text


class ChatGPT:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        chat_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        prompt: str = "",
        context: Optional[List[OneMessage]] = None,
        custom_params: Optional[Dict[str, Any]] = None,
        # 添加推理模型相关参数
        is_reasoning_model: bool = False,
        thought_format: Literal["openai", "string_token", "ollama", "auto", "none"] = "openai",
        start_of_thinking_mark: str = "<think>",
        end_of_thinking_mark: str = "</think>",
        # 新增：推理内容匹配相关参数
        reasoning_key: str = "reasoning",
        reasoning_match_mode: Literal["full", "contains", "contained_by", "path"] = "contains",
        reasoning_case_sensitive: bool = False,
    ):
        self.api_key: str = api_key
        self.base_url: str = base_url
        if chat_url == None:
            self.chat_url: str = f"{base_url}/chat/completions"
        else:
            self.chat_url: str = chat_url
        self.model: str = model
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.top_p: float = top_p
        self.frequency_penalty: float = frequency_penalty
        self.prompt: str = prompt
        self.context: List[OneMessage] = [copy.copy(msg) for msg in context] if context is not None else []
        self.session: Optional[aiohttp.ClientSession] = None
        self.custom_params: Dict[str, Any] = custom_params.copy() if custom_params is not None else {}
        
        # 推理模型相关属性
        self.is_reasoning_model: bool = is_reasoning_model
        self.thought_format: Literal["openai", "string_token", "ollama", "auto", "none"] = thought_format
        self.start_of_thinking_mark: str = start_of_thinking_mark
        self.end_of_thinking_mark: str = end_of_thinking_mark
        
        # 新增：推理内容匹配相关属性
        self.reasoning_key: str = reasoning_key
        self.reasoning_match_mode: Literal["full", "contains", "contained_by", "path"] = reasoning_match_mode
        self.reasoning_case_sensitive: bool = reasoning_case_sensitive

        self._cached_reasoning_path: Optional[str] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None

    def _find_reasoning_content(
        self,
        full_json_data: Dict[str, Any],
        key_ref: str,
        match_mode: str,
        case_sensitive: bool,
    ) -> Optional[str]:
        """
        根据指定的模式在API响应中查找并返回思考内容。
        Args:
            full_json_data: 完整的API响应JSON对象。
            key_ref: 用于匹配的参考键或路径。
            match_mode: 匹配模式 ('full', 'contains', 'contained_by', 'path')。
            case_sensitive: 是否区分大小写 (对'path'模式无效)。
        Returns:
            找到的思考内容字符串，如果字段存在但值为None则返回空字符串，如果未找到则返回None。
        """
        # 路径模式：直接按路径查找
        if match_mode == "path":
            try:
                keys = key_ref.split('.')
                current_value: Any = full_json_data
                for key in keys:
                    if isinstance(current_value, list) and key.isdigit():
                        current_value = current_value[int(key)]
                    elif isinstance(current_value, dict):
                        current_value = current_value[key]
                    else:
                        return None
                
                # 检查字段存在性（即使值为None）
                if current_value is not None:
                    # 如果字段存在但值为None，返回空字符串
                    return current_value if isinstance(current_value, str) else ""
                return None
            except (KeyError, IndexError, TypeError):
                return None

        # 其他模式：在 'choices[0].delta' 或 'choices[0].message' 中查找
        search_dict = {}
        try:
            choice = full_json_data.get("choices", [{}])[0]
            if "delta" in choice:
                search_dict = choice["delta"]
            elif "message" in choice:
                search_dict = choice["message"]
        except (IndexError, TypeError):
            return None

        if not isinstance(search_dict, dict):
            return None
        
        # 准备用于大小写不敏感比较的参考键
        ref = key_ref if case_sensitive else key_ref.lower()

        # 遍历查找字典
        for key, value in search_dict.items():
            current_key = key if case_sensitive else key.lower()

            # 检查是否匹配键
            key_matched = False
            if match_mode == "full" and current_key == ref:
                key_matched = True
            elif match_mode == "contains" and ref in current_key:
                key_matched = True
            elif match_mode == "contained_by" and current_key in ref:
                key_matched = True
                
            # 只要键匹配就返回（即使值为None）
            if key_matched:
                # 值存在但为None时返回空字符串
                if value is None:
                    return ""
                # 值存在且是字符串时返回
                if isinstance(value, str):
                    return value
                # 值存在但不是字符串时返回空字符串
                return ""
                
        return None

    async def get_context(self) -> List[OneMessage]:
        return [copy.copy(msg) for msg in self.context]

    async def update_custom_params(self, params_dict: Optional[dict] = None, **params):
        if params_dict and params:
            raise ValueError("不能同时使用 params_dict 和 **params 参数。")
        elif not params_dict and not params:
            raise ValueError("必须提供 params_dict 或 **params 参数。")
        if params_dict:
            self.custom_params.update(params_dict)
        if params:
            self.custom_params.update(params)

    async def add_dialogue(
        self, content: Union[str, list], role: str, lock: bool = False
    ):
        """添加对话内容，支持纯文本或多模态内容数组"""
        if role not in ["user", "assistant", "system"]:
            raise ValueError("角色必须是 'user', 'assistant' 或 'system' 之一。")

        # 如果是字符串，转换为标准内容格式
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        
        # 创建 OneMessage 实例并添加到上下文
        new_message = OneMessage(
            role=role,
            content=content,
            token_count=0,  # 实际使用时可计算token数量
            lock=lock
        )
        self.context.append(new_message)

    async def add_image(
        self,
        image_url: str,
        text: str = "",
        detail: str = "auto",
        role: str = "user",
        lock: bool = False,
    ):
        """添加图像URL到上下文，支持详细程度设置"""
        content = []

        if text:
            content.append({"type": "text", "text": text})

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": detail},  # auto, low, high
            }
        )

        await self.add_dialogue(content, role, lock)

    async def add_local_file(
        self,
        file_path: str,
        text: str = "",
        detail: str = "auto",
        role: str = "user",
        lock: bool = False,
    ):
        """
        添加本地文件到上下文
        支持图像文件（转换为Base64 data URL）
        对于非图像文件，会尝试读取为文本（仅支持文本文件）
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 获取文件MIME类型
        mime_type, _ = mimetypes.guess_type(file_path)
        content = []

        if text:
            content.append({"type": "text", "text": text})

        # 处理图像文件
        if mime_type and mime_type.startswith("image/"):
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                data_url = f"data:{mime_type};base64,{base64_image}"

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": detail},
                    }
                )
        else:
            # 处理文本文件
            try:
                with open(file_path, "r", encoding="utf-8") as text_file:
                    file_content = text_file.read()
                    content.append(
                        {
                            "type": "text",
                            "text": f"文件内容 ({os.path.basename(file_path)}):\n{file_content}",
                        }
                    )
            except UnicodeDecodeError:
                # 如果是二进制文件，无法读取为文本
                content.append(
                    {
                        "type": "text",
                        "text": f"无法读取二进制文件: {os.path.basename(file_path)}",
                    }
                )

        await self.add_dialogue(content, role, lock)
    
    def _build_payload(self, stream: bool = False) -> Dict[str, Any]:
        """构建请求负载，合并自定义参数并处理冲突"""
        # 基础负载
        payload = {
            "model": self.model,
            "messages": [msg.to_api_format() for msg in self.context],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
        }
        
        # 如果启用流式，添加stream参数
        if stream:
            payload["stream"] = True
        
        # 合并自定义参数（自定义参数优先）
        # 注意: 这可能会覆盖上面的任何参数，包括model, messages等
        payload.update(self.custom_params)
        
        return payload

    async def get_response(
        self,
        input_text: Union[str, list] = "",
        role: str = "user",
        return_raw_response: bool = False,
        no_input: bool = False,
        add_to_context: bool = True,
        override_params: Optional[Dict[str, Any]] = None,
        # 添加推理模型自定义属性参数
        is_reasoning_model: Optional[bool] = None,
        thought_format: Optional[Literal["openai", "string_token", "ollama", "auto", "none"]] = None,
        start_of_thinking_mark: Optional[str] = None,
        end_of_thinking_mark: Optional[str] = None,
        return_reasoning: bool = False,
        # 新增：推理内容匹配的临时覆盖参数
        reasoning_key: Optional[str] = None,
        reasoning_match_mode: Optional[Literal["full", "contains", "contained_by", "path"]] = None,
        reasoning_case_sensitive: Optional[bool] = None,
    ) -> Union[str, tuple[str, str], dict]:
        """
        获取模型响应，支持多模态输入和灵活的推理内容提取。

        Args:
            input_text (Union[str, list], optional): 输入文本，可以是字符串或列表。默认为 ""。
            role (str, optional): 对话角色。默认为 "user"。
            only_content (bool, optional): 是否只返回内容。默认为 True。
            no_input (bool, optional): 是否跳过输入检查。默认为 False。
            add_to_context (bool, optional): 是否将响应添加到上下文。默认为 True。
            override_params (Optional[Dict[str, Any]], optional): 覆盖的参数字典。默认为 None。
            is_reasoning_model (Optional[bool], optional): 是否为推理模型。默认为 None。
            thought_format (Optional[Literal["openai", "string_token", "ollama", "auto", "none"]], optional): 
                推理格式类型。默认为 None。
            start_of_thinking_mark (Optional[str], optional): 推理开始标记。默认为 None。
            end_of_thinking_mark (Optional[str], optional): 推理结束标记。默认为 None。
            return_reasoning (bool, optional): 是否返回推理内容。默认为 False。
            reasoning_key (Optional[str], optional): 推理内容键名。默认为 None。
            reasoning_match_mode (Optional[Literal["full", "contains", "contained_by", "path"]], optional): 
                推理内容匹配模式。默认为 None。
            reasoning_case_sensitive (Optional[bool], optional): 推理匹配是否区分大小写。默认为 None。

        Returns:
            Union[str, tuple[str, str], dict]: 返回响应内容，可能是字符串、元组或字典。

        Raises:
            ValueError: 当输入为空时抛出。
            RuntimeError: 当HTTP请求或JSON解析出错时抛出。
            Exception: 其他异常情况。
        """
        # 确定本次调用的有效参数
        local_is_reasoning_model = is_reasoning_model if is_reasoning_model is not None else self.is_reasoning_model
        local_thought_format = thought_format if thought_format is not None else self.thought_format
        local_start_mark = start_of_thinking_mark if start_of_thinking_mark is not None else self.start_of_thinking_mark
        local_end_mark = end_of_thinking_mark if end_of_thinking_mark is not None else self.end_of_thinking_mark
        local_reasoning_key = reasoning_key if reasoning_key is not None else self.reasoning_key
        local_reasoning_match_mode = reasoning_match_mode if reasoning_match_mode is not None else self.reasoning_match_mode
        local_reasoning_case_sensitive = reasoning_case_sensitive if reasoning_case_sensitive is not None else self.reasoning_case_sensitive
        
        if not no_input:
            if input_text or isinstance(input_text, list):
                await self.add_dialogue(input_text, role)
            else:
                raise ValueError("用户输入为空!")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }

        # 构建基础负载
        payload = self._build_payload(stream=False)
        
        # 如果有本次调用覆盖参数，合并到负载中（优先级最高）
        if override_params:
            payload.update(override_params)

        try:
            # 如果没有外部管理的session，创建临时session
            if not self.session:
                async with aiohttp.ClientSession() as temp_session:
                    async with temp_session.post(
                        self.chat_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        response_json = await response.json()
            else:
                async with self.session.post(
                    self.chat_url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()

            # 处理响应
            message = response_json["choices"][0].get("message", {})
            main_text = ""
            reasoning_text = ""
            
            if local_is_reasoning_model and local_thought_format != "none":
                # 处理OpenAI格式的推理模型（使用新的匹配逻辑）
                if local_thought_format in ["openai", "auto"]:
                    found_reasoning = self._find_reasoning_content(
                        full_json_data=response_json,
                        key_ref=local_reasoning_key,
                        match_mode=local_reasoning_match_mode,
                        case_sensitive=local_reasoning_case_sensitive,
                    )
                    if found_reasoning:
                        reasoning_text = found_reasoning
                    main_text = message.get("content", "")
                # 处理字符串标记格式
                else: # string_token or ollama
                    content_str = message.get("content", "")
                    start_idx = content_str.find(local_start_mark)
                    end_idx = content_str.find(
                        local_end_mark,
                        start_idx + len(local_start_mark)
                    ) if start_idx != -1 else -1
                    
                    if start_idx != -1 and end_idx != -1:
                        reasoning_text = content_str[
                            start_idx + len(local_start_mark):end_idx
                        ]
                        main_text = content_str[:start_idx] + content_str[
                            end_idx + len(local_end_mark):
                        ]
                    else:
                        main_text = content_str
            else:
                main_text = message.get("content", "")

            # 添加到上下文
            if add_to_context and main_text:
                await self.add_dialogue(main_text, "assistant")

            # 根据参数返回结果
            if not return_raw_response:
                if return_reasoning and local_is_reasoning_model:
                    return (main_text, reasoning_text)
                return main_text
            else:
                return response_json
        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP请求错误: {str(e)}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON解析错误: {str(e)}")
        except Exception as e:
            raise e

    async def stream_response(
        self,
        input_text: Union[str, list] = "",
        role: str = "user",
        no_input: bool = False,
        add_to_context: bool = True,
        override_params: Optional[Dict[str, Any]] = None,
        # 添加推理模型自定义属性参数
        is_reasoning_model: Optional[bool] = None,
        thought_format: Optional[Literal["openai", "string_token", "ollama", "auto", "none"]] = None,
        start_of_thinking_mark: Optional[str] = None,
        end_of_thinking_mark: Optional[str] = None,
        # 新增：推理内容匹配的临时覆盖参数
        reasoning_key: Optional[str] = None,
        reasoning_match_mode: Optional[Literal["full", "contains", "contained_by", "path"]] = None,
        reasoning_case_sensitive: Optional[bool] = None,
    ) -> AsyncGenerator[StreamReturn, None]:
        """流式响应生成器，支持灵活的推理格式处理。"""
        # 确定本次调用的有效参数
        local_is_reasoning_model = is_reasoning_model if is_reasoning_model is not None else self.is_reasoning_model
        effective_thought_format = thought_format if thought_format is not None else self.thought_format
        local_start_mark = start_of_thinking_mark if start_of_thinking_mark is not None else self.start_of_thinking_mark
        local_end_mark = end_of_thinking_mark if end_of_thinking_mark is not None else self.end_of_thinking_mark
        local_reasoning_key = reasoning_key if reasoning_key is not None else self.reasoning_key
        local_reasoning_match_mode = reasoning_match_mode if reasoning_match_mode is not None else self.reasoning_match_mode
        local_reasoning_case_sensitive = reasoning_case_sensitive if reasoning_case_sensitive is not None else self.reasoning_case_sensitive

        if effective_thought_format == "ollama":
            effective_thought_format = "string_token"
        
        if not no_input:
            if input_text or isinstance(input_text, list):
                await self.add_dialogue(input_text, role)
            else:
                raise ValueError("用户输入为空!")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }

        # 构建基础负载（启用流式）
        payload = self._build_payload(stream=True)
        
        # 如果有本次调用覆盖参数，合并到负载中（优先级最高）
        if override_params:
            payload.update(override_params)

        full_content = []  # 用于收集完整响应以添加到上下文
        session = self.session or aiohttp.ClientSession()

        # 'string_token' 格式专用缓冲区
        str_token_buffer = ""
        thinking_finished = False
        start_mark_removed = False

        try:
            async with session.post(
                self.chat_url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()

                # 处理流式响应
                async for line in response.content:
                    if not line:
                        continue
                    
                    decoded_line = line.decode("utf-8").strip()
                    if not decoded_line.startswith("data: "):
                        continue
                        
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices")
                        if not choices or not isinstance(choices, list):
                            continue
                        
                        delta = choices[0].get("delta", {})
                        if not delta:
                            continue

                        # ---- 推理与内容处理核心逻辑 ----
                        
                        # 自动格式检测 (仅在第一次有效delta时执行)
                        if local_is_reasoning_model and effective_thought_format == "auto":
                            # 只要找到字段（包括值为None/空）就认为是OpenAI格式
                            found = self._find_reasoning_content(
                                data, 
                                local_reasoning_key, 
                                local_reasoning_match_mode, 
                                local_reasoning_case_sensitive
                            )
                            
                            if found is not None:  # 字段存在（即使值为空）
                                effective_thought_format = "openai"
                                print(f"[INFO] Auto-detected format: openai (reasoning field found: {local_reasoning_key})")
                            else:
                                effective_thought_format = "string_token"
                                print("[INFO] Auto-detected format: string_token/ollama")

                        # 根据格式处理内容
                        if local_is_reasoning_model and effective_thought_format == "openai":
                            # 先尝试使用缓存的路径（如果有）
                            reasoning_chunk = None
                            if self._cached_reasoning_path:
                                reasoning_chunk = self._find_reasoning_content(
                                    data,
                                    self._cached_reasoning_path,
                                    "path",
                                    False  # path模式不区分大小写参数
                                )
                                # 若找不到（None），说明路径失效，清空缓存，后续走匹配逻辑
                                if reasoning_chunk is None:
                                    self._cached_reasoning_path = None

                            # 如果没有缓存或缓存刚失效，再用前三种模式尝试匹配并缓存路径
                            if reasoning_chunk is None and local_reasoning_match_mode in ("full", "contains", "contained_by"):
                                try:
                                    choice_obj = data.get("choices", [{}])[0]
                                except (IndexError, TypeError):
                                    choice_obj = {}
                                # 优先从 delta 取（流式场景通常为 delta）
                                if "delta" in choice_obj and isinstance(choice_obj["delta"], dict):
                                    search_dict = choice_obj["delta"]
                                    base_node = "delta"
                                elif "message" in choice_obj and isinstance(choice_obj["message"], dict):
                                    search_dict = choice_obj["message"]
                                    base_node = "message"
                                else:
                                    search_dict = {}
                                    base_node = None

                                if search_dict:
                                    ref = local_reasoning_key if local_reasoning_case_sensitive else local_reasoning_key.lower()
                                    for key, value in search_dict.items():
                                        current_key = key if local_reasoning_case_sensitive else key.lower()

                                        matched = (
                                            (local_reasoning_match_mode == "full" and current_key == ref)
                                            or (local_reasoning_match_mode == "contains" and ref in current_key)
                                            or (local_reasoning_match_mode == "contained_by" and current_key in ref)
                                        )
                                        if matched:
                                            # 命中即缓存路径（不论值是否为空/None）
                                            if base_node is not None:
                                                self._cached_reasoning_path = f"choices.0.{base_node}.{key}"
                                            # 处理值 -> 字符串
                                            if value is None:
                                                reasoning_chunk = ""
                                            elif isinstance(value, str):
                                                reasoning_chunk = value
                                            else:
                                                reasoning_chunk = ""
                                            break

                            # 主内容
                            content_chunk = delta.get("content", "")

                            # 组织返回
                            yield_data = StreamReturn(is_main_text=False, is_reasoning=False)

                            if reasoning_chunk:
                                yield_data.is_reasoning = True
                                yield_data.reasoning_text = reasoning_chunk

                            if content_chunk:
                                full_content.append(content_chunk)
                                yield_data.is_main_text = True
                                yield_data.main_text = content_chunk

                            if yield_data.is_reasoning or yield_data.is_main_text:
                                yield yield_data


                        else: # 非推理模型或 thought_format 为 'none'
                            content_chunk = delta.get("content", "")
                            # 某些模型可能将所有内容都放在reasoning字段
                            if not content_chunk:
                                content_chunk = self._find_reasoning_content(data, local_reasoning_key, local_reasoning_match_mode, local_reasoning_case_sensitive)

                            if content_chunk:
                                full_content.append(content_chunk)
                                yield StreamReturn(is_main_text=True, is_reasoning=False, main_text=content_chunk)

                    except json.JSONDecodeError:
                        continue
                
                # 处理流结束后缓冲区中的剩余内容
                if str_token_buffer:
                    if not thinking_finished: # 如果到最后也没找到结束标记，则全部视为思考内容
                         yield StreamReturn(is_main_text=False, is_reasoning=True, reasoning_text=str_token_buffer)
                    else: # 如果思考已结束，剩余部分为主内容
                        yield StreamReturn(is_main_text=True, is_reasoning=False, main_text=str_token_buffer)
                        full_content.append(str_token_buffer)

            # 将完整响应添加到上下文
            if add_to_context and full_content:
                await self.add_dialogue("".join(full_content), "assistant")

        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP请求错误: {str(e)}")
        except Exception as e:
            raise e
        finally:
            # 如果创建了临时session，则关闭它
            if not self.session and not session.closed:
                await session.close()


