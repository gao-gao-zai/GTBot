import json
import time
import base64
import os
import mimetypes
from typing import List, Dict, Optional, AsyncGenerator, Union
import tiktoken
import copy
from rich import print
import asyncio
import aiohttp

class ChatGPT:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        prompt: str = "",
        context_max_tokens: int = 4096,
        context: Optional[List[Dict[str, Union[str, bool, list]]]] = None,
        tokeniser_model: str = "gpt-3.5-turbo",
    ):
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.model: str = model
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.top_p: float = top_p
        self.frequency_penalty: float = frequency_penalty
        self.prompt: str = prompt
        self.context_max_tokens: int = context_max_tokens
        self.context: list[dict] = context.copy() if context is not None else []
        self.tokeniser_model: str = tokeniser_model
        self.session: Optional[aiohttp.ClientSession] = None
        
        try:
            self.tokeniser = tiktoken.encoding_for_model(self.model)
        except:
            self.tokeniser = tiktoken.encoding_for_model(self.tokeniser_model)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None

    async def text_to_token(
        self,
        text: Union[str, list],
        model: Optional[str] = None,
        encoding: Optional[tiktoken.Encoding] = None,
    ) -> int:
        """计算文本或内容数组的token数量"""
        model_ = model if model is not None else self.model
        if encoding is None:
            encoding = self.tokeniser
        
        # 处理多模态内容（文本+图像）
        if isinstance(text, list):
            token_count = 0
            for item in text:
                if item["type"] == "text":
                    token_count += len(encoding.encode(item["text"]))
                elif item["type"] == "image_url":
                    # 图像URL的固定token开销（根据OpenAI文档）
                    token_count += 85  # 低分辨率图像的基准token
                    # 如果指定了高分辨率，则增加token
                    if "detail" in item["image_url"] and item["image_url"]["detail"] == "high":
                        token_count += 170 * 2  # 高分辨率图像的额外token
            return token_count
        
        # 处理纯文本
        return len(encoding.encode(text))

    async def context_to_token(
        self,
        context: Optional[list[dict]] = None,
        model: Optional[str] = None,
        encoding: Optional[tiktoken.Encoding] = None,
    ) -> int:
        if context is None:
            context = self.context
        model_ = model if model is not None else self.model
        if encoding is None:
            encoding = self.tokeniser
        token_count = 0
        for entry in context:
            token_count += await self.text_to_token(entry["content"], model=model_, encoding=encoding)
        return token_count

    async def get_context(self) -> list[dict]:
        return copy.deepcopy(self.context)

    async def add_dialogue(
        self, 
        content: Union[str, list], 
        role: str, 
        lock: bool = False
    ):
        """添加对话内容，支持纯文本或多模态内容数组"""
        if role not in ["user", "assistant", "system"]:
            raise ValueError("角色必须是 'user', 'assistant' 或 'system' 之一。")
        
        # 如果是字符串，转换为标准内容格式
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        
        self.context.append({"role": role, "content": content, "lock": lock})
    
    async def add_image(
        self, 
        image_url: str, 
        text: str = "", 
        detail: str = "auto",
        role: str = "user",
        lock: bool = False
    ):
        """添加图像URL到上下文，支持详细程度设置"""
        content = []
        
        if text:
            content.append({"type": "text", "text": text})
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": detail  # auto, low, high
            }
        })
        
        await self.add_dialogue(content, role, lock)

    async def add_local_file(
        self, 
        file_path: str, 
        text: str = "", 
        detail: str = "auto",
        role: str = "user",
        lock: bool = False
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
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": detail
                    }
                })
        else:
            # 处理文本文件
            try:
                with open(file_path, "r", encoding="utf-8") as text_file:
                    file_content = text_file.read()
                    content.append({
                        "type": "text",
                        "text": f"文件内容 ({os.path.basename(file_path)}):\n{file_content}"
                    })
            except UnicodeDecodeError:
                # 如果是二进制文件，无法读取为文本
                content.append({
                    "type": "text",
                    "text": f"无法读取二进制文件: {os.path.basename(file_path)}"
                })
        
        await self.add_dialogue(content, role, lock)

    async def delete_superfluous_dialogue(
        self, context_max_tokens: Optional[int] = None, defy_lock: bool = False
    ):
        if context_max_tokens is None:
            context_max_tokens = self.context_max_tokens
        if context_max_tokens <= 0:
            raise ValueError("上下文最大 token 数必须大于 0。")
        
        current_tokens = await self.context_to_token()
        if current_tokens <= context_max_tokens:
            return
        
        # 创建临时副本来操作
        temp_context = copy.deepcopy(self.context)
        while await self.context_to_token(temp_context) > context_max_tokens:
            removed = False
            for idx, entry in enumerate(temp_context):
                if entry.get("lock", False) and not defy_lock:
                    continue
                del temp_context[idx]
                removed = True
                break
            if not removed:
                # 如果无法删除更多，尝试强制删除（即使锁定的）
                for idx, entry in enumerate(temp_context):
                    del temp_context[idx]
                    removed = True
                    break
                if not removed:
                    raise RuntimeError(f"无法缩减上下文到 {context_max_tokens} tokens")
        
        # 更新实际上下文
        self.context = temp_context

    async def get_response(
        self,
        input_text: Union[str, list] = "",
        role: str = "user",
        only_content: bool = True,
        no_input: bool = False,
        add_to_context: bool = True,
        delete_superfluous_dialogue: bool = True,
    ):
        """
        获取模型响应，支持多模态输入
        Args:
            input_text: 用户输入（文本或内容数组）
            role: 角色(user/assistant/system)
            only_content: 是否只返回内容
            no_input: 是否没有输入
            add_to_context: 是否将模型响应添加到上下文
            delete_superfluous_dialogue: 是否自动删除多余的对话
        """
        if not no_input:
            if input_text or isinstance(input_text, list):
                await self.add_dialogue(input_text, role)
            else:
                raise ValueError("用户输入为空!")
        
        if delete_superfluous_dialogue:
            await self.delete_superfluous_dialogue()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity"
        }
        
        payload = {
            "model": self.model,
            "messages": self.context,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
        }
        
        try:
            # 如果没有外部管理的session，创建临时session
            if not self.session:
                async with aiohttp.ClientSession() as temp_session:
                    async with temp_session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        response.raise_for_status()
                        response_json = await response.json()
            else:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
            
            content = response_json["choices"][0]["message"]["content"]
            
            if add_to_context and content:
                # 将响应转换为标准内容格式
                await self.add_dialogue(content, "assistant")
            
            if content is None:
                raise ValueError("模型未返回响应内容")
            
            return content if only_content else response_json
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
        delete_superfluous_dialogue: bool = True,
        add_to_context: bool = True
    ) -> AsyncGenerator[str, None]:
        """流式响应生成器，逐块返回模型输出，支持多模态输入"""
        if not no_input:
            if input_text or isinstance(input_text, list):
                await self.add_dialogue(input_text, role)
            else:
                raise ValueError("用户输入为空!")
        
        if delete_superfluous_dialogue:
            await self.delete_superfluous_dialogue()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity"
        }
        
        payload = {
            "model": self.model,
            "messages": self.context,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "stream": True  # 启用流式响应
        }
        
        full_content = []  # 用于收集完整响应
        session = self.session or aiohttp.ClientSession()
        
        try:
            # 使用相同的session管理逻辑
            if self.session:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    # 处理流式响应
                    async for line in response.content:
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]  # 去掉"data: "前缀
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0]["delta"]
                                        if "content" in delta:
                                            content_chunk = delta["content"]
                                            full_content.append(content_chunk)
                                            yield content_chunk  # 产生当前内容块
                                except json.JSONDecodeError:
                                    continue
            else:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    # 处理流式响应
                    async for line in response.content:
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]  # 去掉"data: "前缀
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0]["delta"]
                                        if "content" in delta:
                                            content_chunk = delta["content"]
                                            full_content.append(content_chunk)
                                            yield content_chunk  # 产生当前内容块
                                except json.JSONDecodeError:
                                    continue
            
            # 将完整响应添加到上下文
            if add_to_context and full_content:
                full_text = ''.join(full_content)
                await self.add_dialogue(full_text, "assistant")
                
        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP请求错误: {str(e)}")
        except Exception as e:
            raise e
        finally:
            # 如果创建了临时session，则关闭它
            if not self.session and not session.closed:
                await session.close()


async def main():
    base_url = 'http://localhost:30001/openrouter/'
    api_key = 'sk-or-v1-c929789de5ca'
    t1 = time.time()

    # 使用异步上下文管理器
    async with ChatGPT(
        api_key, 
        base_url, 
        model='qwen/qwen2.5-vl-32b-instruct:free'  # 使用支持多模态的模型
    ) as chat:
        t2 = time.time()
        print(f"初始化时间: {t2 - t1:.2f}秒")
        
        # 添加本地图像文件
        image_path = r"D:\QQBOT\nonebot\ggz\plugins\chatai\BF5195EAAB81304B4D3CE0C6CB0209F7.gif"  # 替换为你的图片路径
        await chat.add_local_file(
            image_path, 
            "请分析这张图片：",
            detail="high"  # 高分辨率模式
        )
        
        # 获取多模态响应
        print("\n多模态响应:")
        response = await chat.get_response(no_input=True)
        print(response)
        
        print(f"\n总耗时: {time.time() - t2:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())