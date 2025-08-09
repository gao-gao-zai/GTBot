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
        context: Optional[List[Dict[str, Union[str, bool, list]]]] = None,
    ):
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.model: str = model
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.top_p: float = top_p
        self.frequency_penalty: float = frequency_penalty
        self.prompt: str = prompt
        self.context: list[dict] = context.copy() if context is not None else []
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_context(self) -> list[dict]:
        return copy.deepcopy(self.context)

    async def add_dialogue(
        self, content: Union[str, list], role: str, lock: bool = False
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

    async def get_response(
        self,
        input_text: Union[str, list] = "",
        role: str = "user",
        only_content: bool = True,
        no_input: bool = False,
        add_to_context: bool = True,
    ):
        """
        获取模型响应，支持多模态输入
        Args:
            input_text: 用户输入（文本或内容数组）
            role: 角色(user/assistant/system)
            only_content: 是否只返回内容
            no_input: 是否没有输入
            add_to_context: 是否将模型响应添加到上下文
        """
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
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        response_json = await response.json()
            else:
                async with self.session.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
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
        add_to_context: bool = True,
    ) -> AsyncGenerator[str, None]:
        """流式响应生成器，逐块返回模型输出，支持多模态输入"""
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

        payload = {
            "model": self.model,
            "messages": self.context,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "stream": True,  # 启用流式响应
        }

        full_content = []  # 用于收集完整响应
        session = self.session or aiohttp.ClientSession()

        try:
            # 使用相同的session管理逻辑
            if self.session:
                async with self.session.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                ) as response:
                    response.raise_for_status()

                    # 处理流式响应
                    async for line in response.content:
                        if line:
                            decoded_line = line.decode("utf-8").strip()
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
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                ) as response:
                    response.raise_for_status()

                    # 处理流式响应
                    async for line in response.content:
                        if line:
                            decoded_line = line.decode("utf-8").strip()
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
                full_text = "".join(full_content)
                await self.add_dialogue(full_text, "assistant")

        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP请求错误: {str(e)}")
        except Exception as e:
            raise e
        finally:
            # 如果创建了临时session，则关闭它
            if not self.session and not session.closed:
                await session.close()


from pathlib import Path
base_url = "http://166.108.192.205:40002/v1"
api_key = "sk-6fW34zquoQ0Bk7ry65xcSU0INBFF8o91"
dir_path = Path(__file__).parent
test_dir = dir_path / "test_dir"

async def multimodal_models_test():

    t1 = time.time()

    # 使用异步上下文管理器
    async with ChatGPT(
        api_key,
        base_url,
        model="qwen/qwen2.5-vl-32b-instruct:free",  # 使用支持多模态的模型
    ) as chat:
        t2 = time.time()
        print(f"初始化时间: {t2 - t1:.2f}秒")

        # 添加本地图像文件
        image_path = r"D:\QQBOT\nonebot\ggz\plugins\chatai\BF5195EAAB81304B4D3CE0C6CB0209F7.gif"  # 替换为你的图片路径
        await chat.add_local_file(
            image_path, "请分析这张图片：", detail="high"  # 高分辨率模式
        )

        # 获取多模态响应
        print("\n多模态响应:")
        response = await chat.get_response(no_input=True)
        print(response)

        print(f"\n总耗时: {time.time() - t2:.2f}秒")
    


async def Thinking_models_test():
    # async with ChatGPT(
    #     api_key,
    #     base_url,
    #     model="gpt-oss-20b",  # 使用支持思维链的模型
    # ) as chat:
    #     result = await chat.get_response("简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "gpt_think_models_result.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # async with ChatGPT(
    #     api_key,
    #     base_url,
    #     model="glm-4.5-air",  # 使用支持思维链的模型
    # ) as chat:
    #     result = await chat.get_response("简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "glm_think_models_result.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # async with ChatGPT(
    #     api_key,
    #     base_url,
    #     model="qwen3-235b-a22b",  # 使用支持思维链的模型
    # ) as chat:
    #     result = await chat.get_response("简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "qwen_think_models_result.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # async with ChatGPT(
    #     api_key,
    #     base_url,
    #     model="qwen3-235b-a22b",  # 使用支持思维链的模型
    # ) as chat:
    #     result = await chat.get_response("/no_think\n简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "qwen_no_think_models_result.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # async with ChatGPT(
    #     api_key,
    #     base_url,
    #     model="deepseek-v3-0324", 
    # ) as chat:
    #     result = await chat.get_response("简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "deepseek_no_think_models_result.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    ollama_url = "http://100.112.88.118:11434/v1"
    # async with ChatGPT(
    #     api_key,
    #     ollama_url,
    #     model="deepseek-r1:1.5b",
    # ) as chat:
    #     result = await chat.get_response("简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "deepseek_think_models_result_ollama.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # async with ChatGPT(
    #     api_key,
    #     ollama_url,
    #     model="qwen2.5-coder:1.5b",
    # ) as chat:
    #     result = await chat.get_response("简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "qwen_no_think_models_result_ollama.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # async with ChatGPT(
    #     api_key,
    #     ollama_url,
    #     model="qwen3:0.6b",
    # ) as chat:
    #     result = await chat.get_response("简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "qwen3_think_models_result_ollama.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # async with ChatGPT(
    #     api_key,
    #     ollama_url,
    #     model="qwen3:0.6b",
    # ) as chat:
    #     result = await chat.get_response("/no_think\n简单介绍你自己", only_content=False, delete_superfluous_dialogue=False)
    #     print(result)
    #     with open(dir_path / "qwen3_no_think_text_models_result_ollama.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    async with ChatGPT(
        api_key,
        base_url,
        model="qwen3-235b-a22b",
    ) as chat:
        result = await chat.get_response("/no_think\n简单介绍你自己", only_content=False)
        print(result)
        with open(test_dir / "qwen3-235b-a22b_no_think_text_models_result_ollama.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    async with ChatGPT(
        api_key,
        base_url,
        model="qwen3-235b-a22b",
    ) as chat:
        result = await chat.get_response("简单介绍你自己", only_content=False)
        print(result)
        with open(test_dir / "qwen3-235b-a22b_think_models_result_ollama.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    asyncio.run(Thinking_models_test())
    pass
