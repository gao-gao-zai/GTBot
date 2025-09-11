from pathlib import Path
import sys
from nonebot import logger
from nonebot import get_driver
from nonebot.adapters.onebot.v11 import Bot
import asyncio
import httpx


dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from plugins.chatai.ChatLogRetrievalManager import GroupChatLogRetrievalManager
from config_manager import config_group_data

GROUP_MESSAGE_COLLECTION_NAME = "group_message"


config = config_group_data.Retrieval_Augmented_Generation
api = config_group_data.api

chat_log_retrieval_manager = None

@get_driver().on_startup
async def main():
    global chat_log_retrieval_manager
    if config.enable:
        try:
            chat_log_retrieval_manager = await GroupChatLogRetrievalManager.init(
                chromadb_url=config.chroma_service_url,
                openai_api_key=api.embedding_ai_key,
                openai_url=api.embedding_ai_url,
                model_name=api.embedding_ai_model,
                tenant=config.tenant,
                group_collection_name=GROUP_MESSAGE_COLLECTION_NAME,
            )
        except httpx.ConnectError as e:
            chat_log_retrieval_manager = None
            logger.error(f"连接时错误:{e}, 已禁用RAG")
        
    else:
        chat_log_retrieval_manager = None
    # rag_initialization_complete.set()

@get_driver().on_shutdown
async def shutdown():
    if chat_log_retrieval_manager is not None:
        await chat_log_retrieval_manager.close()

# 定义一个异步的依赖提供器
async def get_rag_manager() -> GroupChatLogRetrievalManager | None:
    if config.enable:
        # 如果还没初始化，等待它初始化完成
        while chat_log_retrieval_manager is None:
            await asyncio.sleep(0.1)  # 短暂等待
        return chat_log_retrieval_manager
    else:
        return None
    