from pathlib import Path
import sys
from nonebot import get_driver
from nonebot.adapters.onebot.v11 import Bot
import asyncio

dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from RAGmanager import GroupRAGManager
from config_manager import config_group_data

GROUP_MESSAGE_COLLECTION_NAME = "group_message"

rag_initialization_complete = asyncio.Event()
config = config_group_data.Retrieval_Augmented_Generation

@get_driver().on_startup
async def main():
    global rag_manager, rag_initialization_complete
    rag_manager = await GroupRAGManager.init(
        config.chroma_service_url,
        config.ollama_service_url,
        config.embedding_model,
        group_collection_name=GROUP_MESSAGE_COLLECTION_NAME,
    )
    rag_initialization_complete.set()

