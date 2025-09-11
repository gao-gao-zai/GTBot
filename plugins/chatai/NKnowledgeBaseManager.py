import asyncio
from pathlib import Path
import sys
from nonebot import get_driver
dir_path = Path(__file__).parent.resolve()
sys.path.append(str(dir_path))

from config_manager import config_group_data
import KnowledgeBaseManager
from chatgpt import ChatGPT

config = config_group_data.Retrieval_Augmented_Generation
api = config_group_data.api

COLLECTION_NAME = "knowledge_base"

knowledge_base_manager: None|KnowledgeBaseManager.KnowledgeBaseManager = None

@get_driver().on_startup
async def _():
    global knowledge_base_manager
    if config.enable_knowledge_base:
        knowledge_base_manager = await KnowledgeBaseManager.KnowledgeBaseManager.create(
            collection=COLLECTION_NAME,
            chromadb_url=config.chroma_service_url,
            openai_api_key=api.embedding_ai_key,
            openai_base_url=api.embedding_ai_url,
            model_name=api.embedding_ai_model,
            tenant=config.tenant
        )
    else:
        knowledge_base_manager = None


async def get_knowledge_base_manager() -> KnowledgeBaseManager.KnowledgeBaseManager | None:
    if config.enable_knowledge_base:
        while knowledge_base_manager is None:
            await asyncio.sleep(0.1)
        return knowledge_base_manager
    else:
        return None

