import aiosqlite
import asyncio
from typing import List, Dict, Optional, Tuple, Union, Any, cast
from pathlib import Path
import time
import sys
from nonebot import get_driver
dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from config_manager import global_config_manager as gcm
from SQLiteManager import SQLiteManager, GroupMessageManager



chat_record_db = SQLiteManager(db_path=gcm.message_handling.chat_record_db_path)
image_description_cache_db = SQLiteManager(db_path=gcm.image_recognition.cache_db_path)
group_message_manager = GroupMessageManager(chat_record_db)



@get_driver().on_shutdown
async def close_db():
    await chat_record_db.close()
    await image_description_cache_db.close()