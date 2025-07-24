import nonebot
from nonebot import logger
from nonebot.adapters.onebot.v11 import Adapter as ONEBOT_V11Adapter
import sys
import os
from pathlib import Path
import json
import asyncio
import time
dir_path = Path(__file__).parent.absolute()
plugins_path = dir_path/"plugins"
logger.info(f"工作目录: {dir_path}")
logger.info(f"插件目录: {plugins_path}")
os.chdir(dir_path)

os.environ["COMMAND_START"] = json.dumps(["/", "#"])
os.environ["COMMAND_SEP"] = json.dumps(["."])


nonebot.init(
    log_level="DEBUG",
)

driver = nonebot.get_driver()
driver.register_adapter(ONEBOT_V11Adapter)

# nonebot.load_builtin_plugins('echo')
# nonebot.load_plugin("nonebot_plugin_status")
# nonebot.load_plugins("plugins")

nonebot.load_from_toml("pyproject.toml")

if __name__ == "__main__":
    nonebot.run()
    