from nonebot import on_command
from nonebot import get_driver
import time
status = on_command("status", aliases={"状态"}, priority=5, block=True)
driver = get_driver()


start_time = 0.0

@driver.on_startup
async def _():
    global start_time
    start_time = time.time()

@status.handle()
async def _(bot, event):
    await status.finish(f"运行时间:{round(time.time()-start_time, 2)}秒")
