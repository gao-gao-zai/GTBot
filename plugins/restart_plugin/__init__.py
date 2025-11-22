import inspect
import socket
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

from nonebot import get_driver, on_command, logger
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata
from nonebot.rule import to_me
import time
from nonebot.adapters.onebot.v11 import Event, Bot, GroupMessageEvent, PrivateMessageEvent

if TYPE_CHECKING:
    from uvicorn.server import Server

from pathlib import Path
import sys
dir_path = Path(__file__).parent

from .Permissions import owner, admin, not_black_user, not_black_group




driver = get_driver()

__plugin_meta__ = PluginMetadata(
    name="再润",
    description="NoneBot2 停机与重启控制插件",
    usage=(
        "（@）（指令前缀）stop|shutdown|停机\n"
        "    停止 Bot 运行。\n"
        "    使用者需要为 Bot 的 SuperUser。\n"
        "（@）（指令前缀）restart|reboot|重启\n"
        "    停止运行并重启 Bot。\n"
        "    使用者需要为 Bot 的 SuperUser。"
    ),
    type="application",
    homepage="https://github.com/NCBM/nonebot-plugin-runagain",
    supported_adapters=None
)

def _restart():
    from os import execlp
    nb = which("nb")
    py = which("python")
    if nb:
        execlp(nb, nb, "run")
    elif py:
        from sys import argv
        if argv and Path(argv[0]).exists():
            execlp(py, py, argv[0])
        if Path("bot.py").exists():
            execlp(py, py, "bot.py")
    raise Exception("cannot restart")


def _uvicorn_getserver() -> "Server":
    from uvicorn.server import Server
    fis = inspect.getouterframes(inspect.currentframe())
    svrs = (fi.frame.f_locals.get("server", None) for fi in fis[::-1])
    server, *_ = (s for s in svrs if isinstance(s, Server))
    return server


def _uvicorn_getsocket() -> list[socket.socket]:
    fis = inspect.getouterframes(inspect.currentframe())
    skvars = (fi.frame.f_locals.get("sockets", None) for fi in fis[::-1])
    try:
        socks, *_ = (
            s for s in skvars
            if isinstance(s, list) and all(isinstance(x, socket.socket) for x in s)
        )
        return socks
    except Exception:
        return []


def _none_stop():
    if TYPE_CHECKING:
        from nonebot.drivers.none import Driver as NoneDriver
        assert isinstance(driver, NoneDriver)
    driver.exit()


def _uvicorn_stop():
    server = _uvicorn_getserver()
    server.should_exit = True


stop_m = on_command("stop", rule=to_me(), aliases={"shutdown", "停机", "关机"}, block=True, priority=-10000000)
restart_m = on_command("restart", rule=to_me(), aliases={"reboot", "重启"}, block=True, priority=-10000000)

import json


@driver.on_bot_connect
async def _(bot:Bot):
    if Path(dir_path/"restart.json").is_file():
        with open(dir_path/"restart.json", "r", encoding="utf-8") as f:
            restart = json.load(f)
        try:
            if restart["is_restart"]:
                time_ = restart["time"]
                if restart["is_group"]:
                    await bot.send_group_msg(group_id=restart["group_id"], message=f"已重启，耗时{round(time.time()-time_, 2)}秒")
                else:
                    await bot.send_private_msg(user_id=restart["user_id"], message=f"已重启，耗时{round(time.time()-time_, 2)}秒")
        except Exception as e:
            logger.error(f"重启信息读取失败: {e}")
        with open(dir_path/"restart.json", "w") as f:
            json.dump({"is_restart":False}, f)

@stop_m.handle()
async def do_stop(bot:Bot, event: Event):
    if owner(bot, event) or admin(bot, event):
        await stop_m.send("正在停止...")
        if "fastapi" in driver.type or "quart" in driver.type:
            _uvicorn_stop()
        if "none" in driver.type:
            _none_stop()
    else:
        await stop_m.finish("权限不足")

@restart_m.handle()
async def do_restart(bot:Bot, event: Event):
    import atexit
    if owner(bot, event) or admin(bot, event):
        import json
        await restart_m.send("正在重启...")
        if isinstance(event, GroupMessageEvent):
            is_group = True
            id = event.group_id
        elif isinstance(event, PrivateMessageEvent):
            is_group = False
            id = event.user_id
        else:
            await restart_m.finish("未知错误")
        with open(dir_path/"restart.json", "w", encoding="utf-8") as f:
            json.dump({"is_restart":True,"time": time.time(),"is_group":is_group,"id":id}, f)
        if "none" in driver.type:
            atexit.register(_restart)
            _none_stop()
        if "fastapi" in driver.type or "quart" in driver.type:
            server = _uvicorn_getserver()
            server.force_exit = True
            await server.shutdown(_uvicorn_getsocket())
            try:
                _restart()
            except Exception:
                await do_stop()
    else:
        await restart_m.finish("权限不足")