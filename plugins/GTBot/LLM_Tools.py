from typing import List

from langchain_core.tools.base import BaseTool
from nonebot import on_command
from nonebot.adapters import Bot, Event
from nonebot.permission import SUPERUSER
from nonebot.log import logger

from plugins.GTBot.services.plugin_system.facade import (
    build_plugin_bundle,
    build_plugin_context,
    reload_plugins,
)

# --- 注册命令 ---
reload_matcher = on_command("重载工具", aliases={"reload_tools"}, permission=SUPERUSER, priority=1)

@reload_matcher.handle()
async def handle_reload(bot: Bot, event: Event):
    await reload_matcher.send("开始重载插件，请稍候...")
    
    try:
        reload_plugins()

        bundle = build_plugin_bundle(build_plugin_context(raw_messages=[], runtime_context=None))
        tool_names = []
        for t in bundle.tools:
            name = getattr(t, "name", None)
            if isinstance(name, str) and name:
                tool_names.append(name)

        msg = f"重载完成！当前共加载 {len(bundle.tools)} 个工具，{len(bundle.agent_middlewares)} 个中间件，{len(bundle.callbacks)} 个 callbacks。\n"
        if tool_names:
            msg += f"工具列表: {', '.join(tool_names)}"
        await reload_matcher.finish(msg)
        
    except Exception as e:
        logger.error(f"重载插件失败: {e}")
        await reload_matcher.finish(f"重载失败，请查看后台日志。\n错误: {str(e)}")

# --- 提供给外部获取工具的接口 ---
def get_current_tools() -> List[BaseTool]:
    bundle = build_plugin_bundle(build_plugin_context(raw_messages=[], runtime_context=None))
    return bundle.tools


__all__ = [
    "build_plugin_bundle",
    "build_plugin_context",
    "get_current_tools",
]
