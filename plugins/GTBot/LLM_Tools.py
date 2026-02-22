# nonebot_plugin.py (或者 __init__.py)
from pathlib import Path
from typing import List
from langchain_core.tools.base import BaseTool
from nonebot import on_command, get_driver
from nonebot.adapters import Bot, Event
from nonebot.permission import SUPERUSER
from nonebot.log import logger

# 导入你的配置管理 (假设路径没变)
from .ConfigManager import total_config, Processed
from .services.plugin_system import PluginManager, PluginContext

# --- 初始化配置 ---
config: Processed.GeneralConfiguration = total_config.processed_configuration.config
tools_dir: Path = config.tools_dir

# --- 构造包名 ---
# 假设当前文件在 mybot.plugins.my_plugin.__init__
# 且工具在 mybot.plugins.my_plugin.tools 目录下
# 这样 ToolLoader 就能利用 Python 的 import 机制处理依赖
current_package = __package__
tools_package_name = f"{current_package}.tools" if current_package else None

# --- 初始化 PluginManager ---
plugin_manager = PluginManager(plugin_dir=tools_dir)

logger.info("正在初始化加载插件...")
plugin_manager.load()

# --- 注册命令 ---
reload_matcher = on_command("重载工具", aliases={"reload_tools"}, permission=SUPERUSER, priority=1)

@reload_matcher.handle()
async def handle_reload(bot: Bot, event: Event):
    await reload_matcher.send("开始重载插件，请稍候...")
    
    try:
        plugin_manager.reload()

        bundle = plugin_manager.build(PluginContext(raw_messages=[]))
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
    bundle = plugin_manager.build(PluginContext(raw_messages=[]))
    return bundle.tools


def build_plugin_context(*, raw_messages: list, runtime_context=None) -> PluginContext:
    return PluginContext(raw_messages=raw_messages, runtime_context=runtime_context)


def build_plugin_bundle(ctx: PluginContext):
    return plugin_manager.build(ctx)
