# nonebot_plugin.py (或者 __init__.py)
from pathlib import Path
from nonebot import on_command, get_driver
from nonebot.adapters import Bot, Event
from nonebot.permission import SUPERUSER
from nonebot.log import logger

# 导入上面拆分出来的 Loader
from .services.tool_loader import ToolLoader
# 导入你的配置管理 (假设路径没变)
from .ConfigManager import total_config, Processed

# --- 初始化配置 ---
config: Processed.GeneralConfiguration = total_config.processed_configuration.config
tools_dir: Path = config.tools_dir

# --- 构造包名 ---
# 假设当前文件在 mybot.plugins.my_plugin.__init__
# 且工具在 mybot.plugins.my_plugin.tools 目录下
# 这样 ToolLoader 就能利用 Python 的 import 机制处理依赖
current_package = __package__
tools_package_name = f"{current_package}.tools" if current_package else None

# --- 初始化 Loader ---
# 将 NoneBot 的 logger 绑定给 ToolLoader，统一日志输出
import logging
loader_logger = logging.getLogger("ToolLoader")
loader_logger.setLevel(logging.INFO)
# 这里简单地把 handler 接管过来，或者直接依赖 print 输出到 stdout 被 nb 捕获
# 更好的做法是 ToolLoader 内部只用 logging，不配置 handler，由主程序配置

tool_loader = ToolLoader(
    tools_dir=tools_dir,
    package_name=tools_package_name
)

# 初始加载
logger.info("正在初始化加载 LangChain 工具...")
global_tools = tool_loader.load_tools()

# --- 注册命令 ---
reload_matcher = on_command("重载工具", aliases={"reload_tools"}, permission=SUPERUSER, priority=1)

@reload_matcher.handle()
async def handle_reload(bot: Bot, event: Event):
    await reload_matcher.send("开始重载工具，请稍候...")
    
    try:
        # 重新执行加载逻辑
        new_tools = tool_loader.load_tools()
        
        # 更新全局变量引用
        global global_tools
        global_tools = new_tools
        
        # 【重要提示】
        # 如果你的 LangChain Agent 是在启动时初始化的，仅仅更新 global_tools 是不够的。
        # 你需要在这里添加逻辑，去重建 Agent 或者更新 Agent 内部的 tools 列表。
        # 例如: agent_manager.refresh_agent(new_tools)
        
        msg = f"重载完成！当前共加载 {len(new_tools)} 个工具。\n"
        msg += f"工具列表: {', '.join([t.name for t in new_tools])}"
        await reload_matcher.finish(msg)
        
    except Exception as e:
        logger.error(f"重载工具失败: {e}")
        await reload_matcher.finish(f"重载失败，请查看后台日志。\n错误: {str(e)}")

# --- 提供给外部获取工具的接口 ---
def get_current_tools():
    return global_tools
