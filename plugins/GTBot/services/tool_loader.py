# tool_loader.py
from __future__ import annotations
import importlib
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Type
import logging

from langchain.tools import BaseTool

# 设置一个默认 logger，外部可以通过 logging.getLogger("ToolLoader") 获取并修改
logger = logging.getLogger("ToolLoader")

class ToolLoader:
    def __init__(self, tools_dir: str | Path, package_name: str | None = None):
        """
        :param tools_dir: 工具所在的物理目录
        :param package_name: 工具目录对应的 Python 包名 (例如 'my_bot.plugins.tools')
                             如果提供，将使用标准 import/reload 机制，支持相对导入。
                             如果不提供，将使用文件路径加载，不支持相对导入。
        """
        self.tools_dir = Path(tools_dir).resolve()
        self.package_name = package_name
        self.tools: List[BaseTool] = []

    def load_tools(self) -> List[BaseTool]:
        """加载所有工具并返回列表"""
        self.tools = []
        if not self.tools_dir.exists():
            logger.warning(f"工具目录不存在: {self.tools_dir}")
            return []

        logger.info(f"开始从 {self.tools_dir} 加载工具...")
        
        # 遍历目录下的 py 文件
        for file_path in self.tools_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            self._load_single_file(file_path)

        logger.info(f"工具加载完成，共加载 {len(self.tools)} 个工具")
        return self.tools

    def _load_single_file(self, file_path: Path):
        module_name = file_path.stem
        
        try:
            module = None
            
            # 策略 A: 如果提供了包名，使用标准 import/reload (推荐)
            # 这允许工具文件内部使用 "from .utils import x"
            if self.package_name:
                full_module_name = f"{self.package_name}.{module_name}"
                if full_module_name in sys.modules:
                    # 模块已存在，执行热重载
                    module = sys.modules[full_module_name]
                    try:
                        module = importlib.reload(module)
                    except ImportError:
                        # 极端情况：模块在 sys.modules 但文件可能被删了或损坏，尝试重新导入
                        module = importlib.import_module(full_module_name)
                else:
                    # 首次导入
                    module = importlib.import_module(full_module_name)
            
            # 策略 B: 纯文件路径加载 (后备方案)
            # 这种方式下，工具文件内部不能使用相对导入
            else:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module # 只有成功创建 module 后才注入
                    spec.loader.exec_module(module)

            if module:
                self._extract_tools_from_module(module)

        except Exception as e:
            logger.error(f"加载文件 {file_path.name} 失败: {e}")
            logger.debug(traceback.format_exc())

    def _extract_tools_from_module(self, module):
        """从模块中提取 BaseTool 实例或类"""
        found_in_module = 0
        
        for name, obj in inspect.getmembers(module):
            # 排除私有变量
            if name.startswith("_"):
                continue

            tool_instance = None

            # 情况 1: obj 是 BaseTool 的实例 (e.g. my_tool = Tool(...))
            if isinstance(obj, BaseTool):
                # 关键修复: 确保这个工具是在当前文件定义的，而不是 import 进来的
                # LangChain 的 @tool 装饰器生成的对象通常 module 会指向定义处
                if getattr(obj, "__module__", "") == module.__name__:
                    tool_instance = obj

            # 情况 2: obj 是 BaseTool 的子类 (e.g. class MyTool(BaseTool): ...)
            elif inspect.isclass(obj) and issubclass(obj, BaseTool) and obj is not BaseTool:
                if obj.__module__ == module.__name__:
                    try:
                        # 尝试无参实例化
                        tool_instance = obj()
                    except TypeError:
                        logger.warning(f"跳过类 {name}: 需要参数才能实例化，请在文件中实例化它。")
            
            if tool_instance:
                self.tools.append(tool_instance)
                found_in_module += 1
        
        if found_in_module > 0:
            logger.info(f"  -> 从 {module.__name__} 加载了 {found_in_module} 个工具")
