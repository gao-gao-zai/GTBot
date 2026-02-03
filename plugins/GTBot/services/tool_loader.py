# tool_loader.py
from __future__ import annotations
import importlib
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from typing import List

from langchain.tools import BaseTool
from nonebot import logger

class ToolLoader:
    def __init__(self, tools_dir: str | Path):
        """
        自动化的工具加载器。
        
        Args:
            tools_dir: 插件所在的文件夹路径 (相对或绝对路径皆可)
                       会自动将其识别为 Python 包，支持相对导入 (from .api import ...)
        """
        self.tools_dir = Path(tools_dir).resolve()
        self.package_name: str | None = None
        self.tools: List[BaseTool] = []
        
        # === 核心自动化逻辑 ===
        self._auto_configure_package()

    def _auto_configure_package(self):
        """自动配置包环境：挂载路径、创建init、推断包名"""
        if not self.tools_dir.exists():
            logger.warning(f"工具目录不存在: {self.tools_dir}")
            return

        # 1. 确保目录包含 __init__.py (否则无法作为包导入)
        init_file = self.tools_dir / "__init__.py"
        if not init_file.exists():
            try:
                init_file.touch()
                logger.debug(f"自动创建 __init__.py 于: {self.tools_dir}")
            except Exception as e:
                logger.error(f"无法创建 __init__.py: {e}")
                return

        # 2. 确定包名：直接使用文件夹名称
        # 例如路径是 .../custom_plugins/finance_tools
        # 包名就是 finance_tools
        self.package_name = self.tools_dir.name

        # 3. 挂载父目录到 sys.path
        # 这样 Python 才能通过 import finance_tools 找到它
        parent_dir = str(self.tools_dir.parent)
        
        # 检查是否已经在 path 中 (避免重复添加)
        # 注意：这里简单的检查可能不够，但在大多数插件场景下足够用了
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            logger.info(f"Auto-Mount: 将插件父目录加入环境: {parent_dir}")
        
        logger.info(f"插件包环境已就绪: package='{self.package_name}' path='{self.tools_dir}'")

    def load_tools(self) -> List[BaseTool]:
        """加载所有工具并返回列表"""
        self.tools = []
        if not self.tools_dir.exists() or not self.package_name:
            return []

        logger.info(f"开始加载插件包: {self.package_name} ...")

        # 遍历目录下的 py 文件
        for file_path in self.tools_dir.glob("*.py"):
            # 跳过私有文件、__init__.py 和 api定义文件
            if file_path.name.startswith("_") or file_path.name == "api.py":
                continue
            
            self._load_single_file(file_path)

        logger.info(f"插件包 {self.package_name} 加载完成，共 {len(self.tools)} 个工具")
        return self.tools

    def _load_single_file(self, file_path: Path):
        module_name = file_path.stem
        # 拼接完整包路径： package.module
        full_module_name = f"{self.package_name}.{module_name}"
        
        try:
            module = None
            
            # 智能加载逻辑：存在则重载，不存在则导入
            if full_module_name in sys.modules:
                module = sys.modules[full_module_name]
                try:
                    module = importlib.reload(module)
                except ImportError:
                    # 模块可能因为文件移动等原因损坏，尝试重新导入
                    module = importlib.import_module(full_module_name)
                except Exception as e:
                    logger.error(f"热重载 {full_module_name} 失败: {e}")
                    return
            else:
                module = importlib.import_module(full_module_name)
            
            if module:
                self._extract_tools_from_module(module)

        except Exception as e:
            logger.error(f"加载模块 {full_module_name} 失败: {e}")
            logger.debug(traceback.format_exc())

    def _extract_tools_from_module(self, module):
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue

            if isinstance(obj, BaseTool):
                self.tools.append(obj)
            
            # (类定义的检查保留，因为类通常会有 __module__ 属性，检查一下更安全)
            elif inspect.isclass(obj) and issubclass(obj, BaseTool) and obj is not BaseTool:
                if obj.__module__ == module.__name__:
                    try:
                        self.tools.append(obj())
                    except:
                        pass