from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


def _is_nonebot_not_initialized_error(exc: Exception) -> bool:
    return "NoneBot has not been initialized." in str(exc)


class PluginLoader:
    def __init__(self, plugin_dir: str | Path) -> None:
        self.plugin_dir = Path(plugin_dir).resolve()
        self.package_name: str | None = None
        self.modules: dict[str, ModuleType] = {}
        self._auto_configure_package()

    def _auto_configure_package(self) -> None:
        if not self.plugin_dir.exists():
            logger.warning(f"插件目录不存在: {self.plugin_dir}")
            return

        init_file = self.plugin_dir / "__init__.py"
        if not init_file.exists():
            try:
                init_file.touch()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"无法创建 __init__.py: {exc}")
                return

        parts: list[str] = []
        cursor = self.plugin_dir
        while (cursor / "__init__.py").exists():
            parts.append(cursor.name)
            cursor = cursor.parent

        if not parts:
            self.package_name = self.plugin_dir.name
            parent_dir = str(self.plugin_dir.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            return

        self.package_name = ".".join(reversed(parts))
        base_dir = str(cursor)
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)

    def iter_plugin_files(self) -> list[Path]:
        if not self.plugin_dir.exists():
            return []
        files: list[Path] = []
        for p in self.plugin_dir.glob("*.py"):
            if p.name.startswith("_") or p.name == "__init__.py":
                continue
            files.append(p)
        files.sort(key=lambda x: x.name)
        return files

    def iter_plugin_packages(self) -> list[Path]:
        if not self.plugin_dir.exists():
            return []

        packages: list[Path] = []
        for p in self.plugin_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith("_") or p.name == "__pycache__":
                continue
            if not (p / "__init__.py").exists():
                continue
            packages.append(p)
        packages.sort(key=lambda x: x.name)
        return packages

    def import_module(self, file_path: Path) -> ModuleType | None:
        if not self.package_name:
            return None

        module_name = file_path.stem
        full_name = f"{self.package_name}.{module_name}"
        try:
            importlib.invalidate_caches()

            # 强制清理可能残留的 pyc，避免在极短时间内写入源码导致 mtime 不变时仍加载旧字节码。
            try:
                pycache_dir = file_path.parent / "__pycache__"
                if pycache_dir.exists():
                    for pyc in pycache_dir.glob(f"{module_name}.*.pyc"):
                        try:
                            pyc.unlink()
                        except Exception:
                            pass
            except Exception:
                pass

            if full_name in sys.modules:
                mod = sys.modules[full_name]
            else:
                mod = importlib.import_module(full_name)

            self.modules[full_name] = mod
            return mod
        except Exception as exc:  # noqa: BLE001
            if _is_nonebot_not_initialized_error(exc):
                logger.warning(f"Skip plugin module requiring NoneBot init: {full_name}")
                logger.debug(traceback.format_exc())
                return None
            logger.error(f"加载插件模块失败: {full_name}: {exc}")
            logger.debug(traceback.format_exc())
            return None

    def import_package(self, package_dir: Path) -> ModuleType | None:
        if not self.package_name:
            return None

        module_name = package_dir.name
        full_name = f"{self.package_name}.{module_name}"
        try:
            importlib.invalidate_caches()

            try:
                pycache_dir = package_dir / "__pycache__"
                if pycache_dir.exists():
                    for pyc in pycache_dir.glob("*.pyc"):
                        try:
                            pyc.unlink()
                        except Exception:
                            pass
            except Exception:
                pass

            if full_name in sys.modules:
                mod = sys.modules[full_name]
            else:
                mod = importlib.import_module(full_name)

            self.modules[full_name] = mod
            return mod
        except Exception as exc:  # noqa: BLE001
            if _is_nonebot_not_initialized_error(exc):
                logger.warning(f"Skip plugin module requiring NoneBot init: {full_name}")
                logger.debug(traceback.format_exc())
                return None
            logger.error(f"加载插件模块失败: {full_name}: {exc}")
            logger.debug(traceback.format_exc())
            return None

    def call_register(self, module: ModuleType, registry: Any) -> bool:
        register_fn = getattr(module, "register", None)
        if register_fn is None:
            return False

        if not callable(register_fn):
            logger.error(f"插件 {module.__name__} 的 register 不是可调用对象")
            return True

        try:
            register_fn(registry)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"插件 register 执行失败: {module.__name__}: {exc}")
            logger.debug(traceback.format_exc())
            return True
