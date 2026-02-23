from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)

from .loader import PluginLoader
from .registry import PluginRegistry
from .types import PluginBundle, PluginContext


@dataclass
class PluginManager:
    plugin_dir: str | Path

    def __post_init__(self) -> None:
        self._loader = PluginLoader(self.plugin_dir)
        self._registry = PluginRegistry()
        self._loaded = False

    def load(self) -> None:
        """加载插件目录。

        Raises:
            RuntimeError: 当插件目录不可用时抛出。
        """

        self._registry = PluginRegistry()
        if not self._loader.plugin_dir.exists():
            logger.warning(f"插件目录不存在，跳过加载: {self._loader.plugin_dir}")
            self._loaded = True
            return

        for file_path in self._loader.iter_plugin_files():
            mod = self._loader.import_module(file_path)
            if mod is None:
                continue

            handled = self._loader.call_register(mod, self._registry)
            if handled:
                continue

        for package_dir in self._loader.iter_plugin_packages():
            mod = self._loader.import_package(package_dir)
            if mod is None:
                continue

            handled = self._loader.call_register(mod, self._registry)
            if handled:
                continue

        self._loaded = True

    def reload(self) -> None:
        """重载插件。"""

        self.load()

    def build(self, ctx: PluginContext) -> PluginBundle:
        """基于当前插件注册信息，构建本次调用的插件 bundle。"""

        if not self._loaded:
            self.load()

        tools: list[Any] = []
        middlewares: list[Any] = []
        callbacks: list[Any] = []

        for item in sorted(self._registry.iter_tools(), key=lambda x: x.priority):
            if item.enabled is not None:
                try:
                    if not bool(item.enabled(ctx)):
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"tool enabled 判断失败，跳过: {exc}")
                    continue
            tools.append(item.tool)

        for item in sorted(self._registry.iter_tool_factories(), key=lambda x: x.priority):
            if item.enabled is not None:
                try:
                    if not bool(item.enabled(ctx)):
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"tool_factory enabled 判断失败，跳过: {exc}")
                    continue

            try:
                produced = item.factory(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"tool_factory 执行失败，跳过: {exc}")
                continue

            if produced:
                tools.extend(list(produced))

        for item in sorted(self._registry.iter_middlewares(), key=lambda x: x.priority):
            if item.enabled is not None:
                try:
                    if not bool(item.enabled(ctx)):
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"middleware enabled 判断失败，跳过: {exc}")
                    continue
            middlewares.append(item.middleware)

        for item in sorted(self._registry.iter_callbacks(), key=lambda x: x.priority):
            if item.enabled is not None:
                try:
                    if not bool(item.enabled(ctx)):
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"callback enabled 判断失败，跳过: {exc}")
                    continue
            callbacks.append(item.callback)

        return PluginBundle(
            tools=tools,
            agent_middlewares=middlewares,
            callbacks=callbacks,
        )
