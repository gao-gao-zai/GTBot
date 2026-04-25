from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)

from .loader import PluginLoader
from .registry import PluginRegistry
from .types import (
    MessageAppendPosition,
    PluginBundle,
    PluginContext,
    PreAgentMessageAppenderBinding,
    PreAgentMessageInjectorBinding,
    PreAgentProcessorBinding,
)


@dataclass
class PluginManager:
    plugin_dir: str | Path

    def __post_init__(self) -> None:
        self._loader = PluginLoader(self.plugin_dir)
        self._registry = PluginRegistry()
        self._loaded = False

    def load(self) -> None:
        """加载插件目录。"""

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

    def build(self, ctx: PluginContext) -> PluginBundle:
        """基于当前插件注册信息构建单次请求的插件产物集合。"""

        if not self._loaded:
            self.load()

        tools: list[Any] = []
        middlewares: list[Any] = []
        callbacks: list[Any] = []
        pre_agent_processors: list[PreAgentProcessorBinding] = []
        pre_agent_message_injectors: list[PreAgentMessageInjectorBinding] = []
        pre_agent_message_appenders: list[PreAgentMessageAppenderBinding] = []

        for tool_item in sorted(self._registry.iter_tools(), key=lambda x: x.priority):
            if not self._is_enabled(item=tool_item, ctx=ctx, label="tool"):
                continue
            tools.append(tool_item.tool)

        for factory_item in sorted(self._registry.iter_tool_factories(), key=lambda x: x.priority):
            if not self._is_enabled(item=factory_item, ctx=ctx, label="tool_factory"):
                continue

            try:
                produced = factory_item.factory(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"tool_factory 执行失败，跳过: {exc}")
                continue

            if produced:
                tools.extend(list(produced))

        for middleware_item in sorted(self._registry.iter_middlewares(), key=lambda x: x.priority):
            if not self._is_enabled(item=middleware_item, ctx=ctx, label="middleware"):
                continue
            middlewares.append(middleware_item.middleware)

        for callback_item in sorted(self._registry.iter_callbacks(), key=lambda x: x.priority):
            if not self._is_enabled(item=callback_item, ctx=ctx, label="callback"):
                continue
            callbacks.append(callback_item.callback)

        for processor_item in sorted(self._registry.iter_pre_agent_processors(), key=lambda x: x.priority):
            if not self._is_enabled(item=processor_item, ctx=ctx, label="pre_agent_processor"):
                continue
            pre_agent_processors.append(
                PreAgentProcessorBinding(
                    processor=processor_item.processor,
                    wait_until_complete=bool(processor_item.wait_until_complete),
                )
            )

        for injector_item in sorted(self._registry.iter_pre_agent_message_injectors(), key=lambda x: x.priority):
            if not self._is_enabled(item=injector_item, ctx=ctx, label="pre_agent_message_injector"):
                continue
            pre_agent_message_injectors.append(
                PreAgentMessageInjectorBinding(injector=injector_item.injector)
            )

        for appender_item in sorted(self._registry.iter_pre_agent_message_appenders(), key=lambda x: x.priority):
            if not self._is_enabled(item=appender_item, ctx=ctx, label="pre_agent_message_appender"):
                continue
            pre_agent_message_appenders.append(
                PreAgentMessageAppenderBinding(
                    appender=appender_item.appender,
                    position=self._normalize_position(appender_item.position),
                )
            )

        return PluginBundle(
            tools=tools,
            agent_middlewares=middlewares,
            callbacks=callbacks,
            pre_agent_processors=pre_agent_processors,
            pre_agent_message_injectors=pre_agent_message_injectors,
            pre_agent_message_appenders=pre_agent_message_appenders,
        )

    @staticmethod
    def _normalize_position(position: str) -> MessageAppendPosition:
        """校验并归一化消息追加位置。"""

        if position not in {"prepend", "append"}:
            raise ValueError(f"不支持的消息追加位置: {position}")
        return cast(MessageAppendPosition, position)

    @staticmethod
    def _is_enabled(*, item: Any, ctx: PluginContext, label: str) -> bool:
        """统一执行 enabled 判定并隔离异常。"""

        if item.enabled is None:
            return True

        try:
            return bool(item.enabled(ctx))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{label} enabled 判断失败，跳过: {exc}")
            return False
