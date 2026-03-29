from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建 spec: {module_qualname} -> {file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_system_package(plugin_system_dir: str) -> str:
    package_name = f"_gtbot_plugin_system_unittestpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [plugin_system_dir]  # type: ignore[attr-defined]
    pkg.__file__ = str(Path(plugin_system_dir) / "__init__.py")
    pkg.__package__ = package_name
    sys.modules[package_name] = pkg

    _load_module_from_path(f"{package_name}.types", str(Path(plugin_system_dir) / "types.py"))
    _load_module_from_path(f"{package_name}.runtime", str(Path(plugin_system_dir) / "runtime.py"))
    _load_module_from_path(f"{package_name}.registry", str(Path(plugin_system_dir) / "registry.py"))
    _load_module_from_path(f"{package_name}.loader", str(Path(plugin_system_dir) / "loader.py"))
    _load_module_from_path(f"{package_name}.manager", str(Path(plugin_system_dir) / "manager.py"))
    init_path = Path(plugin_system_dir) / "__init__.py"
    init_code = init_path.read_text(encoding="utf-8")
    exec(compile(init_code, str(init_path), "exec"), pkg.__dict__)

    return package_name


class _FakeTool:
    name = "fake_tool"
    description = "fake"

    def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return "ok"


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestPluginSystemUnit(unittest.TestCase):
    def test_load_register_and_isolation(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        mod = __import__(pkg, fromlist=["PluginManager", "PluginContext"])
        PluginManager = getattr(mod, "PluginManager")
        PluginContext = getattr(mod, "PluginContext")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / f"pluginpkg_{uuid4().hex}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            _write_text(pkg_dir / "a.py", """
from __future__ import annotations

class ToolA:
    name = "tool_a"
    description = "a"
    def invoke(self, *args, **kwargs):
        return "a"


def register(registry):
    registry.add_tool(ToolA())
""")

            _write_text(pkg_dir / "b.py", """
from __future__ import annotations

def register(registry):
    raise RuntimeError("boom")
""")

            mgr = PluginManager(plugin_dir=pkg_dir)
            mgr.load()
            bundle = mgr.build(PluginContext(raw_messages=[]))

            names = [getattr(t, "name", None) for t in bundle.tools]
            self.assertIn("tool_a", names)

    def test_package_plugin_is_loaded(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        mod = __import__(pkg, fromlist=["PluginManager", "PluginContext"])
        PluginManager = getattr(mod, "PluginManager")
        PluginContext = getattr(mod, "PluginContext")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / f"pluginpkg_{uuid4().hex}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            plugin_pkg = pkg_dir / "pkg_plugin"
            plugin_pkg.mkdir(parents=True, exist_ok=True)
            _write_text(plugin_pkg / "__init__.py", """
from __future__ import annotations

class ToolPkg:
    name = "tool_pkg"
    description = "pkg"
    def invoke(self, *args, **kwargs):
        return "pkg"


def register(registry):
    registry.add_tool(ToolPkg())
""")

            mgr = PluginManager(plugin_dir=pkg_dir)
            mgr.load()
            bundle = mgr.build(PluginContext(raw_messages=[]))

            names = [getattr(t, "name", None) for t in bundle.tools]
            self.assertIn("tool_pkg", names)

    def test_enabled_predicate(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        mod = __import__(pkg, fromlist=["PluginManager", "PluginContext"])
        PluginManager = getattr(mod, "PluginManager")
        PluginContext = getattr(mod, "PluginContext")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / f"pluginpkg_{uuid4().hex}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            _write_text(pkg_dir / "a.py", """
from __future__ import annotations

class ToolA:
    name = "tool_a"
    description = "a"
    def invoke(self, *args, **kwargs):
        return "a"


def register(registry):
    registry.add_tool(ToolA(), enabled=lambda ctx: False)
""")

            mgr = PluginManager(plugin_dir=pkg_dir)
            mgr.load()
            bundle = mgr.build(PluginContext(raw_messages=[]))
            self.assertEqual(bundle.tools, [])

    def test_enabled_predicate_can_read_trigger_mode(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        mod = __import__(pkg, fromlist=["PluginManager", "PluginContext"])
        PluginManager = getattr(mod, "PluginManager")
        PluginContext = getattr(mod, "PluginContext")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / f"pluginpkg_{uuid4().hex}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            _write_text(pkg_dir / "a.py", """
from __future__ import annotations

class ToolA:
    name = "tool_private_only"
    description = "private only"
    def invoke(self, *args, **kwargs):
        return "a"


def register(registry):
    registry.add_tool(ToolA(), enabled=lambda ctx: ctx.trigger_mode == "private")
""")

            mgr = PluginManager(plugin_dir=pkg_dir)
            mgr.load()

            private_bundle = mgr.build(PluginContext(raw_messages=[], trigger_mode="private"))
            self.assertEqual([getattr(t, "name", None) for t in private_bundle.tools], ["tool_private_only"])

            group_bundle = mgr.build(PluginContext(raw_messages=[], trigger_mode="group_at"))
            self.assertEqual(group_bundle.tools, [])

    def test_tool_factory(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        mod = __import__(pkg, fromlist=["PluginManager", "PluginContext"])
        PluginManager = getattr(mod, "PluginManager")
        PluginContext = getattr(mod, "PluginContext")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / f"pluginpkg_{uuid4().hex}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            _write_text(pkg_dir / "a.py", """
from __future__ import annotations

class ToolA:
    name = "tool_a"
    description = "a"
    def invoke(self, *args, **kwargs):
        return "a"


def register(registry):
    registry.add_tool_factory(lambda ctx: [ToolA()])
""")

            mgr = PluginManager(plugin_dir=pkg_dir)
            mgr.load()
            bundle = mgr.build(PluginContext(raw_messages=[]))

            names = [getattr(t, "name", None) for t in bundle.tools]
            self.assertEqual(names, ["tool_a"])

    def test_contextvar_scope(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        runtime_mod = __import__(
            f"{pkg}.runtime",
            fromlist=["get_current_plugin_context", "plugin_context_scope", "set_response_status"],
        )
        types_mod = __import__(f"{pkg}.types", fromlist=["PluginContext"])
        get_current_plugin_context = getattr(runtime_mod, "get_current_plugin_context")
        plugin_context_scope = getattr(runtime_mod, "plugin_context_scope")
        set_response_status = getattr(runtime_mod, "set_response_status")
        PluginContext = getattr(types_mod, "PluginContext")

        self.assertIsNone(get_current_plugin_context())

        ctx = PluginContext(
            raw_messages=[{"a": 1}],
            trigger_mode="group_at",
            trigger_meta={"reason": "scheduled"},
        )
        with plugin_context_scope(ctx):
            got = get_current_plugin_context()
            self.assertIsNotNone(got)
            assert got is not None
            self.assertEqual(got.raw_messages, ctx.raw_messages)
            self.assertEqual(got.trigger_mode, "group_at")
            self.assertEqual(got.trigger_meta, {"reason": "scheduled"})
            self.assertEqual(got.response_status, "initialized")

        self.assertIsNone(get_current_plugin_context())

        runtime_context = SimpleNamespace(response_status="initialized")
        ctx = PluginContext(
            raw_messages=[],
            response_id="resp_123",
            response_status="initialized",
            runtime_context=runtime_context,
        )
        set_response_status(ctx, "agent_running")
        self.assertEqual(ctx.response_status, "agent_running")
        self.assertEqual(runtime_context.response_status, "agent_running")

    def test_plugin_context_response_fields(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        types_mod = __import__(f"{pkg}.types", fromlist=["PluginContext"])
        PluginContext = getattr(types_mod, "PluginContext")

        ctx = PluginContext(raw_messages=[])
        self.assertEqual(ctx.response_id, "")
        self.assertEqual(ctx.response_status, "initialized")

        ctx2 = PluginContext(raw_messages=[], response_id="rid_1", response_status="collecting_prerequisites")
        self.assertEqual(ctx2.response_id, "rid_1")
        self.assertEqual(ctx2.response_status, "collecting_prerequisites")

    def test_pre_agent_processor_registration(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        mod = __import__(pkg, fromlist=["PluginManager", "PluginContext"])
        PluginManager = getattr(mod, "PluginManager")
        PluginContext = getattr(mod, "PluginContext")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / f"pluginpkg_{uuid4().hex}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            _write_text(pkg_dir / "a.py", """
from __future__ import annotations

async def proc_first(ctx):
    ctx.extra["first"] = True


async def proc_second(ctx):
    ctx.extra["second"] = True


def register(registry):
    registry.add_pre_agent_processor(proc_second, priority=5, wait_until_complete=False)
    registry.add_pre_agent_processor(
        proc_first,
        priority=-1,
        wait_until_complete=True,
        enabled=lambda ctx: ctx.trigger_mode == "group_at",
    )
""")

            mgr = PluginManager(plugin_dir=pkg_dir)
            mgr.load()

            enabled_bundle = mgr.build(PluginContext(raw_messages=[], trigger_mode="group_at"))
            self.assertEqual(len(enabled_bundle.pre_agent_processors), 2)
            self.assertTrue(enabled_bundle.pre_agent_processors[0].wait_until_complete)
            self.assertFalse(enabled_bundle.pre_agent_processors[1].wait_until_complete)

            disabled_bundle = mgr.build(PluginContext(raw_messages=[], trigger_mode="group_auto"))
            self.assertEqual(len(disabled_bundle.pre_agent_processors), 1)
            self.assertFalse(disabled_bundle.pre_agent_processors[0].wait_until_complete)

    def test_module_without_register_is_ignored(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        mod = __import__(pkg, fromlist=["PluginManager", "PluginContext"])
        PluginManager = getattr(mod, "PluginManager")
        PluginContext = getattr(mod, "PluginContext")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / f"pluginpkg_{uuid4().hex}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            _write_text(pkg_dir / "a.py", """
from __future__ import annotations

class SomeTool:
    name = "should_not_load"
    description = "no register"
    def invoke(self, *args, **kwargs):
        return "ok"
""")

            mgr = PluginManager(plugin_dir=pkg_dir)
            mgr.load()
            bundle = mgr.build(PluginContext(raw_messages=[]))
            self.assertEqual(bundle.tools, [])


if __name__ == "__main__":
    unittest.main()
