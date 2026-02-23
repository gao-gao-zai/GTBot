from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from uuid import uuid4


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    import sys

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建 spec: {module_qualname} -> {file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_system_package(plugin_system_dir: str) -> str:
    import sys

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

    def test_reload(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        mod = __import__(pkg, fromlist=["PluginManager", "PluginContext"])
        PluginManager = getattr(mod, "PluginManager")
        PluginContext = getattr(mod, "PluginContext")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / f"pluginpkg_{uuid4().hex}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            mod_file = pkg_dir / "a.py"
            _write_text(mod_file, """
from __future__ import annotations

class ToolA:
    name = "tool_a_v1"
    description = "a"
    def invoke(self, *args, **kwargs):
        return "a"


def register(registry):
    registry.add_tool(ToolA())
""")

            mgr = PluginManager(plugin_dir=pkg_dir)
            mgr.load()
            bundle1 = mgr.build(PluginContext(raw_messages=[]))
            names1 = [getattr(t, "name", None) for t in bundle1.tools]
            self.assertIn("tool_a_v1", names1)

            _write_text(mod_file, """
from __future__ import annotations

class ToolA:
    name = "tool_a_v2"
    description = "a"
    def invoke(self, *args, **kwargs):
        return "a"


def register(registry):
    registry.add_tool(ToolA())
""")

            mgr.reload()
            bundle2 = mgr.build(PluginContext(raw_messages=[]))
            names2 = [getattr(t, "name", None) for t in bundle2.tools]
            self.assertIn("tool_a_v2", names2)

    def test_contextvar_scope(self) -> None:
        plugin_system_dir = str(Path(__file__).resolve().parents[1])
        pkg = _load_plugin_system_package(plugin_system_dir)
        runtime_mod = __import__(f"{pkg}.runtime", fromlist=["get_current_plugin_context", "plugin_context_scope"])
        types_mod = __import__(f"{pkg}.types", fromlist=["PluginContext"])
        get_current_plugin_context = getattr(runtime_mod, "get_current_plugin_context")
        plugin_context_scope = getattr(runtime_mod, "plugin_context_scope")
        PluginContext = getattr(types_mod, "PluginContext")

        self.assertIsNone(get_current_plugin_context())

        ctx = PluginContext(raw_messages=[{"a": 1}])
        with plugin_context_scope(ctx):
            got = get_current_plugin_context()
            self.assertIsNotNone(got)
            assert got is not None
            self.assertEqual(got.raw_messages, ctx.raw_messages)

        self.assertIsNone(get_current_plugin_context())

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
