from __future__ import annotations

import importlib.util
import tempfile
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块并注册到 `sys.modules`。"""

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_qualname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


def _install_file_registry_import_stubs(data_root: Path) -> None:
    """为文件注册表单测安装最小导入桩。"""

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    setattr(plugins_mod, "GTBot", gtbot_mod)

    config_manager_mod = sys.modules.setdefault("plugins.GTBot.ConfigManager", ModuleType("plugins.GTBot.ConfigManager"))
    setattr(config_manager_mod, "total_config", SimpleNamespace(get_data_dir_path=lambda: data_root))

    services_mod = sys.modules.setdefault("plugins.GTBot.services", ModuleType("plugins.GTBot.services"))
    file_registry_pkg = sys.modules.setdefault(
        "plugins.GTBot.services.file_registry", ModuleType("plugins.GTBot.services.file_registry")
    )
    setattr(services_mod, "file_registry", file_registry_pkg)


def _load_file_registry_package(package_dir: str) -> tuple[ModuleType, ModuleType, ModuleType]:
    """加载文件注册表包而不触发宿主其他模块导入。"""

    package_name = "plugins.GTBot.services.file_registry"
    pkg = sys.modules[package_name]
    pkg.__path__ = [package_dir]  # type: ignore[attr-defined]
    pkg.__file__ = str(Path(package_dir) / "__init__.py")
    pkg.__package__ = package_name

    models_mod = _load_module_from_path(f"{package_name}.models", str(Path(package_dir) / "models.py"))
    store_mod = _load_module_from_path(f"{package_name}.store", str(Path(package_dir) / "store.py"))
    service_mod = _load_module_from_path(f"{package_name}.service", str(Path(package_dir) / "service.py"))
    _load_module_from_path(package_name, str(Path(package_dir) / "__init__.py"))
    return models_mod, store_mod, service_mod


class TestFileRegistryService(unittest.TestCase):
    store_mod: ModuleType
    service_mod: ModuleType

    """验证统一文件映射系统的核心行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        data_root = Path(tempfile.mkdtemp(prefix="gtbot_file_registry_test_"))
        cls.temp_dir = data_root
        _install_file_registry_import_stubs(data_root)
        package_dir = str(Path(__file__).resolve().parents[1])
        _, cls.store_mod, cls.service_mod = _load_file_registry_package(package_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_register_local_file_should_return_file_id_and_resolve_metadata(self) -> None:
        """注册现有本地文件后应能通过 `file_id` 回查完整句柄。"""

        temp_root = self.temp_dir
        image_path = temp_root / f"{uuid4().hex}.png"
        image_path.write_bytes(b"hello-world")

        store = self.store_mod.ManagedFileStore(temp_root / "registry.sqlite3")
        service = self.service_mod.ManagedFileService(store=store)
        file_id = service.register_local_file(
            image_path,
            kind="avatar",
            source_type="avatar_download",
            session_id="group:1",
            group_id=1,
            user_id=2,
            mime_type="image/png",
            extra={"avatar_type": "user"},
        )

        self.assertTrue(file_id.startswith("gtfile:"))
        handle = service.resolve_file(file_id)
        self.assertEqual(handle.local_path, image_path.resolve())
        self.assertEqual(handle.mime_type, "image/png")
        self.assertEqual(handle.extra["avatar_type"], "user")

    def test_register_bytes_should_persist_file_and_be_recoverable_after_restart(self) -> None:
        """通过字节注册的文件应落盘、持久化并可被新服务实例解析。"""

        temp_root = self.temp_dir
        db_path = temp_root / "registry_restart.sqlite3"
        store = self.store_mod.ManagedFileStore(db_path)
        service = self.service_mod.ManagedFileService(store=store)

        handle = service.register_bytes(
            b"png-bytes",
            suffix=".png",
            kind="draw_result",
            source_type="openai_draw",
            original_name="result.png",
            extra={"job_id": "job-1"},
        )
        self.assertTrue(handle.file_id.startswith("gtfile:"))
        self.assertTrue(handle.local_path.exists())

        restarted_store = self.store_mod.ManagedFileStore(db_path)
        restarted_service = self.service_mod.ManagedFileService(store=restarted_store)
        resolved = restarted_service.resolve_file(handle.file_id)
        self.assertEqual(resolved.local_path.read_bytes(), b"png-bytes")
        self.assertEqual(resolved.extra["job_id"], "job-1")


if __name__ == "__main__":
    unittest.main()
