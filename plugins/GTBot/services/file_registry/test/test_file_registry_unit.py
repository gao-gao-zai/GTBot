from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块并注册到 `sys.modules`。

    Args:
        module_qualname: 目标模块全名。
        file_path: 模块对应的物理文件路径。

    Returns:
        已执行并注册完成的模块对象。
    """

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_qualname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


def _install_file_registry_import_stubs(data_root: Path) -> None:
    """为文件映射服务单测安装最小导入桩。

    Args:
        data_root: 本次测试使用的数据目录根路径。
    """

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    setattr(plugins_mod, "GTBot", gtbot_mod)

    config_manager_mod = sys.modules.setdefault("plugins.GTBot.ConfigManager", ModuleType("plugins.GTBot.ConfigManager"))
    setattr(config_manager_mod, "total_config", SimpleNamespace(get_data_dir_path=lambda: data_root))

    services_mod = sys.modules.setdefault("plugins.GTBot.services", ModuleType("plugins.GTBot.services"))
    file_registry_pkg = sys.modules.setdefault(
        "plugins.GTBot.services.file_registry",
        ModuleType("plugins.GTBot.services.file_registry"),
    )
    setattr(services_mod, "file_registry", file_registry_pkg)


def _load_file_registry_package(package_dir: str) -> tuple[ModuleType, ModuleType, ModuleType]:
    """加载文件映射服务包而不触发 GTBot 其他模块导入。

    Args:
        package_dir: `file_registry` 包目录。

    Returns:
        `models.py`、`store.py` 和 `service.py` 模块对象。
    """

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
    """验证 GTFile 映射系统的核心行为。"""

    store_mod: ModuleType
    service_mod: ModuleType
    temp_dir: ClassVar[Path]

    @classmethod
    def setUpClass(cls) -> None:
        """初始化文件映射单测所需的隔离环境。"""

        data_root = Path(tempfile.mkdtemp(prefix="gtbot_file_registry_test_"))
        cls.temp_dir = data_root
        _install_file_registry_import_stubs(data_root)
        package_dir = str(Path(__file__).resolve().parents[1])
        _, cls.store_mod, cls.service_mod = _load_file_registry_package(package_dir)

    def test_register_local_file_should_return_gfid_and_resolve_display_name(self) -> None:
        """注册本地文件后应能通过 `gfid` 与可选 `gf` 双向解析。"""

        temp_root = self.temp_dir
        image_path = temp_root / "sample.png"
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
            display_name="gf:avatar:user:sample.png",
            extra={"avatar_type": "user"},
            expires_at=time.time() + 60.0,
        )

        self.assertTrue(file_id.startswith("gfid:"))
        handle_by_id = service.resolve_file(file_id)
        handle_by_name = service.resolve_file_ref("gf:avatar:user:sample.png")
        self.assertEqual(handle_by_id.local_path, image_path.resolve())
        self.assertEqual(handle_by_name.file_id, file_id)
        self.assertEqual(handle_by_name.display_name, "gf:avatar:user:sample.png")

    def test_register_bytes_should_reject_storage_responsibility(self) -> None:
        """GTFile 不应承担字节落盘职责，而应要求调用方先自行落盘。"""

        temp_root = self.temp_dir
        store = self.store_mod.ManagedFileStore(temp_root / "registry_restart.sqlite3")
        service = self.service_mod.ManagedFileService(store=store)

        with self.assertRaisesRegex(NotImplementedError, "GTFile 只负责文件映射"):
            service.register_bytes(
                b"png-bytes",
                suffix=".png",
                kind="draw_result",
                source_type="openai_draw",
                original_name="result.png",
                extra={"job_id": "job-1"},
                expires_at=time.time() + 60.0,
            )

    def test_register_local_file_should_reject_infinite_mapping_without_plugin_cleanup(self) -> None:
        """永久映射缺少插件级清理责任时应显式失败。"""

        temp_root = self.temp_dir
        image_path = temp_root / "permanent.png"
        image_path.write_bytes(b"hello-world")

        store = self.store_mod.ManagedFileStore(temp_root / "registry_invalid.sqlite3")
        service = self.service_mod.ManagedFileService(store=store)
        with self.assertRaisesRegex(ValueError, "永久映射必须声明"):
            service.register_local_file(
                image_path,
                kind="meme",
                source_type="meme_store",
                cleanup_policy="delete_file_with_mapping",
            )

    def test_resolve_file_should_tolerate_accidental_image_suffix_on_gfid(self) -> None:
        """解析 `gfid:` 时应兼容误拼接在末尾的常见图片扩展名。"""

        temp_root = self.temp_dir
        image_path = temp_root / "suffix.png"
        image_path.write_bytes(b"suffix-test")

        store = self.store_mod.ManagedFileStore(temp_root / "registry_suffix.sqlite3")
        service = self.service_mod.ManagedFileService(store=store)
        file_id = service.register_local_file(
            image_path,
            kind="chat_image",
            source_type="onebot_image",
            expires_at=time.time() + 60.0,
        )

        resolved = service.resolve_file_ref(f"{file_id}.jpg")
        self.assertEqual(resolved.file_id, file_id)
        self.assertEqual(resolved.local_path, image_path.resolve())

    def test_purge_expired_mappings_should_delete_mapping_and_file(self) -> None:
        """过期映射应能被显式清理，并按策略删除物理文件。"""

        temp_root = self.temp_dir
        image_path = temp_root / "expired.png"
        image_path.write_bytes(b"expired")

        store = self.store_mod.ManagedFileStore(temp_root / "registry_purge.sqlite3")
        service = self.service_mod.ManagedFileService(store=store)
        file_id = service.register_local_file(
            image_path,
            kind="chat_image",
            source_type="onebot_image",
            expires_at=time.time() - 10.0,
        )

        result = service.purge_expired_mappings(now_ts=time.time())
        self.assertEqual(result.scanned_count, 1)
        self.assertEqual(result.deleted_mapping_count, 1)
        self.assertEqual(result.deleted_file_count, 1)
        self.assertFalse(image_path.exists())
        with self.assertRaises(FileNotFoundError):
            service.resolve_file(file_id)


if __name__ == "__main__":
    unittest.main()
