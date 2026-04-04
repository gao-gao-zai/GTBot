from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块并注册到 `sys.modules`。

    Args:
        module_qualname: 目标模块完整名称。
        file_path: 目标文件路径。

    Returns:
        已执行完成的模块对象。
    """

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建 spec: {module_qualname} -> {file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_tool_import_stubs() -> None:
    """为工具模块安装最小依赖桩，避免测试环境缺少完整运行时。"""

    if "langchain.tools" not in sys.modules:
        langchain_mod = sys.modules.setdefault("langchain", ModuleType("langchain"))
        tools_mod = ModuleType("langchain.tools")

        class ToolRuntime:  # noqa: D401
            """测试桩版 ToolRuntime。"""

            def __init__(self, context=None) -> None:
                self.context = context

            def __class_getitem__(cls, _item):
                return cls

        def tool(_name: str):
            def decorator(func):
                return func

            return decorator

        setattr(tools_mod, "ToolRuntime", ToolRuntime)
        setattr(tools_mod, "tool", tool)
        sys.modules["langchain.tools"] = tools_mod
        setattr(langchain_mod, "tools", tools_mod)

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    group_ctx_mod = ModuleType("plugins.GTBot.services.chat.context")

    class GroupChatContext:  # noqa: D401
        """测试桩版 GroupChatContext。"""

    setattr(group_ctx_mod, "GroupChatContext", GroupChatContext)
    sys.modules["plugins.GTBot.services.chat.context"] = group_ctx_mod
    setattr(plugins_mod, "GTBot", gtbot_mod)


def _load_anythingllm_docs_package(plugin_dir: str) -> str:
    """加载 AnythingLLM 文档插件包，但不经过 `plugins.GTBot` 顶层导入链。

    Args:
        plugin_dir: 插件目录路径。

    Returns:
        动态构造的测试包名。
    """

    _install_tool_import_stubs()

    package_name = f"_anythingllm_docs_unittestpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [plugin_dir]  # type: ignore[attr-defined]
    pkg.__file__ = str(Path(plugin_dir) / "__init__.py")
    pkg.__package__ = package_name
    sys.modules[package_name] = pkg

    _load_module_from_path(f"{package_name}.config", str(Path(plugin_dir) / "config.py"))
    _load_module_from_path(f"{package_name}.store", str(Path(plugin_dir) / "store.py"))
    _load_module_from_path(f"{package_name}.client", str(Path(plugin_dir) / "client.py"))
    _load_module_from_path(f"{package_name}.tool", str(Path(plugin_dir) / "tool.py"))
    return package_name


class TestAnythingLLMDocsConfig(unittest.TestCase):
    """验证配置文件读取与回退行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_anythingllm_docs_package(plugin_dir)
        cls.config_mod = __import__(f"{cls.pkg}.config", fromlist=["dummy"])

    def test_load_config_from_plugin_root_json(self) -> None:
        """应能从插件根目录 config.json 正常读取配置。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            example_path = root / "config.json.example"
            config_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "anythingllm": {
                            "base_url": "http://localhost:3001",
                            "api_key": "test-key",
                            "workspace_name": "docs",
                            "workspace_slug": "docs",
                        },
                        "storage": {
                            "temp_dir": "./data/temp",
                            "store_file": "./data/documents.json",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.object(self.config_mod, "_config_path", return_value=config_path), patch.object(
                self.config_mod, "_example_path", return_value=example_path
            ):
                self.config_mod.reload_anythingllm_docs_plugin_config()
                cfg = self.config_mod.get_anythingllm_docs_plugin_config()
                self.assertTrue(cfg.enabled)
                self.assertEqual(cfg.anythingllm.api_key, "test-key")
                self.assertEqual(cfg.anythingllm.workspace_slug, "docs")

    def test_invalid_config_should_fallback_to_defaults(self) -> None:
        """非法配置应回退到默认值并重写配置文件。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            example_path = root / "config.json.example"
            config_path.write_text("[]", encoding="utf-8")

            with patch.object(self.config_mod, "_config_path", return_value=config_path), patch.object(
                self.config_mod, "_example_path", return_value=example_path
            ):
                self.config_mod.reload_anythingllm_docs_plugin_config()
                cfg = self.config_mod.get_anythingllm_docs_plugin_config()
                self.assertEqual(cfg.anythingllm.workspace_slug, "gtbot-global-docs")
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertIsInstance(parsed, dict)


class TestAnythingLLMDocumentStore(unittest.IsolatedAsyncioTestCase):
    """验证本地状态文件的增删查行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_anythingllm_docs_package(plugin_dir)
        cls.store_mod = __import__(f"{cls.pkg}.store", fromlist=["dummy"])
        cls.client_mod = __import__(f"{cls.pkg}.client", fromlist=["dummy"])

    async def test_store_add_list_remove(self) -> None:
        """应能新增、列出并删除文档记录。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            store = self.store_mod.AnythingLLMDocumentStore(Path(temp_dir) / "documents.json")
            created = await store.add_document(
                title="测试文档",
                file_name="test.pdf",
                location="custom-documents/test.pdf-1.json",
                document_name="test.pdf-1.json",
                workspace_slug="gtbot-global-docs",
                uploaded_by=123456,
                word_count=100,
            )
            documents = await store.list_documents()
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].record_id, created.record_id)

            hit = await store.find_by_record_id(created.record_id)
            self.assertIsNotNone(hit)
            fuzzy = await store.search_by_keyword("测试")
            self.assertEqual(len(fuzzy), 1)

            removed = await store.remove_by_record_id(created.record_id)
            self.assertIsNotNone(removed)
            self.assertEqual((await store.list_documents()), [])

    async def test_store_should_not_overwrite_previous_records_when_document_name_is_empty(self) -> None:
        """当 AnythingLLM 未返回文档名时，不应误删此前记录。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            store = self.store_mod.AnythingLLMDocumentStore(Path(temp_dir) / "documents.json")
            await store.add_document(
                title="文档一",
                file_name="one.md",
                location="custom-documents/one.json",
                document_name="",
                workspace_slug="gtbot-global-docs",
                uploaded_by=123456,
            )
            await store.add_document(
                title="文档二",
                file_name="two.md",
                location="custom-documents/two.json",
                document_name="",
                workspace_slug="gtbot-global-docs",
                uploaded_by=123456,
            )
            documents = await store.list_documents()
            self.assertEqual(len(documents), 2)

    async def test_store_sync_should_follow_api_as_source_of_truth(self) -> None:
        """API 有则补本地，API 无则删除本地。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            store = self.store_mod.AnythingLLMDocumentStore(Path(temp_dir) / "documents.json")
            existing = await store.add_document(
                title="旧文档",
                file_name="old.md",
                location="custom-documents/old.json",
                document_name="old.json",
                workspace_slug="gtbot-global-docs",
                uploaded_by=123456,
            )
            api_documents = [
                self.client_mod.AnythingLLMWorkspaceDocument(
                    location="custom-documents/new.json",
                    document_name="new.json",
                    title="新文档",
                    source_url="file:///new.md",
                    created_at="2026-03-28T12:00:00Z",
                )
            ]
            documents = await store.sync_with_workspace_documents(
                workspace_slug="gtbot-global-docs",
                api_documents=api_documents,
            )
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].title, "新文档")
            self.assertNotEqual(documents[0].record_id, existing.record_id)
            self.assertEqual(documents[0].uploaded_by, 0)


class TestAnythingLLMClient(unittest.IsolatedAsyncioTestCase):
    """验证 AnythingLLM 客户端请求拼装与流式解析。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_anythingllm_docs_package(plugin_dir)
        cls.client_mod = __import__(f"{cls.pkg}.client", fromlist=["dummy"])
        cls.config_mod = __import__(f"{cls.pkg}.config", fromlist=["dummy"])

    async def test_parse_sse_query_workspace(self) -> None:
        """应能解析 stream-chat SSE 响应并提取答案与来源。"""

        cfg = self.config_mod.AnythingLLMDocsPluginConfig.model_validate(
            {
                "enabled": True,
                "anythingllm": {
                    "base_url": "http://example.test",
                    "api_key": "token",
                    "workspace_name": "docs",
                    "workspace_slug": "docs",
                    "chat": {
                        "mode": "query",
                        "top_n": 4,
                        "similarity_threshold": 0.5,
                        "session_prefix": "prefix",
                        "reset": True,
                    },
                },
                "storage": {
                    "temp_dir": "./data/temp",
                    "store_file": "./data/documents.json",
                },
            }
        )
        client = self.client_mod.AnythingLLMClient(cfg)

        body_lines = [
            'data: {"id":"1","type":"textResponseChunk","textResponse":"第一段","sources":[],"close":false,"error":null}\n',
            'data: {"id":"1","type":"textResponseChunk","textResponse":"第二段","sources":[{"title":"文档A","chunk":"命中片段"}],"close":true,"error":null}\n',
        ]

        async def fake_stream_events(method: str, path: str, *, json_body: dict) -> list[dict]:
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/api/v1/workspace/docs/stream-chat")
            self.assertEqual(json_body["mode"], "query")
            parsed: list[dict] = []
            for line in body_lines:
                item = client._parse_sse_line(line)
                if item is not None:
                    parsed.append(item)
            return parsed

        with patch.object(client, "_stream_events", side_effect=fake_stream_events):
            result = await client.query_workspace(
                workspace_slug="docs",
                question="这份文档讲了什么？",
                session_id="session-1",
            )
        self.assertIsInstance(result, self.client_mod.AnythingLLMChatResult)
        self.assertEqual(result.answer, "第一段第二段")
        self.assertEqual(result.sources[0]["title"], "文档A")

    async def test_ensure_workspace_should_fail_without_api_key(self) -> None:
        """未配置 API Key 时应直接报错。"""

        cfg = self.config_mod.AnythingLLMDocsPluginConfig()
        client = self.client_mod.AnythingLLMClient(cfg)
        with self.assertRaises(self.client_mod.AnythingLLMClientError):
            await client.ensure_workspace()


class TestAnythingLLMTools(unittest.IsolatedAsyncioTestCase):
    """验证文档列表工具与全库查询工具行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_anythingllm_docs_package(plugin_dir)
        cls.store_mod = __import__(f"{cls.pkg}.store", fromlist=["dummy"])
        cls.tool_mod = __import__(f"{cls.pkg}.tool", fromlist=["dummy"])

    def _record(self, record_id: str, title: str, file_name: str) -> object:
        return self.store_mod.StoredDocumentRecord(
            record_id=record_id,
            title=title,
            file_name=file_name,
            location=f"custom-documents/{file_name}.json",
            document_name=f"{file_name}.json",
            workspace_slug="gtbot-global-docs",
            uploaded_by=123456,
            uploaded_at="2026-03-27T12:00:00+00:00",
        )

    def _runtime(self) -> object:
        context = SimpleNamespace(chat_type="group", group_id=10001, user_id=20002)
        return SimpleNamespace(context=context)

    async def test_list_documents_tool_should_show_records(self) -> None:
        """列表工具应返回可查询文档清单。"""

        fake_client = SimpleNamespace(
            ensure_workspace=AsyncMock(return_value=SimpleNamespace(slug="gtbot-global-docs", name="GTBot Docs")),
            list_workspace_documents=AsyncMock(return_value=[]),
        )
        fake_store = SimpleNamespace(
            sync_with_workspace_documents=AsyncMock(
                return_value=[
                    self._record("ab12cd34", "产品手册", "product.pdf"),
                    self._record("ef56gh78", "售后规则", "after-sale.docx"),
                ]
            )
        )

        with patch.object(self.tool_mod, "get_anythingllm_client", return_value=fake_client), patch.object(
            self.tool_mod, "get_anythingllm_document_store", return_value=fake_store
        ):
            result = await self.tool_mod.list_anythingllm_documents_impl(limit=10)

        self.assertIn("当前可查询文档列表", result)
        self.assertIn("[ab12cd34] 产品手册", result)
        self.assertIn("[ef56gh78] 售后规则", result)

    async def test_query_documents_tool_should_query_workspace(self) -> None:
        """查询工具应直接对整个工作区发起问答。"""

        fake_client = SimpleNamespace(
            ensure_workspace=AsyncMock(return_value=SimpleNamespace(slug="gtbot-global-docs", name="GTBot Docs")),
            query_workspace=AsyncMock(
                return_value=SimpleNamespace(
                    answer="退货时间为 7 天。",
                    sources=[{"title": "产品手册", "chunk": "7 天内可以申请退货。"}],
                )
            ),
        )
        fake_cfg = SimpleNamespace(
            enabled=True,
            anythingllm=SimpleNamespace(chat=SimpleNamespace(session_prefix="gtbot-anythingllm-docs")),
        )

        with patch.object(self.tool_mod, "get_anythingllm_client", return_value=fake_client), patch.object(
            self.tool_mod, "get_anythingllm_docs_plugin_config", return_value=fake_cfg
        ):
            result = await self.tool_mod.query_anythingllm_documents_impl(
                question="退货规则是什么？",
                runtime=self._runtime(),
            )

        called_question = fake_client.query_workspace.await_args.kwargs["question"]
        self.assertEqual(called_question, "退货规则是什么？")
        self.assertIn("[scope]", result)
        self.assertIn("全局文档库", result)
        self.assertIn("退货时间为 7 天。", result)
        self.assertIn("产品手册", result)
        self.assertNotIn("7 天内可以申请退货。", result)


if __name__ == "__main__":
    unittest.main()
