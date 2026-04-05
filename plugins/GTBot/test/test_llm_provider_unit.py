from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    _MODULE_PATH = ROOT / "plugins" / "GTBot" / "llm_provider.py"
    _SPEC = importlib.util.spec_from_file_location("gtbot_llm_provider_test", _MODULE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError(f"unable to load module spec: {_MODULE_PATH}")
    llm_provider = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(llm_provider)

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    llm_provider = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"runtime dependency missing: {_IMPORT_ERROR}")
class TestLLMProviderUnit(unittest.TestCase):
    def test_normalize_chat_provider_type_defaults_to_openai_compatible(self) -> None:
        assert llm_provider is not None
        self.assertEqual(llm_provider.normalize_chat_provider_type(""), "openai_compatible")
        self.assertEqual(llm_provider.normalize_chat_provider_type(None), "openai_compatible")

    def test_normalize_chat_provider_type_rejects_unknown_value(self) -> None:
        assert llm_provider is not None
        with self.assertRaises(ValueError):
            llm_provider.normalize_chat_provider_type("unknown")

    def test_build_chat_model_openai_compatible_uses_extra_body(self) -> None:
        assert llm_provider is not None
        with patch.object(llm_provider, "ChatOpenAI") as chat_openai_cls:
            sentinel = object()
            chat_openai_cls.return_value = sentinel

            model = llm_provider.build_chat_model(
                provider_type="openai_compatible",
                model_id="demo-model",
                base_url="https://example.test/v1",
                api_key="secret",
                streaming=True,
                model_parameters={"temperature": 0.7},
            )

            self.assertIs(model, sentinel)
            kwargs = chat_openai_cls.call_args.kwargs
            self.assertEqual(kwargs["model"], "demo-model")
            self.assertEqual(kwargs["base_url"], "https://example.test/v1")
            self.assertTrue(kwargs["streaming"])
            self.assertEqual(kwargs["extra_body"], {"temperature": 0.7})
            self.assertEqual(kwargs["api_key"].get_secret_value(), "secret")

    def test_build_chat_model_openai_responses_enables_responses_api(self) -> None:
        assert llm_provider is not None
        with patch.object(llm_provider, "ChatOpenAI") as chat_openai_cls:
            sentinel = object()
            chat_openai_cls.return_value = sentinel

            model = llm_provider.build_chat_model(
                provider_type="openai_responses",
                model_id="demo-model",
                base_url="https://example.test/v1",
                api_key="secret",
                streaming=False,
                model_parameters={"temperature": 0.2},
            )

            self.assertIs(model, sentinel)
            kwargs = chat_openai_cls.call_args.kwargs
            self.assertFalse(kwargs["streaming"])
            self.assertEqual(kwargs["extra_body"], {"temperature": 0.2})
            self.assertEqual(kwargs["model_kwargs"], {"use_responses_api": True})

    def test_build_chat_model_anthropic_uses_chat_anthropic(self) -> None:
        assert llm_provider is not None
        with patch.object(llm_provider.importlib, "import_module") as import_module_mock:
            anthropic_ctor = Mock(return_value="anthropic-model")
            import_module_mock.return_value = SimpleNamespace(ChatAnthropic=anthropic_ctor)

            model = llm_provider.build_chat_model(
                provider_type="anthropic",
                model_id="claude-3-7-sonnet",
                base_url="https://api.anthropic.com",
                api_key="secret",
                streaming=False,
                model_parameters={"temperature": 0.3},
            )

            self.assertEqual(model, "anthropic-model")
            import_module_mock.assert_called_once_with("langchain_anthropic")
            kwargs = anthropic_ctor.call_args.kwargs
            self.assertEqual(kwargs["model"], "claude-3-7-sonnet")
            self.assertEqual(kwargs["base_url"], "https://api.anthropic.com")
            self.assertEqual(kwargs["model_kwargs"], {"temperature": 0.3})

    def test_build_chat_model_gemini_uses_google_chat_model(self) -> None:
        assert llm_provider is not None
        with patch.object(llm_provider.importlib, "import_module") as import_module_mock:
            gemini_ctor = Mock(return_value="gemini-model")
            import_module_mock.return_value = SimpleNamespace(ChatGoogleGenerativeAI=gemini_ctor)

            model = llm_provider.build_chat_model(
                provider_type="gemini",
                model_id="gemini-2.5-flash",
                base_url="",
                api_key="secret",
                streaming=True,
                model_parameters={"temperature": 0.9},
            )

            self.assertEqual(model, "gemini-model")
            import_module_mock.assert_called_once_with("langchain_google_genai")
            kwargs = gemini_ctor.call_args.kwargs
            self.assertEqual(kwargs["model"], "gemini-2.5-flash")
            self.assertTrue(kwargs["streaming"])
            self.assertEqual(kwargs["model_kwargs"], {"temperature": 0.9})
            self.assertEqual(kwargs["google_api_key"].get_secret_value(), "secret")

    def test_build_chat_model_dashscope_uses_chat_tongyi(self) -> None:
        assert llm_provider is not None
        with patch.object(llm_provider.importlib, "import_module") as import_module_mock:
            dashscope_ctor = Mock(return_value="dashscope-model")
            import_module_mock.return_value = SimpleNamespace(ChatTongyi=dashscope_ctor)

            model = llm_provider.build_chat_model(
                provider_type="dashscope",
                model_id="qwen-max",
                base_url="https://dashscope.aliyuncs.com/api/v1",
                api_key="secret",
                streaming=True,
                model_parameters={"temperature": 0.5},
            )

            self.assertEqual(model, "dashscope-model")
            import_module_mock.assert_called_once_with("langchain_community.chat_models.tongyi")
            kwargs = dashscope_ctor.call_args.kwargs
            self.assertEqual(kwargs["model"], "qwen-max")
            self.assertEqual(
                kwargs["model_kwargs"],
                {"temperature": 0.5, "base_address": "https://dashscope.aliyuncs.com/api/v1"},
            )
            self.assertEqual(kwargs["api_key"], "secret")

    def test_build_chat_model_rejects_missing_base_url_for_openai_family(self) -> None:
        assert llm_provider is not None
        with self.assertRaises(ValueError):
            llm_provider.build_chat_model(
                provider_type="openai_compatible",
                model_id="demo-model",
                base_url="",
                api_key="secret",
                streaming=False,
                model_parameters={},
            )

    def test_build_chat_model_rejects_qwen35_series_for_dashscope_chat_tongyi(self) -> None:
        assert llm_provider is not None
        with self.assertRaises(ValueError) as ctx:
            llm_provider.build_chat_model(
                provider_type="dashscope",
                model_id="qwen3.5-plus",
                base_url="https://dashscope.aliyuncs.com/api/v1",
                api_key="secret",
                streaming=False,
                model_parameters={},
            )

        self.assertIn("provider_type=dashscope does not support", str(ctx.exception))
        self.assertIn("compatible-mode/v1", str(ctx.exception))
