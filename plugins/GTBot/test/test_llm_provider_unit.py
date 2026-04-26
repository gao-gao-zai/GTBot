from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from plugins.GTBot import llm_provider

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

    def test_build_chat_model_delegates_to_adapter_factory(self) -> None:
        assert llm_provider is not None
        sentinel = object()
        with patch.object(llm_provider, "build_chat_adapter_model", return_value=sentinel) as builder_mock:
            model = llm_provider.build_chat_model(
                provider_type="openai_compatible",
                model_id="demo-model",
                base_url="https://example.test/v1",
                api_key="secret",
                streaming=True,
                model_parameters={"temperature": 0.7},
            )

        self.assertIs(model, sentinel)
        builder_mock.assert_called_once_with(
            provider_type="openai_compatible",
            model_id="demo-model",
            base_url="https://example.test/v1",
            api_key="secret",
            streaming=True,
            model_parameters={"temperature": 0.7},
        )
