from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from requests.exceptions import HTTPError

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from plugins.GTBot import chat_adapter

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    chat_adapter = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"runtime dependency missing: {_IMPORT_ERROR}")
class TestChatAdapterFactoryUnit(unittest.TestCase):
    def test_build_chat_adapter_model_openai_compatible_uses_extra_body(self) -> None:
        assert chat_adapter is not None
        with patch.object(chat_adapter, "RawCaptureChatOpenAI") as openai_cls:
            sentinel = object()
            openai_cls.return_value = sentinel

            model = chat_adapter.build_chat_adapter_model(
                provider_type="openai_compatible",
                model_id="demo-model",
                base_url="https://example.test/v1",
                api_key="secret",
                streaming=True,
                model_parameters={"temperature": 0.7},
            )

        self.assertIs(model, sentinel)
        kwargs = openai_cls.call_args.kwargs
        self.assertEqual(kwargs["model"], "demo-model")
        self.assertEqual(kwargs["base_url"], "https://example.test/v1")
        self.assertEqual(kwargs["provider_type"], "openai_compatible")
        self.assertTrue(kwargs["streaming"])
        self.assertEqual(kwargs["extra_body"], {"temperature": 0.7})
        self.assertEqual(kwargs["api_key"].get_secret_value(), "secret")

    def test_build_chat_adapter_model_openai_responses_enables_responses_api(self) -> None:
        assert chat_adapter is not None
        with patch.object(chat_adapter, "RawCaptureChatOpenAI") as openai_cls:
            sentinel = object()
            openai_cls.return_value = sentinel

            model = chat_adapter.build_chat_adapter_model(
                provider_type="openai_responses",
                model_id="demo-model",
                base_url="https://example.test/v1",
                api_key="secret",
                streaming=False,
                model_parameters={"temperature": 0.2, "output_version": "responses/v1"},
            )

        self.assertIs(model, sentinel)
        kwargs = openai_cls.call_args.kwargs
        self.assertFalse(kwargs["streaming"])
        self.assertTrue(kwargs["use_responses_api"])
        self.assertEqual(kwargs["extra_body"], {"temperature": 0.2})
        self.assertEqual(kwargs["output_version"], "responses/v1")

    def test_build_chat_adapter_model_openai_sets_raw_response_support_flag(self) -> None:
        assert chat_adapter is not None
        model = chat_adapter.RawCaptureChatOpenAI.model_construct()
        self.assertTrue(getattr(model, "_gtbot_raw_response_available"))

    def test_build_chat_adapter_model_anthropic_uses_chat_anthropic(self) -> None:
        assert chat_adapter is not None
        with patch.object(chat_adapter.importlib, "import_module") as import_module_mock:
            anthropic_ctor = Mock(return_value=SimpleNamespace())
            import_module_mock.return_value = SimpleNamespace(ChatAnthropic=anthropic_ctor)

            model = chat_adapter.build_chat_adapter_model(
                provider_type="anthropic",
                model_id="claude-3-7-sonnet",
                base_url="https://api.anthropic.com",
                api_key="secret",
                streaming=False,
                model_parameters={"temperature": 0.3},
            )

        self.assertFalse(getattr(model, "_gtbot_raw_response_available"))
        kwargs = anthropic_ctor.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-3-7-sonnet")
        self.assertEqual(kwargs["base_url"], "https://api.anthropic.com")
        self.assertEqual(kwargs["model_kwargs"], {"temperature": 0.3})

    def test_build_chat_adapter_model_gemini_uses_google_chat_model(self) -> None:
        assert chat_adapter is not None
        with patch.object(chat_adapter.importlib, "import_module") as import_module_mock:
            gemini_ctor = Mock(return_value=SimpleNamespace())
            import_module_mock.return_value = SimpleNamespace(ChatGoogleGenerativeAI=gemini_ctor)

            model = chat_adapter.build_chat_adapter_model(
                provider_type="gemini",
                model_id="gemini-2.5-flash",
                base_url="",
                api_key="secret",
                streaming=True,
                model_parameters={"temperature": 0.9},
            )

        self.assertFalse(getattr(model, "_gtbot_raw_response_available"))
        kwargs = gemini_ctor.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertTrue(kwargs["streaming"])
        self.assertEqual(kwargs["model_kwargs"], {"temperature": 0.9})
        self.assertEqual(kwargs["google_api_key"].get_secret_value(), "secret")

    def test_build_chat_adapter_model_dashscope_uses_chat_tongyi(self) -> None:
        assert chat_adapter is not None
        with patch.object(chat_adapter.importlib, "import_module") as import_module_mock:
            dashscope_ctor = Mock(
                return_value=SimpleNamespace(
                    client=SimpleNamespace(call=Mock()),
                    subtract_client_response=Mock(),
                )
            )
            import_module_mock.return_value = SimpleNamespace(
                ChatTongyi=dashscope_ctor,
                _create_retry_decorator=lambda _llm: (lambda func: func),
            )

            model = chat_adapter.build_chat_adapter_model(
                provider_type="dashscope",
                model_id="qwen-max",
                base_url="https://dashscope.aliyuncs.com/api/v1",
                api_key="secret",
                streaming=True,
                model_parameters={"temperature": 0.5},
            )

        self.assertFalse(getattr(model, "_gtbot_raw_response_available"))
        kwargs = dashscope_ctor.call_args.kwargs
        self.assertEqual(
            kwargs["model_kwargs"],
            {"temperature": 0.5, "base_address": "https://dashscope.aliyuncs.com/api/v1"},
        )
        self.assertEqual(kwargs["api_key"], "secret")

    def test_patch_tongyi_error_handling_avoids_keyerror_request(self) -> None:
        assert chat_adapter is not None

        class _DashScopeResp(dict[str, Any]):
            def __getattr__(self, name: str) -> Any:
                return self[name]

        model = SimpleNamespace(
            client=SimpleNamespace(
                call=Mock(
                    return_value=_DashScopeResp(
                        {
                            "status_code": 500,
                            "request_id": "req-dashscope",
                            "code": "InternalError",
                            "message": "upstream failed",
                        }
                    )
                )
            ),
            subtract_client_response=Mock(),
        )
        tongyi_mod = SimpleNamespace(_create_retry_decorator=lambda _llm: (lambda func: func))

        patched_model = chat_adapter._patch_tongyi_error_handling(model, tongyi_mod)

        with self.assertRaises(HTTPError) as ctx:
            patched_model.completion_with_retry(prompt="hello")

        self.assertIn("req-dashscope", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, KeyError)

    def test_patch_tongyi_error_handling_supports_pydantic_chat_tongyi_instance(self) -> None:
        assert chat_adapter is not None

        model = ChatTongyi.model_construct(
            client=SimpleNamespace(call=Mock(return_value={"status_code": 200})),
            subtract_client_response=Mock(),
        )
        tongyi_mod = SimpleNamespace(_create_retry_decorator=lambda _llm: (lambda func: func))

        patched_model = chat_adapter._patch_tongyi_error_handling(model, tongyi_mod)

        self.assertTrue(getattr(patched_model, "_gtbot_safe_tongyi_error_patched"))
        self.assertEqual(patched_model.completion_with_retry(prompt="hello"), {"status_code": 200})

    def test_build_chat_adapter_model_rejects_qwen35_series_for_dashscope_chat_tongyi(self) -> None:
        assert chat_adapter is not None
        with self.assertRaises(ValueError) as ctx:
            chat_adapter.build_chat_adapter_model(
                provider_type="dashscope",
                model_id="qwen3.5-plus",
                base_url="https://dashscope.aliyuncs.com/api/v1",
                api_key="secret",
                streaming=False,
                model_parameters={},
            )

        self.assertIn("provider_type=dashscope does not support", str(ctx.exception))
        self.assertIn("compatible-mode/v1", str(ctx.exception))


@unittest.skipIf(_IMPORT_ERROR is not None, f"runtime dependency missing: {_IMPORT_ERROR}")
class TestChatAdapterRawResponseUnit(unittest.IsolatedAsyncioTestCase):
    def _build_message_result(self) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="hello"))])

    def test_build_raw_response_payload_falls_back_to_body_text(self) -> None:
        assert chat_adapter is not None
        raw_response = SimpleNamespace(
            headers={"Authorization": "Bearer secret", "x-request-id": "req-1"},
            http_response=SimpleNamespace(
                status_code=200,
                json=Mock(side_effect=ValueError("not json")),
                text="x" * 13050,
            ),
        )

        payload = chat_adapter._build_raw_response_payload(
            provider_type="openai_compatible",
            api_style="chat_completions",
            raw_response=raw_response,
            parsed_response=None,
        )

        self.assertEqual(payload["headers"]["Authorization"], "***")
        self.assertEqual(payload["request_id"], "req-1")
        self.assertIsNone(payload["body_json"])
        self.assertIsNotNone(payload["body_text"])
        assert payload["body_text"] is not None
        self.assertIn("truncated", payload["body_text"])

    def test_generate_injects_raw_response_payload(self) -> None:
        assert chat_adapter is not None
        parsed_response = SimpleNamespace(
            model_dump=lambda **kwargs: {"id": "chatcmpl-1", "choices": [{"message": {"content": "hello"}}]}
        )
        raw_response = SimpleNamespace(
            headers={"Authorization": "Bearer secret", "x-request-id": "req-1"},
            http_response=SimpleNamespace(
                status_code=200,
                json=lambda: {"id": "chatcmpl-1", "choices": [{"message": {"content": "hello"}}]},
                text='{"id":"chatcmpl-1"}',
            ),
            parse=Mock(return_value=parsed_response),
        )
        model = chat_adapter.RawCaptureChatOpenAI.model_construct(
            provider_type="openai_compatible",
            client=SimpleNamespace(with_raw_response=SimpleNamespace(create=Mock(return_value=raw_response))),
            async_client=None,
            root_client=None,
            root_async_client=None,
            model_name="demo-model",
            include_response_headers=False,
            output_version=None,
            use_responses_api=False,
            model_kwargs={},
            extra_body={},
        )
        model._ensure_sync_client_available = Mock()
        model._get_request_payload = Mock(return_value={"messages": []})
        model._use_responses_api = Mock(return_value=False)
        model._create_chat_result = Mock(return_value=self._build_message_result())

        result = model._generate(messages=[])

        payload = result.generations[0].message.additional_kwargs["raw_response"]
        self.assertEqual(payload["status_code"], 200)
        self.assertEqual(payload["request_id"], "req-1")
        self.assertEqual(payload["headers"]["Authorization"], "***")
        self.assertEqual(payload["body_json"]["id"], "chatcmpl-1")
        self.assertTrue(result.generations[0].message.response_metadata["raw_response_available"])

    async def test_agenerate_injects_raw_response_payload_for_responses_api(self) -> None:
        assert chat_adapter is not None
        parsed_response = SimpleNamespace(
            model_dump=lambda **kwargs: {"id": "resp_1", "output": [{"type": "message"}]}
        )
        raw_response = SimpleNamespace(
            headers={"x-request-id": "req-async"},
            http_response=SimpleNamespace(
                status_code=202,
                json=lambda: {"id": "resp_1"},
                text='{"id":"resp_1"}',
            ),
            parse=Mock(return_value=parsed_response),
        )
        model = chat_adapter.RawCaptureChatOpenAI.model_construct(
            provider_type="openai_responses",
            client=None,
            async_client=None,
            root_client=None,
            root_async_client=SimpleNamespace(
                responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=AsyncMock(return_value=raw_response)))
            ),
            model_name="demo-model",
            include_response_headers=False,
            output_version="responses/v1",
            use_responses_api=True,
            model_kwargs={},
            extra_body={},
        )
        model._get_request_payload = Mock(return_value={"messages": []})
        model._use_responses_api = Mock(return_value=True)

        with patch.object(
            chat_adapter,
            "_construct_lc_result_from_responses_api",
            return_value=self._build_message_result(),
        ) as construct_mock:
            result = await model._agenerate(messages=[])

        construct_mock.assert_called_once()
        payload = result.generations[0].message.additional_kwargs["raw_response"]
        self.assertEqual(payload["api_style"], "responses")
        self.assertEqual(payload["status_code"], 202)
        self.assertEqual(payload["request_id"], "req-async")
        self.assertTrue(result.generations[0].message.response_metadata["raw_response_available"])

    async def test_astream_responses_injects_raw_response_payload_on_last_chunk(self) -> None:
        assert chat_adapter is not None

        class _AsyncContextManager:
            def __init__(self, chunks: list[Any]) -> None:
                self._chunks = chunks

            async def __aenter__(self) -> "_AsyncContextManager":
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

            def __aiter__(self) -> "_AsyncContextManager":
                self._iter = iter(self._chunks)
                return self

            async def __anext__(self) -> Any:
                try:
                    return next(self._iter)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        parsed_response = SimpleNamespace(
            model_dump=lambda **kwargs: {"id": "resp_stream", "output": [{"type": "message"}]}
        )
        raw_response = SimpleNamespace(
            headers={"x-request-id": "req-stream"},
            http_response=SimpleNamespace(
                status_code=200,
                json=lambda: {"id": "resp_stream"},
                text='{"id":"resp_stream"}',
            ),
            parse=Mock(
                return_value=_AsyncContextManager(
                    [
                        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_stream")),
                        SimpleNamespace(type="response.completed", response=parsed_response),
                    ]
                )
            ),
        )
        model = chat_adapter.RawCaptureChatOpenAI.model_construct(
            provider_type="openai_responses",
            root_async_client=SimpleNamespace(
                with_raw_response=SimpleNamespace(
                    responses=SimpleNamespace(create=AsyncMock(return_value=raw_response))
                )
            ),
            model_name="demo-model",
            include_response_headers=False,
            output_version="responses/v1",
            use_responses_api=True,
            model_kwargs={},
            extra_body={},
        )
        model._get_request_payload = Mock(return_value={"messages": []})
        model._use_responses_api = Mock(return_value=True)

        with patch.object(
            chat_adapter,
            "_convert_responses_chunk_to_generation_chunk",
            side_effect=[
                (0, 0, 0, None),
                (
                    0,
                    0,
                    0,
                    ChatGenerationChunk(
                        message=chat_adapter.AIMessageChunk(
                            content="hello",
                            additional_kwargs={},
                            response_metadata={},
                            chunk_position="last",
                        )
                    ),
                ),
            ],
        ):
            chunks = [chunk async for chunk in model._astream(messages=[])]

        self.assertEqual(len(chunks), 1)
        payload = chunks[0].message.additional_kwargs["raw_response"]
        self.assertEqual(payload["request_id"], "req-stream")
        self.assertTrue(chunks[0].message.response_metadata["raw_response_available"])
