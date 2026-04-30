from __future__ import annotations

import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage as _AIMessageType
    from plugins.GTBot.services.cost import CostLedgerService as _CostLedgerServiceType
    from plugins.GTBot.services.cost import CostLedgerStore as _CostLedgerStoreType
    from plugins.GTBot.services.plugin_api.permissions import PermissionError as _PermissionErrorType
    from plugins.GTBot.services.plugin_api.permissions import PermissionRole as _PermissionRoleType
    from plugins.GTBot.services.plugin_system.types import PluginContext as _PluginContextType

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AIMessage: Any = None
CostLedgerService: Any = None
CostLedgerStore: Any = None
PermissionError: Any = Exception
PermissionRole: Any = None
plugin_context_scope: Any = None
PluginContext: Any = None
cost_service_mod: Any = None

try:
    from langchain_core.messages import AIMessage as _RuntimeAIMessage

    from plugins.GTBot.services.cost import CostLedgerService as _RuntimeCostLedgerService
    from plugins.GTBot.services.cost import CostLedgerStore as _RuntimeCostLedgerStore
    from plugins.GTBot.services.plugin_api.permissions import PermissionError as _RuntimePermissionError
    from plugins.GTBot.services.plugin_api.permissions import PermissionRole as _RuntimePermissionRole
    from plugins.GTBot.services.plugin_system.runtime import plugin_context_scope
    from plugins.GTBot.services.plugin_system.types import PluginContext as _RuntimePluginContext

    import plugins.GTBot.services.cost.service as cost_service_mod

    AIMessage = _RuntimeAIMessage
    CostLedgerService = _RuntimeCostLedgerService
    CostLedgerStore = _RuntimeCostLedgerStore
    PermissionError = _RuntimePermissionError
    PermissionRole = _RuntimePermissionRole
    PluginContext = _RuntimePluginContext
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestCostServiceUnit(unittest.IsolatedAsyncioTestCase):
    """验证消费账本服务的自动计费、上下文记账与权限边界。"""

    async def asyncSetUp(self) -> None:
        assert CostLedgerStore is not None
        assert CostLedgerService is not None
        self._temp_dir = Path(tempfile.mkdtemp(prefix="gtbot_cost_test_"))
        self.addCleanup(lambda: shutil.rmtree(self._temp_dir, ignore_errors=True))
        db_path = self._temp_dir / "cost.sqlite3"
        self.store = CostLedgerStore(db_path=db_path)
        self.service = CostLedgerService(store=self.store)

    async def test_record_chat_cost_from_response_should_extract_tokens_and_compute_amount(self) -> None:
        """聊天账单应按 provider 规则提取 token 并按模型价格计算金额。"""

        assert AIMessage is not None
        runtime_context = SimpleNamespace(
            user_id=10001,
            group_id=20001,
            session_id="group:20001",
            response_id="resp_cost_1",
        )
        chat_model_config = SimpleNamespace(
            provider_name="ds",
            model_id="chat-model-id",
            cost=SimpleNamespace(
                provider_usage_rules={
                    "ds": SimpleNamespace(
                        input_tokens_path="usage.prompt_tokens",
                        output_tokens_path="usage.completion_tokens",
                        cache_read_tokens_path="usage.prompt_tokens_details.cached_tokens",
                        request_id_path="id",
                    )
                },
                model_pricing={
                    "ds": {
                        "chat-model-id": SimpleNamespace(
                            enabled=True,
                            input_price_per_million=2.0,
                            output_price_per_million=8.0,
                            cache_read_price_per_million=1.0,
                            currency="CNY",
                        )
                    }
                },
            ),
        )
        response = {
            "messages": [
                AIMessage(
                    content="ok",
                    additional_kwargs={
                        "raw_response": {
                            "body_json": {
                                "id": "req_1",
                                "usage": {
                                    "prompt_tokens": 1000,
                                    "completion_tokens": 500,
                                    "prompt_tokens_details": {"cached_tokens": 200},
                                },
                            }
                        }
                    },
                )
            ]
        }

        recorded = await self.service.record_chat_cost_from_response(
            response=response,
            runtime_context=runtime_context,
            chat_model_config=chat_model_config,
        )

        self.assertTrue(recorded)
        records = self.store.list_records(owner_user_id=10001)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.response_id, "resp_cost_1")
        self.assertEqual(record.provider, "ds")
        self.assertEqual(record.model_name, "chat-model-id")
        self.assertAlmostEqual(record.amount, 0.0062, places=6)
        self.assertEqual(record.extra["input_tokens"], 1000.0)
        self.assertEqual(record.extra["output_tokens"], 500.0)
        self.assertEqual(record.extra["cache_read_tokens"], 200.0)
        self.assertEqual(record.extra["request_id"], "req_1")

    async def test_record_chat_cost_from_response_should_treat_missing_cache_tokens_as_zero(self) -> None:
        """缺失缓存读取字段时应按 0 处理，而不是导致整笔账单丢失。"""

        assert AIMessage is not None
        runtime_context = SimpleNamespace(
            user_id=10002,
            group_id=None,
            session_id="private:10002",
            response_id="resp_cost_2",
        )
        chat_model_config = SimpleNamespace(
            provider_name="ds",
            model_id="chat-model-id",
            cost=SimpleNamespace(
                provider_usage_rules={
                    "ds": SimpleNamespace(
                        input_tokens_path="usage.prompt_tokens",
                        output_tokens_path="usage.completion_tokens",
                        cache_read_tokens_path="usage.prompt_tokens_details.cached_tokens",
                        request_id_path="id",
                    )
                },
                model_pricing={
                    "ds": {
                        "chat-model-id": SimpleNamespace(
                            enabled=True,
                            input_price_per_million=1.0,
                            output_price_per_million=2.0,
                            cache_read_price_per_million=3.0,
                            currency="CNY",
                        )
                    }
                },
            ),
        )
        response = {
            "messages": [
                AIMessage(
                    content="ok",
                    additional_kwargs={
                        "raw_response": {
                            "body_json": {
                                "id": "req_2",
                                "usage": {
                                    "prompt_tokens": 100,
                                    "completion_tokens": 50,
                                },
                            }
                        }
                    },
                )
            ]
        }

        recorded = await self.service.record_chat_cost_from_response(
            response=response,
            runtime_context=runtime_context,
            chat_model_config=chat_model_config,
        )

        self.assertTrue(recorded)
        record = self.store.list_records(owner_user_id=10002)[0]
        self.assertEqual(record.extra["cache_read_tokens"], 0.0)
        self.assertAlmostEqual(record.amount, 0.0002, places=6)

    async def test_record_chat_cost_from_response_should_skip_when_model_pricing_missing(self) -> None:
        """缺失模型价格配置时不应写账。"""

        assert AIMessage is not None
        runtime_context = SimpleNamespace(
            user_id=10003,
            group_id=None,
            session_id="private:10003",
            response_id="resp_cost_3",
        )
        chat_model_config = SimpleNamespace(
            provider_name="ds",
            model_id="missing-model",
            cost=SimpleNamespace(
                provider_usage_rules={
                    "ds": SimpleNamespace(
                        input_tokens_path="usage.prompt_tokens",
                        output_tokens_path="usage.completion_tokens",
                        cache_read_tokens_path="",
                        request_id_path="id",
                    )
                },
                model_pricing={},
            ),
        )
        response = {
            "messages": [
                AIMessage(
                    content="ok",
                    additional_kwargs={
                        "raw_response": {
                            "body_json": {
                                "id": "req_3",
                                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                            }
                        }
                    },
                )
            ]
        }

        recorded = await self.service.record_chat_cost_from_response(
            response=response,
            runtime_context=runtime_context,
            chat_model_config=chat_model_config,
        )

        self.assertFalse(recorded)
        self.assertEqual(self.store.list_records(owner_user_id=10003), [])

    async def test_record_chat_cost_from_response_should_fallback_to_usage_metadata(self) -> None:
        """流式响应缺少 raw_response 时，应回退到 usage_metadata 记账。"""

        assert AIMessage is not None
        runtime_context = SimpleNamespace(
            user_id=10004,
            group_id=20004,
            session_id="group:20004",
            response_id="resp_cost_4",
        )
        chat_model_config = SimpleNamespace(
            provider_name="ds",
            model_id="deepseek-v4-flash",
            cost=SimpleNamespace(
                provider_usage_rules={
                    "ds": SimpleNamespace(
                        input_tokens_path="usage.prompt_tokens",
                        output_tokens_path="usage.completion_tokens",
                        cache_read_tokens_path="usage.prompt_tokens_details.cached_tokens",
                        request_id_path="id",
                        input_tokens_include_cache_read=False,
                        streaming=SimpleNamespace(
                            input_tokens_path="input_tokens",
                            output_tokens_path="output_tokens",
                            cache_read_tokens_path="input_token_details.cache_read",
                            request_id_path="",
                            input_tokens_include_cache_read=True,
                        ),
                        non_streaming=None,
                    )
                },
                model_pricing={
                    "ds": {
                        "deepseek-v4-flash": SimpleNamespace(
                            enabled=True,
                            input_price_per_million=1.0,
                            output_price_per_million=2.0,
                            cache_read_price_per_million=0.02,
                            currency="CNY",
                        )
                    }
                },
            ),
        )
        response = {
            "messages": [
                AIMessage(
                    content="ok",
                    additional_kwargs={},
                    response_metadata={
                        "finish_reason": "stop",
                        "model_name": "deepseek-v4-flash",
                    },
                    usage_metadata={
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "input_token_details": {"cache_read": 30},
                        "output_token_details": {},
                    },
                )
            ]
        }

        recorded = await self.service.record_chat_cost_from_response(
            response=response,
            runtime_context=runtime_context,
            chat_model_config=chat_model_config,
        )

        self.assertTrue(recorded)
        record = self.store.list_records(owner_user_id=10004)[0]
        self.assertEqual(record.extra["input_tokens"], 70.0)
        self.assertEqual(record.extra["cache_read_tokens"], 30.0)
        self.assertEqual(record.extra["output_tokens"], 20.0)
        self.assertAlmostEqual(record.amount, 0.0001106, places=7)

    async def test_record_chat_cost_from_response_should_record_each_ai_message_separately(self) -> None:
        """同一轮内多次模型请求应按 AIMessage 分别落账，而不是只记最后一条。"""

        assert AIMessage is not None
        runtime_context = SimpleNamespace(
            user_id=10005,
            group_id=20005,
            session_id="group:20005",
            response_id="resp_cost_5",
        )
        chat_model_config = SimpleNamespace(
            provider_name="ds",
            model_id="chat-model-id",
            cost=SimpleNamespace(
                provider_usage_rules={
                    "ds": SimpleNamespace(
                        input_tokens_path="usage.prompt_tokens",
                        output_tokens_path="usage.completion_tokens",
                        cache_read_tokens_path="usage.prompt_tokens_details.cached_tokens",
                        request_id_path="id",
                    )
                },
                model_pricing={
                    "ds": {
                        "chat-model-id": SimpleNamespace(
                            enabled=True,
                            input_price_per_million=2.0,
                            output_price_per_million=8.0,
                            cache_read_price_per_million=1.0,
                            currency="CNY",
                        )
                    }
                },
            ),
        )
        response = {
            "messages": [
                AIMessage(
                    content="",
                    additional_kwargs={
                        "raw_response": {
                            "body_json": {
                                "id": "req_tool_call",
                                "usage": {
                                    "prompt_tokens": 120,
                                    "completion_tokens": 30,
                                    "prompt_tokens_details": {"cached_tokens": 0},
                                },
                            }
                        }
                    },
                ),
                AIMessage(
                    content="final answer",
                    additional_kwargs={
                        "raw_response": {
                            "body_json": {
                                "id": "req_final_answer",
                                "usage": {
                                    "prompt_tokens": 260,
                                    "completion_tokens": 90,
                                    "prompt_tokens_details": {"cached_tokens": 40},
                                },
                            }
                        }
                    },
                ),
            ]
        }

        recorded = await self.service.record_chat_cost_from_response(
            response=response,
            runtime_context=runtime_context,
            chat_model_config=chat_model_config,
        )

        self.assertTrue(recorded)
        records = self.store.list_records(owner_user_id=10005, limit=10)
        self.assertEqual(len(records), 2)
        request_ids = {str(item.extra["request_id"]) for item in records}
        self.assertEqual(request_ids, {"req_tool_call", "req_final_answer"})
        event_ids = {item.event_id for item in records}
        self.assertEqual(event_ids, {"chat_cost_request:req_tool_call", "chat_cost_request:req_final_answer"})
        self.assertEqual({int(item.extra["message_index"]) for item in records}, {0, 1})
        amount_by_request_id = {str(item.extra["request_id"]): item.amount for item in records}
        self.assertAlmostEqual(amount_by_request_id["req_tool_call"], 0.00048, places=8)
        self.assertAlmostEqual(amount_by_request_id["req_final_answer"], 0.00128, places=8)

    async def test_record_cost_for_current_request_should_fill_runtime_context_fields(self) -> None:
        """当前请求写账入口应自动补齐用户、群号、会话和响应 ID。"""

        assert PluginContext is not None
        assert plugin_context_scope is not None
        plugin_ctx = PluginContext(
            raw_messages=[],
            response_id="resp_plugin_1",
            runtime_context=SimpleNamespace(
                user_id=12345,
                group_id=54321,
                session_id="group:54321",
                response_id="resp_plugin_1",
            ),
        )

        with plugin_context_scope(plugin_ctx):
            recorded = await self.service.record_cost_for_current_request(
                source_name="demo_plugin",
                category="per_request_demo",
                billing_mode="per_request",
                quantity=1.0,
                unit_price=0.3,
                amount=0.3,
            )

        self.assertTrue(recorded)
        record = self.store.list_records(owner_user_id=12345)[0]
        self.assertEqual(record.actor_user_id, 12345)
        self.assertEqual(record.group_id, 54321)
        self.assertEqual(record.session_id, "group:54321")
        self.assertEqual(record.response_id, "resp_plugin_1")

    async def test_leaderboard_should_aggregate_by_owner_user_id(self) -> None:
        """排行榜应仅按账单归属用户聚合，而不是按操作者分裂条目。"""

        await self.service.record_plugin_cost(
            source_name="a",
            category="demo",
            billing_mode="direct_amount",
            quantity=1.0,
            unit_price=None,
            amount=1.2,
            owner_user_id=10001,
            actor_user_id=10001,
        )
        await self.service.record_plugin_cost(
            source_name="b",
            category="demo",
            billing_mode="direct_amount",
            quantity=1.0,
            unit_price=None,
            amount=2.3,
            owner_user_id=10001,
            actor_user_id=99999,
        )
        await self.service.record_plugin_cost(
            source_name="c",
            category="demo",
            billing_mode="direct_amount",
            quantity=1.0,
            unit_price=None,
            amount=1.0,
            owner_user_id=10002,
            actor_user_id=10002,
        )

        leaderboard = await self.service.get_leaderboard(limit=10)
        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(leaderboard[0].owner_user_id, 10001)
        self.assertAlmostEqual(leaderboard[0].total_amount, 3.5, places=6)
        self.assertEqual(leaderboard[0].record_count, 2)

    async def test_ensure_can_query_user_should_allow_self_and_admin_only(self) -> None:
        """权限判断应允许本人查询，并限制他人账单仅管理员及以上可查。"""

        assert cost_service_mod is not None
        assert PermissionRole is not None

        await self.service.ensure_can_query_user(requester_user_id=10001, target_user_id=10001)

        with patch.object(cost_service_mod, "get_role", AsyncMock(return_value=PermissionRole.ADMIN)):
            await self.service.ensure_can_query_user(requester_user_id=20001, target_user_id=10001)

        with patch.object(cost_service_mod, "get_role", AsyncMock(return_value=PermissionRole.USER)):
            with self.assertRaises(PermissionError):
                await self.service.ensure_can_query_user(requester_user_id=20002, target_user_id=10001)


if __name__ == "__main__":
    unittest.main()
