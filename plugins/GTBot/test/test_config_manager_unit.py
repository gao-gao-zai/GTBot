from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import plugins.GTBot.ConfigManager as config_manager

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    config_manager = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestConfigManagerUnit(unittest.TestCase):
    def test_provider_level_process_tool_call_deltas_becomes_model_parameter_default(self) -> None:
        assert config_manager is not None

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_dir = Path(tmp_dir)
            behavioral_prompt = prompt_dir / "behavioral.txt"
            character_prompt = prompt_dir / "character.txt"
            behavioral_prompt.write_text("behavior", encoding="utf-8")
            character_prompt.write_text("character", encoding="utf-8")

            api_config = config_manager.Original.APIConfiguration.model_validate(
                {
                    "aliyun": {
                        "provider_type": "openai_compatible",
                        "process_tool_call_deltas": True,
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": "sk-test",
                        "llm_models": {
                            "default_model": {
                                "model": "qwen-plus",
                                "max_input_tokens": 32768,
                                "supports_vision": False,
                                "supports_audio": False,
                                "parameters": {
                                    "stream": True,
                                },
                            },
                            "override_model": {
                                "model": "qwen-plus",
                                "max_input_tokens": 32768,
                                "supports_vision": False,
                                "supports_audio": False,
                                "parameters": {
                                    "stream": True,
                                    "process_tool_call_deltas": False,
                                },
                            },
                        },
                    }
                }
            )

            original_default_group = config_manager.Original.SingleConfigurationGroup.model_validate(
                {
                    "chat_model": {
                        "model": "aliyun/default_model",
                        "maximum_number_of_incoming_messages": 20,
                        "behavioral_prompt": behavioral_prompt.name,
                        "character_prompt": character_prompt.name,
                    },
                    "message_format_placeholder": "[$message]",
                }
            )
            processed_default_group = config_manager.Processed.CurrentConfigGroup.from_single_configuration_group(
                original=original_default_group,
                api_config=api_config,
                group_name="default",
                prompt_dir_path=prompt_dir,
            )
            self.assertTrue(processed_default_group.chat_model.parameters["process_tool_call_deltas"])

            original_override_group = config_manager.Original.SingleConfigurationGroup.model_validate(
                {
                    "chat_model": {
                        "model": "aliyun/override_model",
                        "maximum_number_of_incoming_messages": 20,
                        "behavioral_prompt": behavioral_prompt.name,
                        "character_prompt": character_prompt.name,
                    },
                    "message_format_placeholder": "[$message]",
                }
            )
            processed_override_group = config_manager.Processed.CurrentConfigGroup.from_single_configuration_group(
                original=original_override_group,
                api_config=api_config,
                group_name="override",
                prompt_dir_path=prompt_dir,
            )
            self.assertFalse(processed_override_group.chat_model.parameters["process_tool_call_deltas"])

    def test_continuation_fields_are_merged_into_runtime_config(self) -> None:
        assert config_manager is not None

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_dir = Path(tmp_dir)
            behavioral_prompt = prompt_dir / "behavioral.txt"
            character_prompt = prompt_dir / "character.txt"
            behavioral_prompt.write_text("behavior", encoding="utf-8")
            character_prompt.write_text("character", encoding="utf-8")

            api_config = config_manager.Original.APIConfiguration.model_validate(
                {
                    "main": {
                        "provider_type": "openai_compatible",
                        "base_url": "https://main.example/v1",
                        "api_key": "main-key",
                        "llm_models": {
                            "chat": {
                                "model": "main-chat",
                                "max_input_tokens": 32768,
                                "supports_vision": False,
                                "supports_audio": False,
                                "parameters": {"temperature": 0.7},
                            }
                        },
                    },
                    "judge": {
                        "provider_type": "openai_compatible",
                        "base_url": "https://judge.example/v1",
                        "api_key": "judge-key",
                        "llm_models": {
                            "mini": {
                                "model": "judge-mini",
                                "max_input_tokens": 8192,
                                "supports_vision": False,
                                "supports_audio": False,
                                "parameters": {"temperature": 0.1},
                            }
                        },
                    },
                }
            )

            original_group = config_manager.Original.SingleConfigurationGroup.model_validate(
                {
                    "chat_model": {
                        "model": "main/chat",
                        "maximum_number_of_incoming_messages": 20,
                        "behavioral_prompt": behavioral_prompt.name,
                        "character_prompt": character_prompt.name,
                        "continuation": {
                            "enabled": True,
                            "window_seconds": 45,
                            "debounce_seconds": 3,
                            "scope": "exclude_auto",
                            "analyzer_model": "judge/mini",
                            "analyzer_parameters": {"top_p": 0.8},
                            "max_pending_messages": 6,
                            "max_accumulated_messages": 9,
                            "pre_history_messages": 4,
                            "max_analyzer_context_messages": 40,
                        },
                    },
                    "message_format_placeholder": "[$message]",
                }
            )

            processed_group = config_manager.Processed.CurrentConfigGroup.from_single_configuration_group(
                original=original_group,
                api_config=api_config,
                group_name="default",
                prompt_dir_path=prompt_dir,
            )

            continuation = processed_group.chat_model.continuation
            self.assertTrue(continuation.enabled)
            self.assertEqual(continuation.scope, "exclude_auto")
            self.assertEqual(continuation.window_seconds, 45)
            self.assertEqual(continuation.debounce_seconds, 3)
            self.assertEqual(continuation.analyzer_provider, "judge")
            self.assertEqual(continuation.analyzer_model_id, "judge-mini")
            self.assertEqual(continuation.analyzer_provider_type, "openai_compatible")
            self.assertEqual(continuation.analyzer_base_url, "https://judge.example/v1")
            self.assertEqual(continuation.analyzer_api_key, "judge-key")
            self.assertEqual(continuation.analyzer_parameters["temperature"], 0.1)
            self.assertEqual(continuation.analyzer_parameters["top_p"], 0.8)
            self.assertEqual(continuation.max_pending_messages, 6)
            self.assertEqual(continuation.max_accumulated_messages, 9)
            self.assertEqual(continuation.pre_history_messages, 4)
            self.assertEqual(continuation.max_analyzer_context_messages, 40)

    def test_continuation_rejects_invalid_accumulated_limit(self) -> None:
        assert config_manager is not None

        with self.assertRaises(ValueError):
            config_manager.Original.SingleConfigurationGroup.model_validate(
                {
                    "chat_model": {
                        "model": "main/chat",
                        "maximum_number_of_incoming_messages": 20,
                        "behavioral_prompt": "behavioral.txt",
                        "character_prompt": "character.txt",
                        "continuation": {
                            "max_pending_messages": 8,
                            "max_accumulated_messages": 7,
                        },
                    },
                    "message_format_placeholder": "[$message]",
                }
            )

    def test_continuation_rejects_negative_pre_history_messages(self) -> None:
        assert config_manager is not None

        with self.assertRaises(ValueError):
            config_manager.Original.SingleConfigurationGroup.model_validate(
                {
                    "chat_model": {
                        "model": "main/chat",
                        "maximum_number_of_incoming_messages": 20,
                        "behavioral_prompt": "behavioral.txt",
                        "character_prompt": "character.txt",
                        "continuation": {
                            "pre_history_messages": -1,
                        },
                    },
                    "message_format_placeholder": "[$message]",
                }
            )

    def test_continuation_rejects_non_positive_max_analyzer_context_messages(self) -> None:
        assert config_manager is not None

        with self.assertRaises(ValueError):
            config_manager.Original.SingleConfigurationGroup.model_validate(
                {
                    "chat_model": {
                        "model": "main/chat",
                        "maximum_number_of_incoming_messages": 20,
                        "behavioral_prompt": "behavioral.txt",
                        "character_prompt": "character.txt",
                        "continuation": {
                            "max_analyzer_context_messages": 0,
                        },
                    },
                    "message_format_placeholder": "[$message]",
                }
            )


if __name__ == "__main__":
    unittest.main()
