from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_find_matched_terms: Any
_iter_text_fragments: Any
_warn_if_contains_forbidden_terms: Any

try:
    from nonebot.adapters.onebot.v11 import Message, MessageSegment
    from plugins.GTBot.tools.outgoing_forbidden_word_audit import (
        _find_matched_terms as _imported_find_matched_terms,
        _iter_text_fragments as _imported_iter_text_fragments,
        _warn_if_contains_forbidden_terms as _imported_warn_if_contains_forbidden_terms,
    )

    _find_matched_terms = _imported_find_matched_terms
    _iter_text_fragments = _imported_iter_text_fragments
    _warn_if_contains_forbidden_terms = _imported_warn_if_contains_forbidden_terms
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    Message = None
    MessageSegment = None
    _find_matched_terms = None
    _iter_text_fragments = None
    _warn_if_contains_forbidden_terms = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestOutgoingForbiddenWordAuditUnit(unittest.TestCase):
    """验证出站违禁词审计插件的文本提取与告警行为。"""

    def test_iter_text_fragments_should_extract_nested_message_payload(self) -> None:
        """应能递归提取普通消息与合并转发节点中的文本内容。"""

        payload: dict[str, Any] = {
            "message": Message(
                [
                    MessageSegment.text("普通文本"),
                    MessageSegment.image(file="https://example.com/demo.png"),
                ]
            ),
            "messages": [
                {
                    "type": "node",
                    "data": {
                        "name": "GTBot",
                        "uin": "10001",
                        "content": Message(MessageSegment.text("这里提到逆向")),
                    },
                }
            ],
        }

        fragments = list(_iter_text_fragments(payload))
        self.assertIn("普通文本", fragments)
        self.assertIn("https://example.com/demo.png", fragments)
        self.assertIn("这里提到逆向", fragments)

    def test_find_matched_terms_should_respect_ascii_boundaries(self) -> None:
        """英文缩写应按边界匹配，避免误命中更长单词。"""

        fragments = [
            "讨论 ddos 风险和 JS算法分析",
            "access account success",
            "单独出现 cc 也要命中",
        ]

        matched = _find_matched_terms(fragments)
        self.assertEqual(matched, ["js算法分析", "ddos", "cc"])

    def test_warn_if_contains_forbidden_terms_should_log_warning_only_on_hit(self) -> None:
        """命中违禁词时应输出告警日志，未命中时不记录。"""

        with patch("plugins.GTBot.tools.outgoing_forbidden_word_audit.logger.warning") as warning_mock:
            _warn_if_contains_forbidden_terms(
                "send_group_msg",
                {"message": Message(MessageSegment.text("这段内容包含注册机 和 cc"))},
            )
            self.assertEqual(warning_mock.call_count, 1)
            logged_text = warning_mock.call_args.args[0]
            self.assertIn("forbidden-word audit hit", logged_text)

        with patch("plugins.GTBot.tools.outgoing_forbidden_word_audit.logger.warning") as warning_mock:
            _warn_if_contains_forbidden_terms(
                "send_group_msg",
                {"message": Message(MessageSegment.text("这是一条普通消息"))},
            )
            warning_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
