from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Callable

from ...ConfigManager import total_config
from ...Logger import logger


_WHITESPACE_RE = re.compile(r"\s+")
_STRING_METHOD_NAMES = frozenset({"lower", "strip", "startswith", "endswith"})


@dataclass(slots=True)
class ChatOptOutContext:
    """承载群关键词免触发判定所需的最小上下文。

    该对象只暴露规则匹配真正需要的基础字段，避免把 `event`、`bot` 等
    宿主对象直接暴露给表达式求值器，从而降低耦合和执行边界的不确定性。

    Attributes:
        text: 原始消息文本。
        normalized_text: 轻量归一化后的消息文本。
        group_id: 当前群号。
        user_id: 当前发送者 QQ 号。
        to_me: 当前消息是否被 NoneBot 识别为显式对机器人说话。
        mentioned_bot: 当前消息是否显式 `@GTBot`。
        trigger_keyword: 当前命中的群关键词；若尚未命中则为 `None`。
    """

    text: str
    normalized_text: str
    group_id: int
    user_id: int
    to_me: bool
    mentioned_bot: bool
    trigger_keyword: str | None = None


class SafeExpressionEvaluator:
    """安全执行受限布尔表达式。

    该执行器只支持少量 AST 节点、上下文字段和白名单函数，目标是让规则
    “看起来像 Python 条件表达式”，但不允许导入模块、访问文件或执行任意代码。
    表达式最终必须返回布尔值；其它返回类型会在调用方被转换为 `bool`。
    """

    def __init__(self) -> None:
        """初始化表达式执行器并注册白名单函数。"""

        self._functions: dict[str, Callable[..., Any]] = {
            "contains": self._contains,
            "contains_any": self._contains_any,
            "startswith": self._startswith,
            "endswith": self._endswith,
            "regex": self._regex,
            "len": self._safe_len,
            "bool": bool,
        }

    def evaluate(self, expression: str, context: ChatOptOutContext) -> bool:
        """执行受限表达式并返回布尔结果。

        Args:
            expression: 待执行的受限表达式源码。
            context: 当前消息的免触发判定上下文。

        Returns:
            表达式计算后的布尔值。

        Raises:
            ValueError: 当表达式语法非法或包含未授权节点时抛出。
        """

        tree = ast.parse(expression, mode="eval")
        self._validate_node(tree)
        result = self._eval_node(tree.body, self._build_scope(context))
        return bool(result)

    def _build_scope(self, context: ChatOptOutContext) -> dict[str, Any]:
        """构造表达式可见的上下文变量。

        Args:
            context: 当前消息的判定上下文。

        Returns:
            供表达式求值使用的只读变量映射。
        """

        return {
            "text": context.text,
            "normalized_text": context.normalized_text,
            "group_id": context.group_id,
            "user_id": context.user_id,
            "to_me": context.to_me,
            "mentioned_bot": context.mentioned_bot,
            "trigger_keyword": context.trigger_keyword,
            "True": True,
            "False": False,
            "None": None,
        }

    def _validate_node(self, node: ast.AST) -> None:
        """校验表达式 AST，只允许白名单节点和调用目标。

        Args:
            node: 待校验的 AST 节点。

        Raises:
            ValueError: 当节点类型或调用目标不受支持时抛出。
        """

        if isinstance(node, ast.Expression):
            self._validate_node(node.body)
            return
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                raise ValueError("只允许 and / or 布尔运算")
            for value in node.values:
                self._validate_node(value)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.Not):
                raise ValueError("只允许 not 一元运算")
            self._validate_node(node.operand)
            return
        if isinstance(node, ast.Compare):
            self._validate_node(node.left)
            for comparator in node.comparators:
                self._validate_node(comparator)
            for op in node.ops:
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
                    raise ValueError("比较表达式仅支持 == / != / in / not in")
            return
        if isinstance(node, ast.Call):
            self._validate_call(node)
            return
        if isinstance(node, ast.Name):
            return
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                self._validate_node(elt)
            return
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    def _validate_call(self, node: ast.Call) -> None:
        """校验函数或字符串方法调用是否合法。

        Args:
            node: 调用表达式节点。

        Raises:
            ValueError: 当调用目标、参数形式不受支持时抛出。
        """

        if node.keywords:
            raise ValueError("表达式调用不支持关键字参数")
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in self._functions:
                raise ValueError(f"不允许调用函数: {func.id}")
        elif isinstance(func, ast.Attribute):
            if func.attr not in _STRING_METHOD_NAMES:
                raise ValueError(f"不允许调用方法: {func.attr}")
            self._validate_node(func.value)
        else:
            raise ValueError("不支持的调用目标")
        for arg in node.args:
            self._validate_node(arg)

    def _eval_node(self, node: ast.AST, scope: dict[str, Any]) -> Any:
        """递归求值已通过校验的 AST。

        Args:
            node: 待求值节点。
            scope: 当前可见变量作用域。

        Returns:
            节点的求值结果。

        Raises:
            ValueError: 当运行时遇到未授权变量或调用形式时抛出。
        """

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for value in node.values:
                    result = self._eval_node(value, scope)
                    if not result:
                        return result
                return result
            result = False
            for value in node.values:
                result = self._eval_node(value, scope)
                if result:
                    return result
            return result
        if isinstance(node, ast.UnaryOp):
            return not bool(self._eval_node(node.operand, scope))
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, scope)
            for op, comparator_node in zip(node.ops, node.comparators):
                right = self._eval_node(comparator_node, scope)
                if isinstance(op, ast.Eq):
                    matched = left == right
                elif isinstance(op, ast.NotEq):
                    matched = left != right
                elif isinstance(op, ast.In):
                    matched = left in right
                else:
                    matched = left not in right
                if not matched:
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            return self._eval_call(node, scope)
        if isinstance(node, ast.Name):
            if node.id not in scope:
                raise ValueError(f"未知变量: {node.id}")
            return scope[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [self._eval_node(elt, scope) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt, scope) for elt in node.elts)
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    def _eval_call(self, node: ast.Call, scope: dict[str, Any]) -> Any:
        """执行已通过校验的函数或字符串方法调用。

        Args:
            node: 调用表达式节点。
            scope: 当前可见变量作用域。

        Returns:
            调用结果。

        Raises:
            ValueError: 当方法调用对象类型不正确时抛出。
        """

        args = [self._eval_node(arg, scope) for arg in node.args]
        func = node.func
        if isinstance(func, ast.Name):
            return self._functions[func.id](*args)
        if not isinstance(func, ast.Attribute):
            raise ValueError("不支持的调用目标")
        owner = self._eval_node(func.value, scope)
        if not isinstance(owner, str):
            raise ValueError("字符串方法只能用于字符串上下文")
        method = getattr(owner, func.attr)
        return method(*args)

    @staticmethod
    def _contains(text: Any, sub: Any) -> bool:
        """判断字符串是否包含子串。"""

        return str(sub) in str(text)

    @staticmethod
    def _contains_any(text: Any, subs: Any) -> bool:
        """判断字符串是否包含任一候选子串。"""

        haystack = str(text)
        if not isinstance(subs, (list, tuple, set, frozenset)):
            return False
        return any(str(item) in haystack for item in subs)

    @staticmethod
    def _startswith(text: Any, prefix: Any) -> bool:
        """判断字符串是否以前缀开头。"""

        return str(text).startswith(str(prefix))

    @staticmethod
    def _endswith(text: Any, suffix: Any) -> bool:
        """判断字符串是否以后缀结尾。"""

        return str(text).endswith(str(suffix))

    @staticmethod
    def _regex(pattern: Any, text: Any) -> bool:
        """使用正则表达式匹配字符串。

        Args:
            pattern: 正则表达式字符串。
            text: 待匹配文本。

        Returns:
            只要出现任意匹配片段即返回 `True`。

        Raises:
            ValueError: 当正则表达式无效时抛出。
        """

        try:
            return re.search(str(pattern), str(text)) is not None
        except re.error as exc:
            raise ValueError(f"无效正则表达式: {exc}") from exc

    @staticmethod
    def _safe_len(value: Any) -> int:
        """返回可迭代或字符串对象的长度。"""

        return len(value)


class ChatOptOutManager:
    """统一管理群关键词免触发规则的读取与匹配。

    管理器当前采用只读配置模式，不依赖数据库，也不缓存规则命中结果，
    以保持第一版实现最小且易于热重载配置。调用方只需提供消息上下文，
    即可得到是否应跳过当前群关键词触发的判定结果。
    """

    def __init__(self) -> None:
        """初始化免触发管理器及其表达式执行器。"""

        self._evaluator = SafeExpressionEvaluator()

    def normalize_text(self, text: str) -> str:
        """对消息文本做轻量归一化。

        当前只做首尾空白裁剪、连续空白折叠和小写转换，目的是让
        `suffix`/`keyword`/`expr` 规则在常见空格差异下保持稳定，同时
        不把该功能复杂化为内容审核系统。

        Args:
            text: 原始消息文本。

        Returns:
            归一化后的文本。
        """

        collapsed = _WHITESPACE_RE.sub(" ", str(text or "").strip())
        return collapsed.lower()

    def match_rule(self, context: ChatOptOutContext) -> str | None:
        """按配置顺序匹配当前消息的免触发规则。

        Args:
            context: 当前消息的判定上下文。

        Returns:
            命中时返回规则 ID；未命中时返回 `None`。
        """

        cfg = total_config.processed_configuration.current_config_group.chat_model.chat_opt_out
        if not cfg.enabled:
            return None

        for rule in cfg.rules:
            if not rule.enabled:
                continue
            try:
                if self._match_single_rule(rule_type=rule.type, value=rule.value, context=context):
                    return rule.id
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "chat opt-out rule evaluation failed: rule_id=%s type=%s error=%s",
                    rule.id,
                    rule.type,
                    exc,
                )
        return None

    def _match_single_rule(
        self,
        *,
        rule_type: str,
        value: str,
        context: ChatOptOutContext,
    ) -> bool:
        """执行单条规则匹配。

        Args:
            rule_type: 规则类型。
            value: 规则主体内容。
            context: 当前消息的判定上下文。

        Returns:
            当前规则是否命中。

        Raises:
            ValueError: 当规则类型不受支持或表达式执行失败时抛出。
        """

        if rule_type == "keyword":
            return value.lower() in context.normalized_text
        if rule_type == "suffix":
            return context.normalized_text.endswith(value.lower())
        if rule_type == "expr":
            return self._evaluator.evaluate(value, context)
        raise ValueError(f"unsupported chat opt-out rule type: {rule_type}")


_chat_opt_out_manager = ChatOptOutManager()


def get_chat_opt_out_manager() -> ChatOptOutManager:
    """返回群关键词免触发管理器单例。

    Returns:
        进程级复用的免触发管理器实例。
    """

    return _chat_opt_out_manager
