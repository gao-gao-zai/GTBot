from __future__ import annotations

import ast
import cmath
import decimal
import fractions
import math
import mpmath
import statistics
from types import ModuleType
from typing import Any

from langchain.tools import tool

from .config import get_python_expression_solver_plugin_config

_DEFAULT_MAX_OUTPUT_LENGTH = 50
_ALLOWED_MODULES: dict[str, ModuleType] = {
    "math": math,
    "cmath": cmath,
    "statistics": statistics,
    "fractions": fractions,
    "decimal": decimal,
    "mpmath": mpmath,
}
_ALLOWED_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "pow": pow,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
}
_ALLOWED_BINARY_OPERATORS: tuple[type[ast.operator], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
)
_ALLOWED_UNARY_OPERATORS: tuple[type[ast.unaryop], ...] = (
    ast.UAdd,
    ast.USub,
    ast.Not,
)
_ALLOWED_BOOLEAN_OPERATORS: tuple[type[ast.boolop], ...] = (
    ast.And,
    ast.Or,
)
_ALLOWED_COMPARISON_OPERATORS: tuple[type[ast.cmpop], ...] = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
)
_ALLOWED_LITERAL_CONSTANT_TYPES = (int, float, complex, bool, str, bytes, type(None))


def _public_module_attrs(module: ModuleType) -> dict[str, Any]:
    """提取模块中允许被 Agent 访问的公开属性。

    当前规则只放行不以下划线开头的模块级导出，后续求值时还会额外限制属性访问只能
    发生在白名单模块对象本身上，因此不会出现“任意对象属性链”逃逸。

    Args:
        module: 需要暴露的白名单模块。

    Returns:
        模块公开属性名到对象的映射。
    """

    return {
        name: getattr(module, name)
        for name in dir(module)
        if not name.startswith("_")
    }


_ALLOWED_MODULE_ATTRS: dict[str, dict[str, Any]] = {
    module_name: _public_module_attrs(module)
    for module_name, module in _ALLOWED_MODULES.items()
}
_EVAL_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    **_ALLOWED_BUILTINS,
    **_ALLOWED_MODULES,
}


class SafeExpressionValidator(ast.NodeVisitor):
    """校验表达式 AST 是否满足受限求值规则。

    该校验器只允许有限的表达式节点类型，目标是支持常见数学与字面量表达式，而不
    引入导入、赋值、推导式、任意函数定义、复杂属性链等会扩大攻击面的能力。
    """

    def visit_Expression(self, node: ast.Expression) -> None:
        """校验表达式根节点。

        Args:
            node: AST 根节点。
        """

        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        """校验字面量常量类型是否被允许。

        Args:
            node: 字面量 AST 节点。

        Raises:
            ValueError: 当常量类型不在白名单中时抛出。
        """

        if not isinstance(node.value, _ALLOWED_LITERAL_CONSTANT_TYPES):
            raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")

    def visit_Name(self, node: ast.Name) -> None:
        """校验名称引用是否属于白名单环境。

        Args:
            node: 名称 AST 节点。

        Raises:
            ValueError: 当名称不在白名单中时抛出。
        """

        if node.id not in _EVAL_GLOBALS:
            raise ValueError(f"不允许访问名称: {node.id}")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """校验二元运算符及其两侧子表达式。

        Args:
            node: 二元运算 AST 节点。

        Raises:
            ValueError: 当运算符不在白名单中时抛出。
        """

        if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
            raise ValueError(f"不支持的二元运算: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """校验一元运算符及其操作数。

        Args:
            node: 一元运算 AST 节点。

        Raises:
            ValueError: 当运算符不在白名单中时抛出。
        """

        if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
            raise ValueError(f"不支持的一元运算: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """校验布尔运算及其所有子表达式。

        该校验用于支持条件表达式中的 `and`、`or` 组合逻辑，但仍只允许递归引用
        已经放行的安全子表达式，不会因此放开函数定义、推导式等其他结构。

        Args:
            node: 布尔运算 AST 节点。

        Raises:
            ValueError: 当布尔运算符不在白名单中时抛出。
        """

        if not isinstance(node.op, _ALLOWED_BOOLEAN_OPERATORS):
            raise ValueError(f"不支持的布尔运算: {type(node.op).__name__}")
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> None:
        """校验比较表达式及其比较链。

        这里允许常见的大小与相等性比较，也允许 Python 原生的链式比较语法，例如
        `1 < x <= 10`。每个比较符与参与比较的子表达式都必须逐一通过白名单校验。

        Args:
            node: 比较表达式 AST 节点。

        Raises:
            ValueError: 当比较运算符不在白名单中时抛出。
        """

        self.visit(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=False):
            if not isinstance(operator, _ALLOWED_COMPARISON_OPERATORS):
                raise ValueError(f"不支持的比较运算: {type(operator).__name__}")
            self.visit(comparator)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """校验三元条件表达式。

        当前仅支持 Python 表达式级条件分支，即 `a if cond else b`。条件部分与两个
        分支都必须继续满足白名单规则，因此不会额外引入语句级控制流能力。

        Args:
            node: 条件表达式 AST 节点。
        """

        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """校验下标与切片访问。

        当前允许对白名单环境中已经合法的表达式结果继续做索引或切片访问，例如
        `items[0]`、`text[1:3]`、`mapping["key"]`。该能力不会放开任意属性访问，
        只是在既有安全表达式之上补充常见容器读取语法。

        Args:
            node: 下标访问 AST 节点。
        """

        self.visit(node.value)
        self.visit(node.slice)

    def visit_Slice(self, node: ast.Slice) -> None:
        """校验切片表达式。

        Args:
            node: 切片 AST 节点。
        """

        if node.lower is not None:
            self.visit(node.lower)
        if node.upper is not None:
            self.visit(node.upper)
        if node.step is not None:
            self.visit(node.step)

    def visit_Call(self, node: ast.Call) -> None:
        """校验函数调用节点。

        仅允许调用白名单内建函数，或白名单模块上的公开属性。关键字参数也允许，但
        每个关键字值仍必须递归通过校验。

        Args:
            node: 函数调用 AST 节点。

        Raises:
            ValueError: 当调用目标不是白名单函数时抛出。
        """

        if isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_BUILTINS:
                raise ValueError(f"不允许调用函数: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            self._validate_module_attribute(node.func)
        else:
            raise ValueError("只允许调用白名单函数或白名单模块属性")

        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("不允许使用 **kwargs 展开")
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """校验属性访问节点。

        当前只允许 `math.sin` 这一类“白名单模块.公开属性”的单层访问，不允许在任意
        求值结果上继续取属性，也不允许访问以下划线开头的成员。

        Args:
            node: 属性访问 AST 节点。

        Raises:
            ValueError: 当属性访问超出白名单时抛出。
        """

        self._validate_module_attribute(node)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """校验元组字面量。

        Args:
            node: 元组 AST 节点。
        """

        for elt in node.elts:
            self.visit(elt)

    def visit_List(self, node: ast.List) -> None:
        """校验列表字面量。

        Args:
            node: 列表 AST 节点。
        """

        for elt in node.elts:
            self.visit(elt)

    def visit_Set(self, node: ast.Set) -> None:
        """校验集合字面量。

        Args:
            node: 集合 AST 节点。
        """

        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node: ast.Dict) -> None:
        """校验字典字面量。

        Args:
            node: 字典 AST 节点。
        """

        for key in node.keys:
            if key is not None:
                self.visit(key)
        for value in node.values:
            self.visit(value)

    def generic_visit(self, node: ast.AST) -> None:
        """拒绝所有未显式放行的 AST 节点。

        Args:
            node: 任意 AST 节点。

        Raises:
            ValueError: 当节点类型未在白名单中时抛出。
        """

        raise ValueError(f"不支持的表达式结构: {type(node).__name__}")

    def _validate_module_attribute(self, node: ast.Attribute) -> None:
        """校验属性访问是否是合法的白名单模块公开属性。

        Args:
            node: 属性访问 AST 节点。

        Raises:
            ValueError: 当属性访问的基对象不是白名单模块，或属性名不允许时抛出。
        """

        if not isinstance(node.value, ast.Name):
            raise ValueError("只允许访问白名单模块的单层公开属性")

        module_name = node.value.id
        attr_name = node.attr
        if module_name not in _ALLOWED_MODULE_ATTRS:
            raise ValueError(f"不允许访问模块属性: {module_name}.{attr_name}")
        if attr_name.startswith("_"):
            raise ValueError(f"不允许访问私有属性: {module_name}.{attr_name}")
        if attr_name not in _ALLOWED_MODULE_ATTRS[module_name]:
            raise ValueError(f"模块属性不存在或未开放: {module_name}.{attr_name}")


def _normalize_max_output_length(
    requested_limit: int,
    configured_cap: int,
) -> int:
    """规范化本次调用的结果长度上限。

    Agent 可以把单次结果上限设置得更小，但不能超过配置文件给出的用户级硬上限。
    该函数统一负责边界校验，避免主工具函数混杂业务校验细节。

    Args:
        requested_limit: Agent 本次请求的最大返回长度。
        configured_cap: 配置文件定义的允许上限。

    Returns:
        通过校验后的本次最大返回长度。

    Raises:
        ValueError: 当请求上限不是正整数，或超过配置上限时抛出。
    """

    normalized_requested_limit = int(requested_limit)
    normalized_configured_cap = int(configured_cap)
    if normalized_requested_limit <= 0:
        raise ValueError("max_output_length 必须大于 0")
    if normalized_requested_limit > normalized_configured_cap:
        raise ValueError(
            "max_output_length 超过当前用户配置允许的最大上限: "
            f"{normalized_configured_cap}"
        )
    return normalized_requested_limit


def _safe_eval_expression(expression: str) -> Any:
    """在受限环境中求值 Python 表达式。

    该函数先做 AST 级白名单校验，再使用空内建环境执行 `eval`。只要白名单没有放
    开新的节点或对象访问链，表达式就无法触达导入、文件系统和宿主运行时对象。

    Args:
        expression: 待求值的 Python 表达式文本。

    Returns:
        表达式求值结果。

    Raises:
        ValueError: 当表达式为空、语法非法或包含不允许的结构时抛出。
    """

    text = str(expression).strip()
    if not text:
        raise ValueError("expression 不能为空")

    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc.msg}") from exc

    SafeExpressionValidator().visit(parsed)
    compiled = compile(parsed, "<python_expression_solver>", "eval")
    return eval(compiled, _EVAL_GLOBALS, {})  # noqa: S307


def solve_python_expression_impl(
    expression: str,
    *,
    max_output_length: int = _DEFAULT_MAX_OUTPUT_LENGTH,
) -> str:
    """求解受限 Python 表达式并返回字符串结果。

    结果不会被截断；如果字符串化后的长度超过本次允许上限，会直接返回明确的超限
    错误信息，便于 Agent 自行缩小范围、改写表达式或降低期望输出长度。

    Args:
        expression: 待求值的 Python 表达式。
        max_output_length: 本次调用允许返回的最大字符数，默认 50。

    Returns:
        成功时返回表达式结果字符串；失败时返回可直接展示给调用方的错误说明。
    """

    cfg = get_python_expression_solver_plugin_config()
    if not bool(cfg.enabled):
        return "python_expression_solver 插件当前已禁用。"

    try:
        allowed_length = _normalize_max_output_length(
            requested_limit=max_output_length,
            configured_cap=int(cfg.max_user_result_length_cap),
        )
        result = _safe_eval_expression(expression)
        result_text = str(result)
        if len(result_text) > allowed_length:
            return (
                "结果超出本次允许的最大返回长度。"
                f"result_length={len(result_text)} limit={allowed_length}"
            )
        return result_text
    except ValueError as exc:
        return str(exc)


@tool("solve_python_expression")
def solve_python_expression(
    expression: str,
    max_output_length: int = _DEFAULT_MAX_OUTPUT_LENGTH,
) -> str:
    """求值受限 Python 表达式并返回结果字符串。

    该工具只开放白名单表达式能力，适合数学计算、常量推导和少量安全模块调用。
    当前支持 `math`、`cmath`、`statistics`、`fractions`、`decimal`、`mpmath`
    模块，以及 `abs`、`round`、`pow`、`min`、`max`、`sum`、`len` 等安全内建函数。
    `max_output_length` 表示 Agent 希望本次最多拿到多少字符的结果，默认 `50`，
    且不能超过当前用户配置允许的最大返回长度上限。若结果过长，工具会返回错误
    文本而不是截断结果。

    Args:
        expression: 待求值的 Python 表达式，例如 `math.sqrt(2)` 或 `sum([1, 2, 3])`。
        max_output_length: 本次调用允许的最大返回字符数，默认 50，不能超过配置上限。

    Returns:
        成功时返回结果字符串；失败时返回错误文本，便于 Agent 调整表达式或请求参数。
    """

    return solve_python_expression_impl(
        expression,
        max_output_length=max_output_length,
    )
