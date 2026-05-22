from __future__ import annotations

from .config import get_python_expression_solver_plugin_config


def _build_tool_description(max_user_limit: int) -> str:
    """构造提供给 Agent 的动态工具描述文本。

    该描述会在插件注册阶段写回到工具对象上，让 Agent 在看到工具定义时就能
    知道当前用户配置允许的最大返回长度上限，避免它在超限场景下盲目请求过长
    输出。

    Args:
        max_user_limit: 当前配置文件中允许用户设置的最大返回长度上限。

    Returns:
        适合直接写入工具 `description` 字段的中文说明文本。
    """

    return (
        "求值受限 Python 表达式，只开放安全白名单环境。"
        "支持以下库及导入关键字：`math`->`math`、`cmath`->`cmath`、"
        "`statistics`->`statistics`、`fractions`->`fractions`、"
        "`decimal`->`decimal`、`mpmath`->`mpmath`。"
        "同时支持少量安全内建函数：`abs`、`round`、`pow`、`min`、`max`、`sum`、`len`。"
        "支持常见表达式语法，包括算术、比较、布尔运算、三元条件表达式、下标、切片、"
        "成员测试以及位运算。"
        "禁止导入、文件读写、推导式、Lambda、任意属性链与其他非表达式执行。"
        f"参数 `max_output_length` 表示本次希望返回的最大字符数，默认 50，最高上限为 {max_user_limit}。"
        "如果结果字符串长度超过本次上限，工具不会截断，而是直接返回超限错误信息。"
    )


def register(registry) -> None:  # noqa: ANN001
    """注册 Python 表达式求解器工具。

    注册时会先确保配置文件存在并读取当前上限配置，然后把动态描述写回工具对象，
    让 Agent 在工具元数据里直接看到当前允许的最大返回长度限制。

    Args:
        registry: GTBot 插件注册器，用于接收本插件暴露的工具定义。
    """

    cfg = get_python_expression_solver_plugin_config()

    from .tool import solve_python_expression

    solve_python_expression.description = _build_tool_description(
        int(cfg.max_user_result_length_cap)
    )
    registry.add_tool(solve_python_expression)
