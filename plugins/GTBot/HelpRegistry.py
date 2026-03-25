from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from .PermissionManager import PermissionRole, get_permission_manager


def _normalize_command_token(token: str) -> str:
    """将命令名或别名归一化为统一索引键。

    Args:
        token: 原始命令文本，允许包含前导 `/` 或大小写差异。

    Returns:
        str: 去除前导斜杠并转为小写后的索引键。
    """
    normalized = str(token).strip().lower()
    while normalized.startswith("/"):
        normalized = normalized[1:]
    return normalized


def _normalize_category_token(token: str) -> str:
    """将命令集合名称归一化为统一索引键。

    Args:
        token: 原始集合名称。

    Returns:
        str: 去除首尾空白并转为小写后的集合键。
    """
    return str(token).strip().lower()


@dataclass(frozen=True, slots=True)
class HelpArgumentSpec:
    """描述单个帮助参数的结构化信息。

    Attributes:
        name: 参数名称，建议与命令示例中的占位符保持一致。
        description: 参数用途说明。
        required: 是否为必填参数。
        value_hint: 参数值形态或类型提示，例如 `QQ号`、`群聊|私聊`。
        example: 参数示例值，用于帮助详情展示。
    """

    name: str
    description: str
    required: bool = True
    value_hint: str = ""
    example: str = ""


@dataclass(frozen=True, slots=True)
class HelpCommandSpec:
    """描述单条命令帮助信息的结构化定义。

    Attributes:
        name: 主命令名，不包含前导 `/`。
        category: 帮助菜单中的分组名称。
        summary: 菜单列表里展示的一句话简介。
        description: 命令详情说明。
        aliases: 可用于查询详情的别名列表。
        arguments: 参数定义列表。
        examples: 示例命令列表。
        required_role: 查看和执行该命令所需的最低权限。
        audience: 命令适用场景说明，例如群聊、私聊或管理员后台。
        sort_key: 同分组内排序权重，值越小越靠前。
    """

    name: str
    category: str
    summary: str
    description: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    arguments: tuple[HelpArgumentSpec, ...] = field(default_factory=tuple)
    examples: tuple[str, ...] = field(default_factory=tuple)
    required_role: PermissionRole = PermissionRole.USER
    audience: str = "通用"
    sort_key: int = 100

    @property
    def normalized_name(self) -> str:
        """返回主命令名的标准化索引键。"""
        return _normalize_command_token(self.name)

    @property
    def normalized_aliases(self) -> tuple[str, ...]:
        """返回别名列表的标准化索引键。"""
        return tuple(_normalize_command_token(alias) for alias in self.aliases)

    @property
    def all_tokens(self) -> tuple[str, ...]:
        """返回主命令名与别名合并后的全部索引键。"""
        return (self.normalized_name, *self.normalized_aliases)


class HelpRegistry:
    """统一管理 GTBot 结构化帮助项。

    该注册中心负责三类职责：
    1. 校验并登记命令帮助项，避免命令名和别名冲突。
    2. 根据权限过滤当前用户可见的帮助项。
    3. 提供列表查询、详情查询和模糊建议能力。
    """

    def __init__(self) -> None:
        """初始化空的帮助注册中心。"""
        self._specs_by_name: dict[str, HelpCommandSpec] = {}
        self._token_to_name: dict[str, str] = {}

    def register(self, spec: HelpCommandSpec) -> HelpCommandSpec:
        """注册一条帮助项。

        Args:
            spec: 要注册的命令帮助定义。

        Returns:
            HelpCommandSpec: 原样返回已注册的帮助定义，便于链式调用。

        Raises:
            ValueError: 主命令名为空，或命令名/别名与已存在帮助项冲突。
        """
        normalized_name = spec.normalized_name
        if not normalized_name:
            raise ValueError("帮助命令名不能为空")

        existing = self._specs_by_name.get(normalized_name)
        if existing is not None:
            raise ValueError(f"帮助命令已存在: {spec.name}")

        for token in spec.all_tokens:
            if not token:
                raise ValueError(f"帮助命令存在空别名: {spec.name}")
            conflict_name = self._token_to_name.get(token)
            if conflict_name is not None:
                raise ValueError(f"帮助命令或别名冲突: {token}")

        self._specs_by_name[normalized_name] = spec
        for token in spec.all_tokens:
            self._token_to_name[token] = normalized_name
        return spec

    def get_all_specs(self) -> list[HelpCommandSpec]:
        """返回全部已注册帮助项。

        Returns:
            list[HelpCommandSpec]: 按分组和排序键稳定排序后的帮助项列表。
        """
        return sorted(
            self._specs_by_name.values(),
            key=lambda item: (item.category, item.sort_key, item.normalized_name),
        )

    async def get_visible_specs(self, user_id: int) -> list[HelpCommandSpec]:
        """返回当前用户可见的帮助项列表。

        Args:
            user_id: 请求帮助的用户 QQ 号。

        Returns:
            list[HelpCommandSpec]: 经过权限过滤后的帮助项列表。
        """
        permission_manager = get_permission_manager()
        visible_specs: list[HelpCommandSpec] = []
        for spec in self.get_all_specs():
            if await permission_manager.has_role(int(user_id), spec.required_role):
                visible_specs.append(spec)
        return visible_specs

    async def get_visible_categories(self, user_id: int) -> list[tuple[str, list[HelpCommandSpec]]]:
        """返回当前用户可见的命令集合及其下属命令。

        Args:
            user_id: 请求帮助的用户 QQ 号。

        Returns:
            list[tuple[str, list[HelpCommandSpec]]]:
                以 `(集合名, 命令列表)` 形式返回的可见分组结果。
        """
        visible_specs = await self.get_visible_specs(user_id)
        grouped: dict[str, list[HelpCommandSpec]] = {}
        for spec in visible_specs:
            grouped.setdefault(spec.category, []).append(spec)

        categories = sorted(
            grouped.items(),
            key=lambda item: (
                min(spec.sort_key for spec in item[1]) if item[1] else 10**9,
                _normalize_category_token(item[0]),
            ),
        )
        return categories

    async def find_visible_category(self, query: str, user_id: int) -> tuple[str, list[HelpCommandSpec]] | None:
        """按集合名查询当前用户可见的命令集合。

        Args:
            query: 用户输入的集合名称。
            user_id: 请求帮助的用户 QQ 号。

        Returns:
            tuple[str, list[HelpCommandSpec]] | None:
                命中的集合名及其可见命令列表；未命中时返回 `None`。
        """
        normalized_query = _normalize_category_token(query)
        if not normalized_query:
            return None

        for category, specs in await self.get_visible_categories(user_id):
            if _normalize_category_token(category) == normalized_query:
                return category, specs
        return None

    async def find_visible_spec(self, query: str, user_id: int) -> HelpCommandSpec | None:
        """按命令名或别名查询当前用户可见的帮助项。

        Args:
            query: 用户输入的命令名或别名。
            user_id: 请求帮助的用户 QQ 号。

        Returns:
            HelpCommandSpec | None: 命中的帮助项；无命中或无权限时返回 `None`。
        """
        normalized_query = _normalize_command_token(query)
        if not normalized_query:
            return None

        spec = self.find_any_spec(query)
        if spec is None:
            return None

        permission_manager = get_permission_manager()
        if not await permission_manager.has_role(int(user_id), spec.required_role):
            return None
        return spec

    def find_any_spec(self, query: str) -> HelpCommandSpec | None:
        """按命令名或别名查询帮助项，不做权限过滤。

        Args:
            query: 用户输入的命令名或别名。

        Returns:
            HelpCommandSpec | None: 命中的帮助项；没有命中时返回 `None`。
        """
        normalized_query = _normalize_command_token(query)
        if not normalized_query:
            return None

        normalized_name = self._token_to_name.get(normalized_query)
        if normalized_name is None:
            return None
        return self._specs_by_name.get(normalized_name)

    async def suggest_visible_commands(self, query: str, user_id: int, limit: int = 3) -> list[str]:
        """基于当前用户可见命令生成模糊匹配建议。

        Args:
            query: 用户输入的目标命令名。
            user_id: 请求帮助的用户 QQ 号。
            limit: 最多返回的建议条数。

        Returns:
            list[str]: 建议命令名列表，使用主命令名输出。
        """
        normalized_query = _normalize_command_token(query)
        if not normalized_query:
            return []

        visible_specs = await self.get_visible_specs(user_id)
        token_to_display_name: dict[str, str] = {}
        for spec in visible_specs:
            for token in spec.all_tokens:
                token_to_display_name[token] = spec.name

        matched_tokens = get_close_matches(
            normalized_query,
            list(token_to_display_name.keys()),
            n=max(1, int(limit)),
            cutoff=0.4,
        )

        suggestions: list[str] = []
        for token in matched_tokens:
            display_name = token_to_display_name[token]
            if display_name not in suggestions:
                suggestions.append(display_name)
        return suggestions

    async def suggest_visible_categories(self, query: str, user_id: int, limit: int = 3) -> list[str]:
        """基于当前用户可见命令集合生成模糊匹配建议。

        Args:
            query: 用户输入的目标集合名。
            user_id: 请求帮助的用户 QQ 号。
            limit: 最多返回的建议条数。

        Returns:
            list[str]: 建议的集合名称列表。
        """
        normalized_query = _normalize_category_token(query)
        if not normalized_query:
            return []

        visible_categories = await self.get_visible_categories(user_id)
        category_names = [category for category, _ in visible_categories]
        normalized_to_display = {
            _normalize_category_token(category): category
            for category in category_names
        }
        matched_tokens = get_close_matches(
            normalized_query,
            list(normalized_to_display.keys()),
            n=max(1, int(limit)),
            cutoff=0.4,
        )
        return [normalized_to_display[token] for token in matched_tokens]


def _format_role_label(role: PermissionRole) -> str:
    """将权限枚举格式化为中文标签。

    Args:
        role: 目标权限等级。

    Returns:
        str: 中文权限名称。
    """
    labels = {
        PermissionRole.USER: "普通用户",
        PermissionRole.ADMIN: "管理员",
        PermissionRole.OWNER: "所有者",
    }
    return labels[role]


def _split_rendered_text(text: str, max_length: int = 900) -> list[str]:
    """按段落长度拆分帮助文本，避免单条消息过长。

    Args:
        text: 已渲染完成的完整帮助文本。
        max_length: 单段文本的最大字符数。

    Returns:
        list[str]: 适合逐条发送的文本片段列表。
    """
    normalized_text = str(text).strip()
    if not normalized_text:
        return []
    if len(normalized_text) <= max_length:
        return [normalized_text]

    paragraphs = normalized_text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_length:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_length:
            current = paragraph
            continue

        lines = paragraph.splitlines()
        line_buffer = ""
        for line in lines:
            candidate_line = line if not line_buffer else f"{line_buffer}\n{line}"
            if len(candidate_line) <= max_length:
                line_buffer = candidate_line
                continue
            if line_buffer:
                chunks.append(line_buffer)
            line_buffer = line
        if line_buffer:
            current = line_buffer

    if current:
        chunks.append(current)
    return chunks


def render_help_menu(specs: list[HelpCommandSpec]) -> list[str]:
    """将帮助项列表渲染为分组菜单文本。

    Args:
        specs: 已按权限过滤后的帮助项列表。

    Returns:
        list[str]: 可直接发送给 QQ 用户的文本分片列表。
    """
    if not specs:
        return ["当前没有可查看的帮助命令。"]

    lines: list[str] = [
        "GTBot 帮助菜单",
        "发送 /帮助 <命令名> 可查看详细用法。",
    ]

    current_category = ""
    for spec in specs:
        if spec.category != current_category:
            current_category = spec.category
            lines.append("")
            lines.append(f"【{current_category}】")
        lines.append(f"/{spec.name} - {spec.summary}")

    return _split_rendered_text("\n".join(lines))


def _format_command_signature(spec: HelpCommandSpec) -> str:
    """将命令和参数列表格式化为简洁签名。

    Args:
        spec: 目标帮助项。

    Returns:
        str: 带参数占位符的命令签名。
    """
    parts = [f"/{spec.name}"]
    for argument in spec.arguments:
        parts.append(argument.name)
    return " ".join(parts)


def render_help_categories(categories: list[tuple[str, list[HelpCommandSpec]]]) -> list[str]:
    """将命令集合列表渲染为帮助第一层菜单。

    Args:
        categories: 当前用户可见的命令集合及其命令列表。

    Returns:
        list[str]: 可直接发送给 QQ 用户的文本分片列表。
    """
    if not categories:
        return ["当前没有可查看的帮助命令。"]

    lines: list[str] = [
        "GTBot 命令集合",
        "发送 /帮助 <命令集合> 查看该集合下的命令。",
    ]
    for category, specs in categories:
        lines.append(f"- {category} ({len(specs)} 条)")

    return _split_rendered_text("\n".join(lines))


def render_help_category(category: str, specs: list[HelpCommandSpec]) -> list[str]:
    """将单个命令集合渲染为第二层命令列表。

    Args:
        category: 目标集合名称。
        specs: 该集合下已按权限过滤的命令列表。

    Returns:
        list[str]: 可直接发送给 QQ 用户的文本分片列表。
    """
    if not specs:
        return [f"命令集合“{category}”下当前没有可查看的命令。"]

    lines: list[str] = [
        f"命令集合: {category}",
        "以下仅展示命令名和参数格式。",
        "如需查看某条命令的完整说明，请发送 /帮助 <命令名>。",
    ]

    for spec in specs:
        lines.append("")
        lines.append(_format_command_signature(spec))
        lines.append(f"说明: {spec.summary}")

    return _split_rendered_text("\n".join(lines))


def render_help_detail(spec: HelpCommandSpec) -> list[str]:
    """将单条帮助项渲染为命令详情文本。

    Args:
        spec: 目标命令帮助项。

    Returns:
        list[str]: 可直接发送的文本分片列表。
    """
    lines: list[str] = [
        f"命令: /{spec.name}",
        f"分组: {spec.category}",
        f"简介: {spec.summary}",
        f"说明: {spec.description}",
        f"权限: {_format_role_label(spec.required_role)}",
        f"适用场景: {spec.audience}",
    ]

    if spec.aliases:
        lines.append(f"别名: {', '.join('/' + alias.lstrip('/') for alias in spec.aliases)}")

    if spec.arguments:
        lines.append("")
        lines.append("参数:")
        for argument in spec.arguments:
            required_text = "必填" if argument.required else "可选"
            hint_text = f" ({argument.value_hint})" if argument.value_hint else ""
            example_text = f"；示例: {argument.example}" if argument.example else ""
            lines.append(
                f"- {argument.name}{hint_text} [{required_text}]：{argument.description}{example_text}"
            )

    if spec.examples:
        lines.append("")
        lines.append("示例:")
        for example in spec.examples:
            lines.append(f"- {example}")

    return _split_rendered_text("\n".join(lines))


_help_registry = HelpRegistry()


def get_help_registry() -> HelpRegistry:
    """返回共享的帮助注册中心实例。

    Returns:
        HelpRegistry: 全局帮助注册器。
    """
    return _help_registry


def register_help(spec: HelpCommandSpec) -> HelpCommandSpec:
    """向全局帮助注册中心注册帮助项。

    Args:
        spec: 待注册的帮助项定义。

    Returns:
        HelpCommandSpec: 已注册的帮助项定义。
    """
    return _help_registry.register(spec)
