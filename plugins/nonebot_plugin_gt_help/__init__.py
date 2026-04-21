from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from nonebot import get_driver, on_command
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata

from local_plugins.nonebot_plugin_gt_permission import PermissionRole, get_permission_config, has_role

__plugin_meta__ = PluginMetadata(
    name="GT Help",
    description="提供结构化帮助项注册、帮助菜单渲染与按权限过滤的帮助查询命令。",
    usage="发送 /帮助 查看命令集合，发送 /帮助 <命令集合|命令名> 查看更详细的用法。",
    type="application",
    supported_adapters={"nonebot.adapters.onebot.v11"},
)


def _normalize_command_token(token: str) -> str:
    """将命令名或别名归一化为统一检索键。

    Args:
        token: 原始命令文本，允许包含 `/` 前缀和大小写差异。

    Returns:
        去除命令前导斜杠并转为小写后的检索键。
    """

    normalized = str(token).strip().lower()
    while normalized.startswith("/"):
        normalized = normalized[1:]
    return normalized


def _normalize_category_token(token: str) -> str:
    """将命令集合名称归一化为统一检索键。

    Args:
        token: 原始命令集合名称。

    Returns:
        去除首尾空白并转为小写后的集合检索键。
    """

    return str(token).strip().lower()


@dataclass(frozen=True, slots=True)
class HelpArgumentSpec:
    """描述命令单个参数的结构化信息。

    该结构仅承载帮助展示所需的静态元数据，不负责参数解析。
    `name` 建议与命令签名中的占位符保持一致，以便帮助页直接复用。
    """

    name: str
    description: str
    required: bool = True
    value_hint: str = ""
    example: str = ""


@dataclass(frozen=True, slots=True)
class HelpCommandSpec:
    """描述单条命令帮助项的结构化定义。

    帮助插件不会自动扫描 matcher，而是要求宿主显式注册帮助项。
    这样能避免命令实现、权限策略和帮助文案之间隐式耦合，也方便外部插件单独接入。
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
        """返回主命令名的标准化检索键。

        Returns:
            适用于帮助检索和冲突校验的标准化命令键。
        """

        return _normalize_command_token(self.name)

    @property
    def normalized_aliases(self) -> tuple[str, ...]:
        """返回全部别名的标准化检索键。

        Returns:
            由标准化别名组成的元组。
        """

        return tuple(_normalize_command_token(alias) for alias in self.aliases)

    @property
    def all_tokens(self) -> tuple[str, ...]:
        """返回主命令和别名合并后的全部检索键。

        Returns:
            由主命令键和别名键组成的元组。
        """

        return (self.normalized_name, *self.normalized_aliases)


class HelpRegistry:
    """管理全局结构化帮助项的注册、检索与权限过滤。

    该注册中心不关心命令具体实现，只维护帮助元数据。
    所有“用户是否可见”的判断统一委托给权限插件，以保证帮助系统和执行权限保持一致。
    """

    def __init__(self) -> None:
        """初始化空的帮助注册中心。"""

        self._specs_by_name: dict[str, HelpCommandSpec] = {}
        self._token_to_name: dict[str, str] = {}

    def register(self, spec: HelpCommandSpec) -> HelpCommandSpec:
        """注册一条帮助项。

        Args:
            spec: 待注册的帮助项定义。

        Returns:
            原样返回已注册的帮助项，便于调用侧链式使用。

        Raises:
            ValueError: 当命令名为空、别名为空或与现有命令冲突时抛出。
        """

        normalized_name = spec.normalized_name
        if not normalized_name:
            raise ValueError("帮助命令名不能为空。")
        if normalized_name in self._specs_by_name:
            raise ValueError(f"帮助命令已存在: {spec.name}")

        for token in spec.all_tokens:
            if not token:
                raise ValueError(f"帮助命令存在空别名: {spec.name}")
            if token in self._token_to_name:
                raise ValueError(f"帮助命令或别名冲突: {token}")

        self._specs_by_name[normalized_name] = spec
        for token in spec.all_tokens:
            self._token_to_name[token] = normalized_name
        return spec

    def get_all_specs(self) -> list[HelpCommandSpec]:
        """返回全部已注册帮助项。

        Returns:
            按分组、排序键和命令名稳定排序后的帮助项列表。
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
            经过权限过滤后的帮助项列表。
        """

        visible_specs: list[HelpCommandSpec] = []
        for spec in self.get_all_specs():
            if await has_role(int(user_id), spec.required_role):
                visible_specs.append(spec)
        return visible_specs

    async def get_visible_categories(self, user_id: int) -> list[tuple[str, list[HelpCommandSpec]]]:
        """返回当前用户可见的命令集合及其命令列表。

        Args:
            user_id: 请求帮助的用户 QQ 号。

        Returns:
            形如 `(分类名, 帮助项列表)` 的有序集合列表。
        """

        visible_specs = await self.get_visible_specs(user_id)
        grouped: dict[str, list[HelpCommandSpec]] = {}
        for spec in visible_specs:
            grouped.setdefault(spec.category, []).append(spec)
        return sorted(
            grouped.items(),
            key=lambda item: (
                min(spec.sort_key for spec in item[1]) if item[1] else 10**9,
                _normalize_category_token(item[0]),
            ),
        )

    async def find_visible_category(self, query: str, user_id: int) -> tuple[str, list[HelpCommandSpec]] | None:
        """查找当前用户可见的命令集合。

        Args:
            query: 用户输入的命令集合名称。
            user_id: 请求帮助的用户 QQ 号。

        Returns:
            命中的集合名和帮助项列表；未命中时返回 `None`。
        """

        normalized_query = _normalize_category_token(query)
        if not normalized_query:
            return None
        for category, specs in await self.get_visible_categories(user_id):
            if _normalize_category_token(category) == normalized_query:
                return category, specs
        return None

    async def find_visible_spec(self, query: str, user_id: int) -> HelpCommandSpec | None:
        """查找当前用户可见的单条帮助项。

        Args:
            query: 用户输入的命令名或别名。
            user_id: 请求帮助的用户 QQ 号。

        Returns:
            命中的帮助项；未命中或无权查看时返回 `None`。
        """

        spec = self.find_any_spec(query)
        if spec is None:
            return None
        if not await has_role(int(user_id), spec.required_role):
            return None
        return spec

    def find_any_spec(self, query: str) -> HelpCommandSpec | None:
        """在不做权限过滤的前提下查找帮助项。

        Args:
            query: 用户输入的命令名或别名。

        Returns:
            命中的帮助项；未命中时返回 `None`。
        """

        normalized_query = _normalize_command_token(query)
        if not normalized_query:
            return None
        normalized_name = self._token_to_name.get(normalized_query)
        if normalized_name is None:
            return None
        return self._specs_by_name.get(normalized_name)

    async def suggest_visible_commands(self, query: str, user_id: int, limit: int = 3) -> list[str]:
        """基于当前用户可见命令生成模糊建议。

        Args:
            query: 用户输入的近似命令名。
            user_id: 请求帮助的用户 QQ 号。
            limit: 最多返回的建议数量。

        Returns:
            建议命令名列表，使用主命令名输出。
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
        """基于当前用户可见命令集合生成模糊建议。

        Args:
            query: 用户输入的近似集合名。
            user_id: 请求帮助的用户 QQ 号。
            limit: 最多返回的建议数量。

        Returns:
            建议集合名列表。
        """

        normalized_query = _normalize_category_token(query)
        if not normalized_query:
            return []
        visible_categories = await self.get_visible_categories(user_id)
        normalized_to_display = {
            _normalize_category_token(category): category
            for category, _ in visible_categories
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
        对终端用户友好的中文权限名称。
    """

    labels = {
        PermissionRole.USER: "普通用户",
        PermissionRole.ADMIN: "管理员",
        PermissionRole.OWNER: "所有者",
    }
    return labels[role]


def _split_rendered_text(text: str, max_length: int = 900) -> list[str]:
    """将较长帮助文本按段落拆成多条消息。

    Args:
        text: 完整帮助文本。
        max_length: 单段允许的最大字符数。

    Returns:
        适合逐条发送的文本片段列表。
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
    """将帮助项列表渲染为按分组排列的菜单文本。

    Args:
        specs: 已按权限过滤后的帮助项列表。

    Returns:
        适合直接发送给用户的文本片段列表。
    """

    if not specs:
        return ["当前没有可查看的帮助命令。"]
    lines: list[str] = [
        "帮助菜单",
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
    """将帮助项格式化为命令签名。

    Args:
        spec: 目标帮助项。

    Returns:
        带参数占位符的命令签名字符串。
    """

    parts = [f"/{spec.name}"]
    for argument in spec.arguments:
        parts.append(argument.name)
    return " ".join(parts)


def render_help_categories(categories: list[tuple[str, list[HelpCommandSpec]]]) -> list[str]:
    """将命令集合列表渲染为帮助首页。

    Args:
        categories: 当前用户可见的命令集合及其帮助项列表。

    Returns:
        适合直接发送给用户的文本片段列表。
    """

    if not categories:
        return ["当前没有可查看的帮助命令。"]
    lines: list[str] = [
        "命令集合",
        "发送 /帮助 <命令集合> 查看该集合下的命令。",
    ]
    for category, specs in categories:
        lines.append(f"- {category} ({len(specs)} 条)")
    return _split_rendered_text("\n".join(lines))


def render_help_category(category: str, specs: list[HelpCommandSpec]) -> list[str]:
    """将单个命令集合渲染为第二层命令列表。

    Args:
        category: 目标命令集合名称。
        specs: 该集合下的帮助项列表。

    Returns:
        适合直接发送给用户的文本片段列表。
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
        spec: 目标帮助项。

    Returns:
        适合直接发送给用户的文本片段列表。
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
            lines.append(f"- {argument.name}{hint_text} [{required_text}]：{argument.description}{example_text}")
    if spec.examples:
        lines.append("")
        lines.append("示例:")
        for example in spec.examples:
            lines.append(f"- {example}")
    return _split_rendered_text("\n".join(lines))


_help_registry = HelpRegistry()


def get_help_registry() -> HelpRegistry:
    """返回全局共享的帮助注册中心实例。

    Returns:
        当前进程内共享的帮助注册中心。
    """

    return _help_registry


def register_help(spec: HelpCommandSpec) -> HelpCommandSpec:
    """向全局帮助注册中心注册帮助项。

    Args:
        spec: 待注册的帮助项定义。

    Returns:
        已注册的帮助项定义。
    """

    return _help_registry.register(spec)


def _register_builtin_help_items() -> None:
    """注册帮助插件和权限插件的内置帮助项。

    该函数只在进程内执行一次，避免因插件被重复导入而触发帮助项冲突。
    权限命令帮助项是否注册取决于权限插件配置中的 `enable_commands`。
    """

    if _help_registry.find_any_spec("帮助") is None:
        register_help(
            HelpCommandSpec(
                name="帮助",
                aliases=("菜单", "help"),
                category="帮助系统",
                summary="查看命令集合、集合下命令或单条命令的详细说明。",
                description="留空时优先显示当前可见的命令集合；也可按命令集合或命令名继续向下查询。",
                arguments=(
                    HelpArgumentSpec(
                        name="[命令名]",
                        description="可选的目标命令集合名称、命令名或别名。",
                        required=False,
                        value_hint="命令集合或命令名",
                        example="查看权限",
                    ),
                ),
                examples=("/帮助", "/帮助 权限管理", "/帮助 查看权限"),
                required_role=PermissionRole.USER,
                audience="群聊和私聊",
                sort_key=0,
            )
        )

    if not get_permission_config().enable_commands:
        return
    if _help_registry.find_any_spec("查看权限") is None:
        register_help(
            HelpCommandSpec(
                name="查看权限",
                category="权限管理",
                summary="查看自己或指定用户的权限等级。",
                description="留空时查看自己的权限；管理员和所有者可带 QQ 号查询其他用户。",
                arguments=(
                    HelpArgumentSpec(
                        name="[用户QQ号]",
                        description="可选的目标用户 QQ 号。",
                        required=False,
                        value_hint="整数 QQ 号",
                        example="123456789",
                    ),
                ),
                examples=("/查看权限", "/查看权限 123456789"),
                required_role=PermissionRole.USER,
                audience="群聊和私聊",
                sort_key=10,
            )
        )
        register_help(
            HelpCommandSpec(
                name="查看管理员列表",
                category="权限管理",
                summary="查看所有者与动态管理员列表。",
                description="管理员和所有者可使用该命令确认当前的权限人员名单。",
                examples=("/查看管理员列表",),
                required_role=PermissionRole.ADMIN,
                audience="群聊和私聊",
                sort_key=20,
            )
        )
        register_help(
            HelpCommandSpec(
                name="提拔管理员",
                category="权限管理",
                summary="将指定用户加入管理员列表。",
                description="只有所有者可执行；owner 身份来自静态配置，不能通过该命令重复设置。",
                arguments=(
                    HelpArgumentSpec(
                        name="<用户QQ号>",
                        description="需要提拔为管理员的目标 QQ 号。",
                        value_hint="整数 QQ 号",
                        example="123456789",
                    ),
                ),
                examples=("/提拔管理员 123456789",),
                required_role=PermissionRole.OWNER,
                audience="群聊和私聊",
                sort_key=30,
            )
        )
        register_help(
            HelpCommandSpec(
                name="降级管理员",
                category="权限管理",
                summary="移除指定用户的管理员权限。",
                description="只有所有者可执行；所有者本身由静态配置管理，不能通过该命令降级。",
                arguments=(
                    HelpArgumentSpec(
                        name="<用户QQ号>",
                        description="需要移除管理员权限的目标 QQ 号。",
                        value_hint="整数 QQ 号",
                        example="123456789",
                    ),
                ),
                examples=("/降级管理员 123456789",),
                required_role=PermissionRole.OWNER,
                audience="群聊和私聊",
                sort_key=40,
            )
        )


HelpCommand = None


def _get_help_command():
    """返回已注册的帮助命令 matcher。

    帮助插件允许在 NoneBot 未初始化时被安全导入，因此 `HelpCommand`
    在模块加载早期可能仍为 `None`。真正执行发送或结束消息前，这个函数会
    统一校验命令已经完成注册，既消除类型检查中的可选值告警，也避免后续维护者
    在不同调用点重复编写空值判断。

    Returns:
        已注册完成的帮助命令 matcher 对象。

    Raises:
        RuntimeError: 当帮助命令尚未注册就尝试发送消息时抛出。
    """

    if HelpCommand is None:
        raise RuntimeError("帮助命令尚未注册，无法发送帮助消息。")
    return HelpCommand


async def _send_chunks(chunks: list[str]) -> None:
    """按顺序发送帮助文本分片。

    Args:
        chunks: 已切分好的帮助文本列表。
    """

    help_command = _get_help_command()
    if not chunks:
        await help_command.finish("当前没有可展示的帮助内容。")
    for chunk in chunks[:-1]:
        await help_command.send(chunk)
    await help_command.finish(chunks[-1])


def _register_help_command() -> None:
    """在 NoneBot 已初始化时注册统一帮助命令。"""

    global HelpCommand
    if HelpCommand is not None:
        return
    try:
        get_driver()
    except ValueError:
        return

    HelpCommand = on_command("帮助", aliases={"菜单", "help"}, priority=4, block=True)

    @HelpCommand.handle()
    async def handle_help_command(event: MessageEvent, args: Message = CommandArg()) -> None:
        """处理统一帮助命令。

        Args:
            event: 触发帮助命令的消息事件。
            args: 命令参数，允许为空或传入命令集合/命令名。
        """

        help_command = _get_help_command()
        registry = get_help_registry()
        arg_text = args.extract_plain_text().strip()
        user_id = int(event.user_id)
        if not arg_text:
            await _send_chunks(render_help_categories(await registry.get_visible_categories(user_id)))
        matched_category = await registry.find_visible_category(arg_text, user_id)
        if matched_category is not None:
            category, specs = matched_category
            await _send_chunks(render_help_category(category, specs))
        spec = await registry.find_visible_spec(arg_text, user_id)
        if spec is not None:
            await _send_chunks(render_help_detail(spec))
        category_suggestions = await registry.suggest_visible_categories(arg_text, user_id)
        if category_suggestions:
            await help_command.finish(
                f"未找到命令集合或命令“{arg_text}”，或你无权查看它。\n你可以试试这些命令集合：{'、'.join(category_suggestions)}"
            )
        suggestions = await registry.suggest_visible_commands(arg_text, user_id)
        if suggestions:
            await help_command.finish(
                f"未找到命令集合或命令“{arg_text}”，或你无权查看它。\n你可以试试这些命令：{'、'.join(f'/{name}' for name in suggestions)}"
            )
        await help_command.finish(
            f"未找到命令集合或命令“{arg_text}”，或你无权查看它。\n发送 /帮助 查看当前可用命令集合。"
        )


_register_builtin_help_items()
_register_help_command()


__all__ = [
    "HelpArgumentSpec",
    "HelpCommandSpec",
    "HelpRegistry",
    "get_help_registry",
    "register_help",
    "render_help_categories",
    "render_help_category",
    "render_help_detail",
    "render_help_menu",
]
