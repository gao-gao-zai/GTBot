from __future__ import annotations

import asyncio
import secrets
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite
from nonebot import logger
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PLUGIN_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PLUGIN_ROOT / "group_gatekeeper.db"
CAPTCHA_IMAGE_DIR = PLUGIN_ROOT / "captcha_images"
DEFAULT_FONT_CANDIDATES = (
    Path(r"C:/Windows/Fonts/msyh.ttc"),
    Path(r"C:/Windows/Fonts/simhei.ttf"),
    Path(r"C:/Windows/Fonts/arial.ttf"),
)
LEGACY_WELCOME_TEMPLATE = (
    "欢迎入群，{user_at} 请在 {timeout_seconds} 秒内发送验证码：{code}。"
    "若超时未验证，将自动移出群聊。"
)
LEGACY_FAILURE_TEMPLATE = "验证码不正确，请重新输入。"

DEFAULT_SETTINGS: dict[str, str] = {
    "min_level": "16",
    "request_expire_seconds": "120",
    "verify_timeout_seconds": "300",
    "code_length": "4",
    "reject_on_low_level": "1",
    "kick_on_timeout": "1",
    "welcome_template": (
        "{user_at} 欢迎入群，请在 {timeout_seconds} 秒内直接发送图片中的验证码。"
        "不要发送@、空格或其他内容，超时未验证将自动移出群聊。"
    ),
    "success_template": "{user_at} 验证码正确，欢迎入群。",
    "failure_template": "{user_at} 请直接发送图片中的验证码，不要@或发送其他内容。剩余 {timeout_seconds} 秒。",
    "timeout_template": "{user_at} 验证超时，已移出群聊。",
    "reject_reason": "QQ 等级未达到入群要求",
}


def _to_bool(raw_value: str, default: bool) -> bool:
    """将 SQLite 中保存的字符串配置转换为布尔值。

    该函数只识别常见的真值与假值文本，未知内容会回落到调用方提供的默认值，
    以避免数据库被手动修改后让插件直接报错。

    Args:
        raw_value: 数据库中读取出的原始字符串值。
        default: 无法识别时使用的默认布尔值。

    Returns:
        转换后的布尔值。
    """

    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(slots=True)
class GatekeeperConfig:
    """描述入群守卫插件当前生效的配置快照。

    该对象由 SQLite 配置表和目标群表共同组装而成，供事件处理和命令层只读使用。
    由于运行时会频繁读取，本类保持为普通数据对象，避免在业务层反复拼接字典字段。
    """

    target_groups: set[int]
    min_level: int
    request_expire_seconds: int
    verify_timeout_seconds: int
    code_length: int
    reject_on_low_level: bool
    kick_on_timeout: bool
    welcome_template: str
    success_template: str
    failure_template: str
    timeout_template: str
    reject_reason: str


@dataclass(slots=True)
class PendingVerification:
    """保存单个待验证成员的临时状态。

    状态仅用于当前进程内的验证码校验生命周期，不写入 SQLite；
    这类数据天然具有短生命周期，放在内存中能减少数据库写入和清理复杂度。
    """

    group_id: int
    user_id: int
    code: str
    deadline_ts: float
    joined_at_ts: float
    image_path: Path | None = field(default=None)
    timeout_task: asyncio.Task[None] | None = field(default=None)


@dataclass(slots=True)
class ApprovedJoin:
    """记录一次由本插件自动批准的短期入群授权。

    该状态用于把“自动审批通过”与后续的 `group_increase` 事件关联起来，
    从而确保只有本插件亲自放行的成员才进入验证码流程。
    """

    group_id: int
    user_id: int
    approved_at_ts: float


class GatekeeperStore:
    """统一管理 SQLite 配置持久化与内存态验证码状态。

    该类将“长期配置”和“短期验证状态”集中在同一处维护，避免事件处理层同时关心数据库细节、
    并发锁和临时任务回收逻辑。配置数据库固定保存在插件根目录下，便于独立迁移和备份。
    """

    def __init__(self, database_path: Path | None = None) -> None:
        """初始化存储对象并准备惰性建表所需的状态。

        Args:
            database_path: 可选的 SQLite 文件路径；未提供时使用插件根目录下的默认路径。
        """

        self._database_path = Path(database_path) if database_path is not None else DATABASE_PATH
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        CAPTCHA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        self._init_lock = asyncio.Lock()
        self._pending_lock = asyncio.Lock()
        self._approved_lock = asyncio.Lock()
        self._initialized = False
        self._pending: dict[tuple[int, int], PendingVerification] = {}
        self._approved_joins: dict[tuple[int, int], ApprovedJoin] = {}

    @property
    def database_path(self) -> Path:
        """返回当前配置数据库的绝对路径。

        Returns:
            当前存储实例使用的 SQLite 文件路径。
        """

        return self._database_path

    async def ensure_ready(self) -> None:
        """确保 SQLite 表结构和默认配置已经初始化完成。

        该方法带有幂等和锁保护，适合在插件启动、命令执行或事件首次处理前重复调用，
        不会因并发访问而重复创建表或覆盖已有配置。
        """

        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self._database_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS target_groups (
                        group_id INTEGER PRIMARY KEY,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        created_at REAL NOT NULL DEFAULT 0
                    )
                    """
                )
                for key, value in DEFAULT_SETTINGS.items():
                    await db.execute(
                        "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                        (key, value),
                    )
                await db.execute(
                    """
                    UPDATE settings
                    SET value = ?
                    WHERE key = 'welcome_template' AND value = ?
                    """,
                    (DEFAULT_SETTINGS["welcome_template"], LEGACY_WELCOME_TEMPLATE),
                )
                await db.execute(
                    """
                    UPDATE settings
                    SET value = ?
                    WHERE key = 'failure_template' AND value = ?
                    """,
                    (DEFAULT_SETTINGS["failure_template"], LEGACY_FAILURE_TEMPLATE),
                )
                await db.commit()
            self._initialized = True

    async def get_config(self) -> GatekeeperConfig:
        """读取并组装当前完整配置快照。

        Returns:
            包含目标群和所有运行参数的配置快照对象。
        """

        await self.ensure_ready()
        async with aiosqlite.connect(self._database_path) as db:
            async with db.execute("SELECT key, value FROM settings") as cursor:
                rows = await cursor.fetchall()
            async with db.execute(
                "SELECT group_id FROM target_groups WHERE enabled = 1 ORDER BY group_id ASC"
            ) as cursor:
                target_group_rows = await cursor.fetchall()

        raw = {str(key): str(value) for key, value in rows}
        return GatekeeperConfig(
            target_groups={int(row[0]) for row in target_group_rows},
            min_level=max(1, int(raw.get("min_level", DEFAULT_SETTINGS["min_level"]))),
            request_expire_seconds=max(
                1,
                int(raw.get("request_expire_seconds", DEFAULT_SETTINGS["request_expire_seconds"])),
            ),
            verify_timeout_seconds=max(
                1,
                int(raw.get("verify_timeout_seconds", DEFAULT_SETTINGS["verify_timeout_seconds"])),
            ),
            code_length=max(1, int(raw.get("code_length", DEFAULT_SETTINGS["code_length"]))),
            reject_on_low_level=_to_bool(
                raw.get("reject_on_low_level", DEFAULT_SETTINGS["reject_on_low_level"]),
                True,
            ),
            kick_on_timeout=_to_bool(
                raw.get("kick_on_timeout", DEFAULT_SETTINGS["kick_on_timeout"]),
                True,
            ),
            welcome_template=raw.get("welcome_template", DEFAULT_SETTINGS["welcome_template"]),
            success_template=raw.get("success_template", DEFAULT_SETTINGS["success_template"]),
            failure_template=raw.get("failure_template", DEFAULT_SETTINGS["failure_template"]),
            timeout_template=raw.get("timeout_template", DEFAULT_SETTINGS["timeout_template"]),
            reject_reason=raw.get("reject_reason", DEFAULT_SETTINGS["reject_reason"]),
        )

    async def list_target_groups(self) -> list[int]:
        """返回当前已启用入群守卫的目标群列表。

        Returns:
            按群号升序排列的目标群号列表。
        """

        return sorted((await self.get_config()).target_groups)

    async def enable_group(self, group_id: int) -> None:
        """将指定群加入受保护的目标群集合。

        Args:
            group_id: 需要开启入群守卫的目标群号。
        """

        await self.ensure_ready()
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute(
                """
                INSERT INTO target_groups (group_id, enabled, created_at)
                VALUES (?, 1, ?)
                ON CONFLICT(group_id) DO UPDATE SET enabled = 1
                """,
                (int(group_id), time.time()),
            )
            await db.commit()

    async def disable_group(self, group_id: int) -> bool:
        """关闭指定群的入群守卫并清理运行中的待验证状态。

        Args:
            group_id: 需要关闭入群守卫的目标群号。

        Returns:
            当数据库中确实存在该群且本次成功关闭时返回 `True`，否则返回 `False`。
        """

        await self.ensure_ready()
        async with aiosqlite.connect(self._database_path) as db:
            cursor = await db.execute(
                "UPDATE target_groups SET enabled = 0 WHERE group_id = ? AND enabled = 1",
                (int(group_id),),
            )
            await db.commit()
            updated = cursor.rowcount > 0

        if updated:
            await self.clear_group_pending(int(group_id))
        return updated

    async def set_setting(self, key: str, value: str) -> None:
        """更新指定配置项的持久化值。

        Args:
            key: 配置键名，必须是本插件约定支持的字段。
            value: 将要写入数据库的新值，统一按字符串保存。

        Raises:
            ValueError: 当传入未知配置键时抛出。
        """

        if key not in DEFAULT_SETTINGS:
            raise ValueError(f"不支持的配置项: {key}")
        await self.ensure_ready()
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )
            await db.commit()

    async def reset_setting(self, key: str) -> None:
        """将指定配置项恢复为默认值。

        Args:
            key: 需要恢复默认值的配置键名。

        Raises:
            ValueError: 当传入未知配置键时抛出。
        """

        if key not in DEFAULT_SETTINGS:
            raise ValueError(f"不支持的配置项: {key}")
        await self.set_setting(key, DEFAULT_SETTINGS[key])

    async def upsert_pending(self, pending: PendingVerification) -> PendingVerification | None:
        """写入或替换某个成员的待验证状态。

        Args:
            pending: 新的待验证状态。

        Returns:
            如果存在被覆盖的旧状态，则返回旧状态；否则返回 `None`。
        """

        key = (pending.group_id, pending.user_id)
        async with self._pending_lock:
            previous = self._pending.get(key)
            self._pending[key] = pending
            return previous

    async def get_pending(self, group_id: int, user_id: int) -> PendingVerification | None:
        """获取指定成员当前的待验证状态。

        Args:
            group_id: 群号。
            user_id: 成员 QQ 号。

        Returns:
            待验证状态；若不存在则返回 `None`。
        """

        async with self._pending_lock:
            return self._pending.get((group_id, user_id))

    async def pop_pending(self, group_id: int, user_id: int) -> PendingVerification | None:
        """移除并返回指定成员的待验证状态。

        Args:
            group_id: 群号。
            user_id: 成员 QQ 号。

        Returns:
            被移除的待验证状态；若不存在则返回 `None`。
        """

        async with self._pending_lock:
            return self._pending.pop((group_id, user_id), None)

    async def clear_group_pending(self, group_id: int) -> list[PendingVerification]:
        """清空指定群的全部待验证状态。

        关闭某个群的守卫功能时，需要把该群中尚未完成的验证码流程一起撤销，
        避免配置关闭后旧任务仍继续踢人。

        Args:
            group_id: 需要清理的群号。

        Returns:
            被移除的待验证状态列表。
        """

        async with self._pending_lock:
            removed_keys = [key for key in self._pending if key[0] == int(group_id)]
            removed_values = [self._pending.pop(key) for key in removed_keys]
            return removed_values

    async def clear_pending(self) -> list[PendingVerification]:
        """清空全部待验证状态。

        Returns:
            清空前保存的全部待验证状态列表。
        """

        async with self._pending_lock:
            values = list(self._pending.values())
            self._pending.clear()
            return values

    async def cancel_pending_tasks(self, pendings: list[PendingVerification]) -> None:
        """取消给定待验证状态关联的超时任务。

        Args:
            pendings: 需要批量取消任务的待验证状态列表。
        """

        for pending in pendings:
            await cancel_task(pending.timeout_task)
            delete_file(pending.image_path)

    async def mark_approved_join(self, group_id: int, user_id: int) -> None:
        """记录一条由本插件自动批准的短期入群授权。

        Args:
            group_id: 目标群号。
            user_id: 被自动批准的成员 QQ 号。
        """

        async with self._approved_lock:
            self._approved_joins[(int(group_id), int(user_id))] = ApprovedJoin(
                group_id=int(group_id),
                user_id=int(user_id),
                approved_at_ts=time.time(),
            )

    async def consume_approved_join(
        self,
        group_id: int,
        user_id: int,
        *,
        ttl_seconds: int,
    ) -> bool:
        """消费一条自动批准授权，并校验其是否仍在有效期内。

        有效期外的旧授权会被视为失效并自动移除，避免 Bot 延迟恢复后错误触发验证码流程。

        Args:
            group_id: 群号。
            user_id: 成员 QQ 号。
            ttl_seconds: 授权允许保留的最大秒数。

        Returns:
            当存在且仍有效的授权记录时返回 `True`，否则返回 `False`。
        """

        key = (int(group_id), int(user_id))
        async with self._approved_lock:
            approved = self._approved_joins.pop(key, None)
        if approved is None:
            return False
        return time.time() - approved.approved_at_ts <= max(1, int(ttl_seconds))

    async def clear_approved_joins(self) -> None:
        """清空全部自动批准授权记录。"""

        async with self._approved_lock:
            self._approved_joins.clear()


def build_code(length: int) -> str:
    """生成指定长度的数字验证码。

    Args:
        length: 验证码长度，调用方应保证其为正整数。

    Returns:
        随机数字验证码字符串。
    """

    digits = string.digits
    return "".join(secrets.choice(digits) for _ in range(length))


def _load_captcha_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载验证码渲染所需的字体对象。

    优先使用系统中常见的中文字体，确保数字在中文 Windows 环境中显示稳定；
    若全部失败，则退回 Pillow 默认字体，保证功能可用。

    Args:
        font_size: 目标字号。

    Returns:
        可用于 Pillow 绘制文本的字体对象。
    """

    for font_path in DEFAULT_FONT_CANDIDATES:
        if not font_path.exists():
            continue
        try:
            return ImageFont.truetype(str(font_path), font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_captcha_image(code: str, group_id: int, user_id: int) -> Path:
    """在插件目录下生成一张数字验证码图片。

    图片文件会写入插件根目录下的 `captcha_images` 子目录，便于 OneBot 直接读取本地文件。
    当前实现有轻量噪点、干扰线和轻微模糊，足以覆盖普通数字验证码场景。

    Args:
        code: 需要展示在图片中的数字验证码。
        group_id: 群号，用于区分不同群的验证码文件名。
        user_id: 成员 QQ 号，用于区分不同成员的验证码文件名。

    Returns:
        已生成完成的验证码 PNG 文件路径。
    """

    code = str(code).strip()
    width = max(160, 42 * len(code) + 36)
    height = 78
    background_color = (248, 250, 252)
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    for _ in range(6):
        x1 = secrets.randbelow(width)
        y1 = secrets.randbelow(height)
        x2 = secrets.randbelow(width)
        y2 = secrets.randbelow(height)
        color = (
            140 + secrets.randbelow(70),
            140 + secrets.randbelow(70),
            140 + secrets.randbelow(70),
        )
        draw.line((x1, y1, x2, y2), fill=color, width=1)

    for _ in range(90):
        x = secrets.randbelow(width)
        y = secrets.randbelow(height)
        color = (
            120 + secrets.randbelow(100),
            120 + secrets.randbelow(100),
            120 + secrets.randbelow(100),
        )
        draw.point((x, y), fill=color)

    font = _load_captcha_font(40)
    for index, char in enumerate(code):
        x = 18 + index * 38 + secrets.randbelow(6)
        y = 14 + secrets.randbelow(10)
        color = (
            20 + secrets.randbelow(70),
            70 + secrets.randbelow(70),
            110 + secrets.randbelow(80),
        )
        draw.text((x, y), char, font=font, fill=color)

    image = image.filter(ImageFilter.SMOOTH)
    file_name = f"captcha_{group_id}_{user_id}_{int(time.time() * 1000)}.png"
    image_path = CAPTCHA_IMAGE_DIR / file_name
    image.save(image_path, format="PNG")
    image.close()
    return image_path


def extract_level(raw_info: dict[str, object]) -> int | None:
    """从 LLOneBot 的陌生人信息响应中提取 QQ 等级。

    文档说明等级位于 `data.level`，但兼容考虑下也允许直接从顶层 `level` 读取。

    Args:
        raw_info: `get_stranger_info` 返回的原始对象。

    Returns:
        解析成功时返回等级整数，否则返回 `None`。
    """

    data = raw_info.get("data", raw_info)
    if not isinstance(data, dict):
        return None
    level = data.get("level")
    if level is None:
        return None
    try:
        return int(level)
    except (TypeError, ValueError):
        return None


async def cancel_task(task: asyncio.Task[None] | None) -> None:
    """取消超时任务并安全吞掉正常的取消异常。

    Args:
        task: 待取消的异步任务；为空或已结束时直接返回。
    """

    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return


def delete_file(path: Path | None) -> None:
    """删除验证码缓存文件并忽略缺失或权限边角异常。

    Args:
        path: 需要删除的本地文件路径；为空时直接返回。
    """

    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        return


store = GatekeeperStore()


async def log_database_path() -> None:
    """在启动时输出数据库文件位置，便于运维确认配置落点。"""

    await store.ensure_ready()
    logger.info("群入群验证码守卫数据库已就绪: %s", store.database_path)
