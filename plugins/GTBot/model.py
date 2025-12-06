import time
import asyncio
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from sqlalchemy import (
    select,
    String,
    INTEGER,
    BOOLEAN,
    FLOAT,
    Text,
    UniqueConstraint,
    update,
    delete,
    desc,
    asc,
    and_,
    or_,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from pydantic import BaseModel, Field
from sympy import im

# --- 配置初始化和数据库路径配置 ---
try:
    from .ConfigManager import total_config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from ConfigManager import total_config

# 初始化配置
try:
    DATA_DIR = total_config.processed_configuration.config.data_dir_path
    DB_PATH = DATA_DIR / "data.db"
    ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"
except Exception as e:
    # 如果配置加载失败，使用默认路径
    print(f"警告：配置加载失败，使用默认数据库路径: {e}")
    DB_PATH = Path("./data.db")
    ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# --- 基础模型 ---
class Base(DeclarativeBase):
    pass

class CacheBaseMixin:
    """缓存模型通用字段"""

    last_update_time: Mapped[float] = mapped_column(
        FLOAT, default=0.0, index=True, nullable=False
    )
    last_access_time: Mapped[float] = mapped_column(
        FLOAT, default=0.0, index=True, nullable=False
    )


class GroupMessages(Base):
    """群聊消息模型 (ORM)"""
    __tablename__ = "group_messages"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    """数据库自增ID"""
    
    message_id: Mapped[int] = mapped_column(INTEGER, index=True, unique=True) 
    """消息ID (平台侧ID)"""
    
    group_id: Mapped[int] = mapped_column(INTEGER, index=True, default=0)
    """群组ID"""
    
    user_id: Mapped[int] = mapped_column(INTEGER, index=True)
    """用户ID"""
    
    user_name: Mapped[str] = mapped_column(String, default="")
    """用户昵称"""
    
    content: Mapped[str] = mapped_column(String, default="")
    """消息内容"""
    
    send_time: Mapped[float] = mapped_column(FLOAT, index=True, default=0.0)
    """发送时间戳"""
    
    is_withdrawn: Mapped[bool] = mapped_column(BOOLEAN, default=False)
    """是否已被撤回"""

    def to_pydantic(self) -> "GroupMessage":
        """转换为Pydantic模型"""
        return GroupMessage(
            db_id=self.id,
            message_id=self.message_id,
            group_id=self.group_id,
            user_id=self.user_id,
            user_name=self.user_name,
            content=self.content,
            send_time=self.send_time,
            is_withdrawn=self.is_withdrawn
        )

class GroupMessage(BaseModel):
    """群聊消息 (Pydantic)"""
    db_id: Optional[int] = None
    """数据库内部ID"""
    message_id: int
    """消息ID"""
    group_id: int = 0
    """群号"""
    user_id: int
    """发送者QQ号"""
    user_name: str = ""
    """发送者昵称"""
    content: str = ""
    """消息内容"""
    send_time: float = Field(default_factory=lambda: time.time())
    """发送时间"""
    is_withdrawn: bool = False
    """是否撤回"""



# ============================================================================
# 缓存数据 Pydantic 模型
# ============================================================================

class GroupAllInfo(BaseModel):
    """群详细信息 (groupAll 字段)"""
    group_code: str = ""
    """群号字符串"""
    owner_uid: str = ""
    """群主UID"""
    group_flag: int = 0
    """群标志"""
    group_flag_ext: int = 0
    """群标志扩展"""
    max_member_num: int = 0
    """最大成员数"""
    member_num: int = 0
    """成员数"""
    group_option: int = 0
    """群选项"""
    class_ext: int = 0
    """分类扩展"""
    group_name: str = ""
    """群名称"""
    finger_memo: str = ""
    """群介绍"""
    group_question: str = ""
    """入群问题"""
    cert_type: int = 0
    """认证类型"""
    shut_up_all_timestamp: int = 0
    """全体禁言时间戳"""
    shut_up_me_timestamp: int = 0
    """自己被禁言时间戳"""
    group_type_flag: int = 0
    """群类型标志"""
    privilege_flag: int = 0
    """权限标志"""
    group_sec_level: int = 0
    """群安全等级"""
    group_flag_ext3: int = 0
    """群标志扩展3"""
    is_conf_group: int = 0
    """是否讨论组"""
    is_modify_conf_group_face: int = 0
    """是否可修改讨论组头像"""
    is_modify_conf_group_name: int = 0
    """是否可修改讨论组名称"""
    no_figer_open_flag: int = 0
    no_code_finger_open_flag: int = 0
    group_flag_ext4: int = 0
    """群标志扩展4"""
    group_memo: str = ""
    """群公告"""
    cmd_uin_msg_seq: int = 0
    cmd_uin_join_time: int = 0
    """Bot加入时间"""
    cmd_uin_uin_flag: int = 0
    cmd_uin_msg_mask: int = 0
    group_sec_level_info: int = 0
    cmd_uin_privilege: int = 0
    """Bot权限"""
    cmd_uin_flag_ex2: int = 0
    appeal_deadline: int = 0
    remark_name: str = ""
    """备注名"""
    is_top: bool = False
    """是否置顶"""
    rich_finger_memo: str = ""
    """富文本群介绍"""
    group_answer: str = ""
    """入群答案"""
    join_group_auth: str = ""
    """入群验证"""
    is_allow_modify_conf_group_name: int = 0

    class Config:
        extra = "allow"
        populate_by_name = True

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "GroupAllInfo":
        """从原始数据构建对象 (处理驼峰命名)"""
        if not data:
            return cls()
        return cls(
            group_code=data.get("groupCode", ""),
            owner_uid=data.get("ownerUid", ""),
            group_flag=data.get("groupFlag", 0),
            group_flag_ext=data.get("groupFlagExt", 0),
            max_member_num=data.get("maxMemberNum", 0),
            member_num=data.get("memberNum", 0),
            group_option=data.get("groupOption", 0),
            class_ext=data.get("classExt", 0),
            group_name=data.get("groupName", ""),
            finger_memo=data.get("fingerMemo", ""),
            group_question=data.get("groupQuestion", ""),
            cert_type=data.get("certType", 0),
            shut_up_all_timestamp=data.get("shutUpAllTimestamp", 0),
            shut_up_me_timestamp=data.get("shutUpMeTimestamp", 0),
            group_type_flag=data.get("groupTypeFlag", 0),
            privilege_flag=data.get("privilegeFlag", 0),
            group_sec_level=data.get("groupSecLevel", 0),
            group_flag_ext3=data.get("groupFlagExt3", 0),
            is_conf_group=data.get("isConfGroup", 0),
            is_modify_conf_group_face=data.get("isModifyConfGroupFace", 0),
            is_modify_conf_group_name=data.get("isModifyConfGroupName", 0),
            no_figer_open_flag=data.get("noFigerOpenFlag", 0),
            no_code_finger_open_flag=data.get("noCodeFingerOpenFlag", 0),
            group_flag_ext4=data.get("groupFlagExt4", 0),
            group_memo=data.get("groupMemo", ""),
            cmd_uin_msg_seq=data.get("cmdUinMsgSeq", 0),
            cmd_uin_join_time=data.get("cmdUinJoinTime", 0),
            cmd_uin_uin_flag=data.get("cmdUinUinFlag", 0),
            cmd_uin_msg_mask=data.get("cmdUinMsgMask", 0),
            group_sec_level_info=data.get("groupSecLevelInfo", 0),
            cmd_uin_privilege=data.get("cmdUinPrivilege", 0),
            cmd_uin_flag_ex2=data.get("cmdUinFlagEx2", 0),
            appeal_deadline=data.get("appealDeadline", 0),
            remark_name=data.get("remarkName", ""),
            is_top=data.get("isTop", False),
            rich_finger_memo=data.get("richFingerMemo", ""),
            group_answer=data.get("groupAnswer", ""),
            join_group_auth=data.get("joinGroupAuth", ""),
            is_allow_modify_conf_group_name=data.get("isAllowModifyConfGroupName", 0),
        )


class GroupInfo(BaseModel):
    """群信息 (Pydantic)"""
    group_id: int
    """群号"""
    group_name: str = ""
    """群名称"""
    group_memo: str = ""
    """群备注"""
    group_create_time: int = 0
    """群创建时间"""
    member_count: int = 0
    """成员数量"""
    max_member_count: int = 0
    """最大成员数量"""
    remark_name: str = ""
    """备注名"""
    # groupAll 详细信息
    group_all: Optional[GroupAllInfo] = None
    """群详细信息"""
    # 缓存元数据
    last_update_time: float = 0.0
    """最后更新时间"""
    last_access_time: float = 0.0
    """最后访问时间"""

    class Config:
        extra = "allow"

    @property
    def is_shutup_all(self) -> bool:
        """是否全体禁言"""
        if self.group_all:
            return self.group_all.shut_up_all_timestamp > 0
        return False

    @property
    def bot_join_time(self) -> int:
        """Bot加入群时间"""
        if self.group_all:
            return self.group_all.cmd_uin_join_time
        return 0


class GroupMemberInfo(BaseModel):
    """群成员信息 (Pydantic)"""
    group_id: int
    """群号"""
    user_id: int
    """用户QQ号"""
    nickname: str = ""
    """用户昵称"""
    card: str = ""
    """群名片"""
    sex: str = "unknown"
    """性别"""
    age: int = 0
    """年龄"""
    area: str = ""
    """地区"""
    level: str = ""
    """群等级"""
    qq_level: int = 0
    """QQ等级"""
    join_time: int = 0
    """加入时间"""
    last_sent_time: int = 0
    """最后发言时间"""
    title_expire_time: int = 0
    """头衔过期时间"""
    unfriendly: bool = False
    """是否不友好"""
    card_changeable: bool = False
    """是否可改群名片"""
    is_robot: bool = False
    """是否机器人"""
    shut_up_timestamp: int = 0
    """禁言时间戳"""
    role: str = "member"
    """角色 (owner/admin/member)"""
    title: str = ""
    """头衔"""
    # 额外数据
    extra: Dict[str, Any] = Field(default_factory=dict)
    """额外数据"""
    # 缓存元数据
    last_update_time: float = 0.0
    """最后更新时间"""
    last_access_time: float = 0.0
    """最后访问时间"""

    class Config:
        extra = "allow"

    @property
    def display_name(self) -> str:
        """显示名称 (优先群名片，其次昵称)"""
        return self.card if self.card else self.nickname


class StrangerInfo(BaseModel):
    """陌生人信息 (Pydantic)"""
    user_id: int
    """用户QQ号"""
    nickname: str = ""
    """昵称"""
    sex: str = "unknown"
    """性别"""
    age: int = 0
    """年龄"""
    qid: str = ""
    """QID"""
    level: int = 0
    """等级"""
    login_days: int = 0
    """登录天数"""
    reg_time: int = 0
    """注册时间"""
    long_nick: str = ""
    """个性签名"""
    city: str = ""
    """城市"""
    country: str = ""
    """国家"""
    birthday_year: int = 0
    """生日年"""
    birthday_month: int = 0
    """生日月"""
    birthday_day: int = 0
    """生日日"""
    # 额外数据
    extra: Dict[str, Any] = Field(default_factory=dict)
    """额外数据"""
    # 缓存元数据
    last_update_time: float = 0.0
    """最后更新时间"""
    last_access_time: float = 0.0
    """最后访问时间"""

    class Config:
        extra = "allow"


class CachedGroupInfo(CacheBaseMixin, Base):
    """群信息缓存"""

    __tablename__ = "cached_group_info"

    group_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    data: Mapped[str] = mapped_column(Text, default="{}")

    def to_pydantic(self) -> "GroupInfo":
        """转换为Pydantic模型"""
        import json
        raw = json.loads(self.data)
        
        # 解析 groupAll 字段
        group_all_raw = raw.get("groupAll")
        group_all = GroupAllInfo.from_raw(group_all_raw) if group_all_raw else None
        
        return GroupInfo(
            group_id=raw.get("group_id", self.group_id),
            group_name=raw.get("group_name", ""),
            group_memo=raw.get("group_memo", ""),
            group_create_time=raw.get("group_create_time", 0),
            member_count=raw.get("member_count", 0),
            max_member_count=raw.get("max_member_count", 0),
            remark_name=raw.get("remark_name", ""),
            group_all=group_all,
            last_update_time=self.last_update_time,
            last_access_time=self.last_access_time
        )


class CachedGroupMemberInfo(CacheBaseMixin, Base):
    """群成员信息缓存"""

    __tablename__ = "cached_group_member_info"
    __table_args__ = (
        UniqueConstraint("group_id", "user_id", name="uq_group_member"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(INTEGER, index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(INTEGER, index=True, nullable=False)
    data: Mapped[str] = mapped_column(Text, default="{}")

    def to_pydantic(self) -> "GroupMemberInfo":
        """转换为Pydantic模型"""
        import json
        raw = json.loads(self.data)
        known_fields = {
            "group_id", "user_id", "nickname", "card", "sex", "age", "area",
            "level", "qq_level", "join_time", "last_sent_time", "title_expire_time",
            "unfriendly", "card_changeable", "is_robot", "shut_up_timestamp",
            "role", "title"
        }
        extra = {k: v for k, v in raw.items() if k not in known_fields}
        return GroupMemberInfo(
            group_id=raw.get("group_id", self.group_id),
            user_id=raw.get("user_id", self.user_id),
            nickname=raw.get("nickname", ""),
            card=raw.get("card", ""),
            sex=raw.get("sex", "unknown"),
            age=raw.get("age", 0),
            area=raw.get("area", ""),
            level=str(raw.get("level", "")),
            qq_level=raw.get("qq_level", 0),
            join_time=raw.get("join_time", 0),
            last_sent_time=raw.get("last_sent_time", 0),
            title_expire_time=raw.get("title_expire_time", 0),
            unfriendly=raw.get("unfriendly", False),
            card_changeable=raw.get("card_changeable", False),
            is_robot=raw.get("is_robot", False),
            shut_up_timestamp=raw.get("shut_up_timestamp", 0),
            role=raw.get("role", "member"),
            title=raw.get("title", ""),
            extra=extra,
            last_update_time=self.last_update_time,
            last_access_time=self.last_access_time
        )


class CachedStrangerInfo(CacheBaseMixin, Base):
    """陌生人信息缓存"""

    __tablename__ = "cached_stranger_info"

    user_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    data: Mapped[str] = mapped_column(Text, default="{}")

    def to_pydantic(self) -> "StrangerInfo":
        """转换为Pydantic模型"""
        import json
        raw = json.loads(self.data)
        known_fields = {
            "user_id", "nickname", "sex", "age", "qid", "level", "login_days",
            "reg_time", "long_nick", "city", "country", "birthday_year",
            "birthday_month", "birthday_day"
        }
        extra = {k: v for k, v in raw.items() if k not in known_fields}
        return StrangerInfo(
            user_id=raw.get("user_id", self.user_id),
            nickname=raw.get("nickname", ""),
            sex=raw.get("sex", "unknown"),
            age=raw.get("age", 0),
            qid=raw.get("qid", ""),
            level=raw.get("level", 0),
            login_days=raw.get("login_days", 0),
            reg_time=raw.get("reg_time", 0),
            long_nick=raw.get("long_nick", ""),
            city=raw.get("city", ""),
            country=raw.get("country", ""),
            birthday_year=raw.get("birthday_year", 0),
            birthday_month=raw.get("birthday_month", 0),
            birthday_day=raw.get("birthday_day", 0),
            extra=extra,
            last_update_time=self.last_update_time,
            last_access_time=self.last_access_time
        )


engine = create_async_engine(ASYNC_DB_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def init_all_tables() -> None:
    """初始化 ORM 表结构。

    在 NoneBot 启动或首次使用缓存前调用，以确保 `data.db` 中存在
    `group_messages` 及各类缓存表。

    Raises:
        SQLAlchemyError: 当底层数据库连接或建表失败时抛出。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
