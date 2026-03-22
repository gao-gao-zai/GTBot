from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import time



class Message(BaseModel):
    """消息数据类 (Pydantic)"""
    message_id: int
    """消息ID"""
    user_id: int
    """发送者QQ号"""
    content: str = ""
    """消息内容"""
    send_time: float = Field(default_factory=lambda: time.time())
    """发送时间"""
    is_withdrawn: bool = False
    """是否撤回"""

class GroupMessage(Message):
    """群聊消息（领域模型）。

    该模型用于业务逻辑与对话上下文构建，不应包含任何数据库实现细节。

    Attributes:
        group_id: 群号。
    """
    group_id: int = 0
    """群号"""
    user_name: str = ""
    """发送者昵称"""

class PrivateMessage(Message):
    """私聊消息（领域模型）。

    该模型用于业务逻辑与对话上下文构建，不应包含任何数据库实现细节。

    Attributes:
        pass  # 私聊消息目前没有额外字段
    """
    pass

class GroupMessageRecord(GroupMessage):
    """群聊消息（持久化记录模型）。

    该模型用于数据库/DAO 层读写，携带数据库内部自增主键等存储细节。
    业务层如不需要定位具体行记录，应优先使用 `GroupMessage`。

    Attributes:
        db_id: 数据库内部自增 ID。
    """

    db_id: int
    """数据库内部自增ID"""

    def to_domain(self) -> GroupMessage:
        """转换为领域模型。

        Returns:
            不包含数据库字段的领域消息对象。
        """

        data: Dict[str, Any] = self.model_dump(exclude={"db_id"})
        return GroupMessage(**data)

    @classmethod
    def from_domain(cls, msg: GroupMessage, db_id: int) -> "GroupMessageRecord":
        """由领域模型与数据库主键构建持久化记录。

        Args:
            msg: 领域消息对象。
            db_id: 数据库内部自增 ID。

        Returns:
            持久化记录对象。
        """

        return cls(db_id=db_id, **msg.model_dump())

# ============================================================================
# 缓存数据模型
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


# ============================================================================
# 用户画像模型
# ============================================================================


class UserProfile(BaseModel):
    """用户画像模型 (Pydantic)"""
    user_id: int
    """用户QQ号"""
    description: list[str] = []
    """用户描述列表"""


class GroupProfile(BaseModel):
    """群画像模型 (Pydantic)"""
    group_id: int
    """群号"""
    description: list[str] = []
    """群描述列表"""








class MessageTask(BaseModel):
    """消息发送任务数据类（仅携带可序列化数据）。

    Attributes:
        messages: 要发送的消息列表。
        group_id: 目标群组 ID。
        interval: 发送多条消息时的间隔时间（秒）。
    """
    messages: List[Any]
    group_id: int
    interval: float
