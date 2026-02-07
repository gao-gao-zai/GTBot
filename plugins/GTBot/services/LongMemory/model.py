from dataclasses import dataclass





@dataclass
class UserProfile:
    """用户画像"""
    id: int
    """用户 ID (QQ号)"""
    description: list[str]
    """用户描述列表"""


@dataclass(frozen=True)
class UserProfileDescriptionWithId:
    """带存储 ID 的用户描述条目。

    Attributes:
        doc_id: 该条描述在向量数据库（Chroma）中的记录 ID。
        text: 描述文本内容。
    """

    doc_id: str
    text: str


@dataclass
class UserProfileWithDescriptionIds:
    """用户画像（包含每条描述在数据库中的记录 ID）。

    与 `UserProfile` 的区别：
        - `description` 不再是纯文本列表，而是包含 `doc_id + text` 的条目列表。

    Attributes:
        id: 用户 ID (QQ号)。
        description: 带存储 ID 的描述条目列表。
    """

    id: int
    description: list[UserProfileDescriptionWithId]


@dataclass(frozen=True)
class UserProfileSearchHit:
    """用户画像检索命中项。

    该结构用于“跨用户”的向量检索结果返回。

    Attributes:
        doc_id: 该条描述在向量数据库（Chroma）中的记录 ID。
        user_id: 该条描述所属的用户 ID (QQ号)。
        text: 描述文本内容。
        distance: 距离分数（通常越小越相似，具体含义取决于 Chroma 的度量方式）。
        similarity: 相似度分数（默认使用 `1 - distance` 计算，仅作便捷展示）。
    """

    doc_id: str
    user_id: int
    text: str
    distance: float
    similarity: float

@dataclass
class GroupProfile:
    """群画像"""
    id: int
    """群 ID (QQ群号)"""
    description: list[str]
    """群描述列表"""


@dataclass
class PublicKnowledge:
    """公共知识"""
    content: str
    """公共知识内容"""
    tags: list[str]
    """公共知识标签列表"""

@dataclass
class TimeSlot:
    """时间段"""
    start_time: float
    """时间段开始时间"""
    end_time: float
    """时间段结束时间"""

@dataclass
class EventLog:
    time_slots: list[TimeSlot]
    """事件相关的时间段列表"""
    details: str
    """事件详情"""
    relevant_members: list[int]
    """相关成员 ID 列表 (QQ号)"""
    session_id: int
    """所属会话 ID (group_ + QQ群号 或 private_ + QQ号)"""

@dataclass
class Relationships:
    """关系画像"""
    member_a_id: int
    """成员 A ID (QQ号)"""
    member_b_id: int
    """成员 B ID (QQ号)"""
    relationship: str
    """关系描述"""



