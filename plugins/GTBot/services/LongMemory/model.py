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


@dataclass(frozen=True)
class GroupProfileDescriptionWithId:
    """带存储 ID 的群描述条目。

    Attributes:
        doc_id: 该条描述在向量数据库（Qdrant/Chroma）中的记录 ID。
        text: 描述文本内容。
    """

    doc_id: str
    text: str


@dataclass
class GroupProfileWithDescriptionIds:
    """群画像（包含每条描述在数据库中的记录 ID）。

    Attributes:
        id: 群 ID (QQ群号)。
        description: 带存储 ID 的描述条目列表。
    """

    id: int
    description: list[GroupProfileDescriptionWithId]


@dataclass
class PublicKnowledge:
    """公共知识。

    Attributes:
        title: 公共知识标题。
        content: 公共知识正文内容。
    """

    title: str
    """公共知识标题"""
    content: str
    """公共知识内容"""


@dataclass(frozen=True)
class PublicKnowledgeWithId:
    """带存储 ID 的公共知识条目。

    Attributes:
        doc_id: 该条记录在向量数据库（Qdrant）中的 point id。
        title: 公共知识标题。
        content: 公共知识正文内容。
    """

    doc_id: str
    title: str
    content: str


@dataclass(frozen=True)
class PublicKnowledgeSearchHit:
    """公共知识检索命中项。

    Attributes:
        doc_id: 该条记录在向量数据库（Qdrant）中的 point id。
        title: 公共知识标题。
        content: 公共知识正文内容。
        distance: 距离分数（本项目中通常以 `1 - similarity` 作为便捷展示）。
        similarity: 相似度分数（Qdrant `COSINE` 下通常为 score，越大越相似）。
    """

    doc_id: str
    title: str
    content: str
    distance: float
    similarity: float

@dataclass
class TimeSlot:
    """时间段。

    Attributes:
        start_time: 时间段开始时间（Unix 时间戳，秒）。
        end_time: 时间段结束时间（Unix 时间戳，秒）。
    """

    start_time: float
    end_time: float

@dataclass
class EventLog:
    """事件日志（不含存储 ID）。

    Attributes:
        time_slots: 事件相关的时间段列表。
        details: 事件详情（自然语言）。
        relevant_members: 相关成员 ID 列表（QQ号）。
        session_id: 所属会话 ID（例如群号或其它会话标识）。
    """

    time_slots: list[TimeSlot]
    details: str
    relevant_members: list[int]
    session_id: int


@dataclass(frozen=True)
class EventLogWithId:
    """带存储 ID 的事件日志条目。

    Attributes:
        doc_id: 事件在向量数据库（Qdrant）中的 point id。
        event_name: 事件名（允许重复，仅作标识/展示）。
        session_id: 会话 ID。
        relevant_members: 相关成员 ID 列表。
        time_slots: 时间段列表。
        details: 事件详情。
        status: 事件状态（open/closed）。
    """

    doc_id: str
    event_name: str
    session_id: int
    relevant_members: list[int]
    time_slots: list[TimeSlot]
    details: str
    status: str


@dataclass(frozen=True)
class EventLogSearchHit:
    """事件日志检索命中项。

    Attributes:
        doc_id: point id。
        event_name: 事件名。
        session_id: 会话 ID。
        relevant_members: 相关成员 ID 列表。
        details: 事件详情。
        status: 事件状态。
        distance: 便捷展示的距离（通常为 `1 - similarity`）。
        similarity: 相似度（Qdrant `score`，越大越相似）。
    """

    doc_id: str
    event_name: str
    session_id: int
    relevant_members: list[int]
    details: str
    status: str
    distance: float
    similarity: float

@dataclass
class Relationships:
    """关系画像"""
    member_a_id: int
    """成员 A ID (QQ号)"""
    member_b_id: int
    """成员 B ID (QQ号)"""
    relationship: str
    """关系描述"""



