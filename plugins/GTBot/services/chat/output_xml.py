"""GTBot 结构化输出 XML 解析工具。

本模块负责把模型输出的 XML 片段解析为可消费的顶层节点序列，
用于替代分散在运行时中的正则抽取逻辑。当前同时覆盖“完整响应”
和“流式增量响应”两类场景，为运行时提供统一、可回退的 XML 解析能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, cast
from xml.etree import ElementTree as ET

ALLOWED_OUTPUT_TAGS: Final[frozenset[str]] = frozenset(
    {"msg", "meme", "note", "thinking", "silent"}
)
"""当前协议约定的结构化输出标签集合。"""

OutputXMLParseError = ET.ParseError
"""结构化输出 XML 解析异常别名。"""


@dataclass(slots=True)
class OutputXMLNode:
    """表示模型结构化输出中的一个顶层 XML 节点。

    该对象只描述最外层兄弟节点，不尝试把业务语义进一步解释为“要发送的消息”
    或“要写入的记事本”，以便上层运行时根据上下文决定消费方式。节点会同时
    保留归一化后的标签名、纯文本内容、属性字典以及可重新拼接的 XML 文本。

    Attributes:
        tag: 归一化为小写后的标签名。
        text: 当前节点及其所有子节点拼接后的纯文本内容，已去除首尾空白。
        attrs: 开始标签上的属性字典；当前协议暂未大量使用，但保留以便扩展。
        xml: 当前节点重新序列化后的 XML 文本，用于按需重建剩余片段。
    """

    tag: str
    text: str
    attrs: dict[str, str]
    xml: str

    @classmethod
    def from_element(cls, element: ET.Element) -> "OutputXMLNode":
        """从 `ElementTree` 节点构造业务侧可消费的顶层节点对象。

        这里会把标签名统一转成小写，并使用 `itertext()` 聚合文本内容，
        以兼容模型偶发产生的轻微嵌套结构。对于不需要属性的场景，上层可以
        直接忽略 `attrs` 字段，而无需重新触碰 `ElementTree` 对象。

        Args:
            element: 已完成 XML 解析的顶层子节点。

        Returns:
            适合上层业务直接消费的 `OutputXMLNode` 实例。
        """

        return cls(
            tag=str(element.tag).strip().lower(),
            text="".join(element.itertext()).strip(),
            attrs={str(key): str(value) for key, value in element.attrib.items()},
            xml=ET.tostring(element, encoding="unicode"),
        )


@dataclass(slots=True)
class ParsedOutputXMLDocument:
    """表示一次模型 XML 片段解析后的顶层节点序列。

    当前协议约定输出是若干个并列的顶层标签片段，因此解析时会额外包一层
    虚拟根节点，再把其直接子节点投影到本对象中。该对象不保存 `ElementTree`
    原始节点，避免运行时到处传递可变的树结构；后续若要支持更多标签或属性，
    只需扩展 `OutputXMLNode` 的消费逻辑即可。

    Attributes:
        nodes: 按原始出现顺序排列的顶层节点列表。
    """

    nodes: list[OutputXMLNode]

    def select(self, tags: set[str]) -> list[OutputXMLNode]:
        """按标签名筛选顶层节点。

        Args:
            tags: 需要保留的标签名集合；比较时会按小写标签名匹配。

        Returns:
            与输入顺序一致的节点列表。若没有匹配节点，则返回空列表。
        """

        normalized = {str(tag).strip().lower() for tag in tags}
        return [node for node in self.nodes if node.tag in normalized]

    def contains(self, tag: str) -> bool:
        """判断当前片段中是否存在指定标签。

        该方法主要用于像 `<silent />` 这类“控制语义大于文本语义”的标签，
        让上层可以在不重新遍历节点列表的情况下快速决定是否跳过发送。

        Args:
            tag: 需要判断的标签名。

        Returns:
            当任一顶层节点匹配该标签时返回 `True`，否则返回 `False`。
        """

        normalized = str(tag).strip().lower()
        return any(node.tag == normalized for node in self.nodes)

    def notes(self) -> list[str]:
        """提取所有 `<note>` 节点的纯文本内容。

        Returns:
            按出现顺序提取并去除首尾空白后的 note 文本列表。空 note 会被忽略，
            以避免把仅包含空白的占位标签误写入记事本。
        """

        return [node.text for node in self.nodes if node.tag == "note" and node.text]

    def render_without(self, excluded_tags: set[str]) -> str:
        """在保留原始顺序的前提下重建排除部分标签后的 XML 片段。

        该方法用于“先解析所有节点，再移除某类控制标签”的场景，例如先剥离
        `<note>` 再把剩余 `<msg>` / `<meme>` / `<silent>` 交给发送链路处理。
        由于当前协议不允许依赖裸文本，本方法只重建顶层节点本身，不保留节点
        间散落的自由文本。

        Args:
            excluded_tags: 需要从结果中排除的标签名集合。

        Returns:
            重新拼接后的 XML 片段字符串；若排除后没有任何节点，则返回空串。
        """

        normalized = {str(tag).strip().lower() for tag in excluded_tags}
        return "".join(node.xml for node in self.nodes if node.tag not in normalized).strip()


@dataclass(slots=True)
class OutputXMLStreamEvent:
    """表示流式 XML 解析过程中产出的一条结构化事件。

    流式解析与整段解析的差异在于：调用方往往既关心“某个标签开始了”
    这样的时机信号，也关心“某个完整节点已经结束，可以消费其文本内容”
    的最终结果。因此该对象会同时记录事件阶段和对应节点的关键信息。

    Attributes:
        phase: 事件阶段，取值为 `start` 或 `end`。
        tag: 归一化后的标签名。
        text: 当阶段为 `end` 时，对应节点的聚合文本内容。
        attrs: 当前节点的属性字典。
        xml: 当阶段为 `end` 时，对应节点重新序列化后的 XML 文本。
    """

    phase: str
    tag: str
    text: str = ""
    attrs: dict[str, str] = field(default_factory=dict)
    xml: str = ""


class StreamingOutputXMLParser:
    """基于 `XMLPullParser` 的流式结构化输出解析器。

    该解析器面向“模型输出被切成多个 chunk 逐步到达”的场景。调用方每收到一段
    文本，就调用一次 `feed()`；解析器会在顶层标签开始或结束时产出事件，供上层
    决定是否发送消息、记录 note 或触发思考中副作用。若后续喂入的内容破坏了
    XML 结构，`feed()` 或 `finalize()` 会显式抛出 `OutputXMLParseError`，由
    上层回退到更宽松的旧状态机逻辑。
    """

    def __init__(self) -> None:
        """初始化一个可持续增量喂入文本的 XML pull 解析器。

        解析器内部会预先写入一个虚拟 `<root>` 起始标签，用于把协议规定的
        “多个并列顶层标签片段”转换为标准 XML 文档结构。调用方在流结束时只需
        调用 `finalize()`，无需自行补根节点结束标签。
        """

        self._parser = ET.XMLPullParser(events=("start", "end"))
        self._parser.feed("<root>")
        self._stack: list[str] = []

    def has_open_tag(self) -> bool:
        """判断当前是否仍处于某个顶层输出标签的内部。

        该状态可帮助上层决定是否可以把“已成功处理过的原始流文本”从重放缓冲区
        中裁掉，以降低 XML 失败后 fallback 重放时出现重复发送的概率。

        Returns:
            当仍有未闭合的输出标签时返回 `True`，否则返回 `False`。
        """

        return bool(self._stack)

    def feed(self, chunk: str) -> list[OutputXMLStreamEvent]:
        """向流式 XML 解析器增量喂入一段模型输出文本。

        该方法会把当前 chunk 追加到解析器内部状态，并返回本次新增产生的所有
        顶层事件。对于被切开的开始标签、结束标签或属性值，解析器会自动等待后续
        chunk 补全，不会因为“标签尚未结束”而立即报错。

        Args:
            chunk: 本次新增的原始文本片段。

        Returns:
            本次新增解析得到的流式事件列表；若本次没有完整事件，则返回空列表。

        Raises:
            OutputXMLParseError: 当新增片段破坏整体 XML 结构时抛出。
        """

        if not chunk:
            return []

        self._parser.feed(chunk)
        return self._drain_events()

    def finalize(self) -> list[OutputXMLStreamEvent]:
        """在流结束时关闭虚拟根节点并取出剩余事件。

        调用方应在确认不会再收到更多 chunk 后调用本方法，以便让解析器检查
        是否还存在未闭合标签。若最终结构仍不合法，本方法会抛出解析异常，调用方
        可以据此决定是否使用 fallback 状态机重放最后一段未稳定文本。

        Returns:
            关闭根节点后新增产生的流式事件列表。

        Raises:
            OutputXMLParseError: 当流结束时 XML 结构仍未闭合或标签错配时抛出。
        """

        self._parser.feed("</root>")
        return self._drain_events()

    def _drain_events(self) -> list[OutputXMLStreamEvent]:
        """读取 pull 解析器当前已就绪的事件并转换为业务事件。

        该内部方法会把 `ElementTree` 的 `start/end` 事件收敛成“仅关注顶层标签”
        的业务事件：当某个顶层标签开始时发出 `start`，当其完整结束时发出 `end`。
        内部若出现轻微嵌套，也会通过栈结构保证只在最外层节点结束时产出完整文本。

        Returns:
            当前所有已就绪的业务事件列表。
        """

        events: list[OutputXMLStreamEvent] = []
        for raw_event in self._parser.read_events():
            if len(raw_event) != 2:
                continue

            phase, raw_element = raw_event
            if not isinstance(raw_element, ET.Element):
                continue

            element = cast(ET.Element, raw_element)
            if not isinstance(element.tag, str):
                continue

            tag = str(element.tag).strip().lower()
            if tag == "root":
                continue

            if phase == "start":
                self._stack.append(tag)
                if len(self._stack) == 1:
                    events.append(
                        OutputXMLStreamEvent(
                            phase="start",
                            tag=tag,
                            attrs={str(key): str(value) for key, value in element.attrib.items()},
                        )
                    )
                continue

            if not self._stack:
                element.clear()
                continue

            self._stack.pop()
            if not self._stack:
                node = OutputXMLNode.from_element(element)
                events.append(
                    OutputXMLStreamEvent(
                        phase="end",
                        tag=node.tag,
                        text=node.text,
                        attrs=node.attrs,
                        xml=node.xml,
                    )
                )
            element.clear()

        return events


def parse_output_xml_fragment(content: str) -> ParsedOutputXMLDocument:
    """把模型输出的 XML 片段解析为顶层节点序列。

    该函数假定调用方传入的是“若干并列标签组成的 XML 片段”，而不是完整 XML
    文档，因此内部会自动包裹一个虚拟 `<root>` 节点。若模型输出整体不满足
    XML 语法约束，例如标签错配、属性引号不闭合或出现非法字符，则会抛出
    `OutputXMLParseError`，由上层决定是否回退到旧解析逻辑。

    Args:
        content: 待解析的 XML 片段文本。

    Returns:
        解析完成后的 `ParsedOutputXMLDocument`。

    Raises:
        OutputXMLParseError: 当输入片段不是合法 XML 片段时抛出。
    """

    wrapped = f"<root>{content}</root>"
    root = ET.fromstring(wrapped)
    nodes = [
        OutputXMLNode.from_element(child)
        for child in root
        if isinstance(child.tag, str)
    ]
    return ParsedOutputXMLDocument(nodes=nodes)


__all__ = [
    "ALLOWED_OUTPUT_TAGS",
    "OutputXMLNode",
    "OutputXMLParseError",
    "OutputXMLStreamEvent",
    "ParsedOutputXMLDocument",
    "StreamingOutputXMLParser",
    "parse_output_xml_fragment",
]
