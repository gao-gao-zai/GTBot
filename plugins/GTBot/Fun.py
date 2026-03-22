import hashlib
from .Logger import logger
from sympy import im
import aiofiles
import asyncio
import re
from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from typing import Optional, Dict, Any, List
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.event import Event





async def file_to_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """
    异步计算文件的SHA-256哈希值
    
    参数:
        file_path: 文件路径
        chunk_size: 读取块大小(默认64KB)
    
    返回:
        SHA-256哈希值的十六进制字符串
    """
    sha256 = hashlib.sha256()
    
    try:
        # 异步打开文件
        async with aiofiles.open(file_path, 'rb') as f:
            # 分块读取文件内容
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
                
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except Exception as e:
        raise RuntimeError(f"处理文件时出错: {str(e)}")
    
    return sha256.hexdigest()


async def get_image_aspect_ratio(file_path: str) -> float:
    """
    获取图片的宽高比
    
    参数:
        file_path: 图片文件路径
        
    返回:
        float: 宽高比 (宽度/高度)
        
    异常:
        FileNotFoundError: 文件不存在
        RuntimeError: 处理文件时出错或无法获取图片尺寸
    """
    try:
        from PIL import Image
        import io
        
        # 使用Pillow库打开图片获取尺寸
        async with aiofiles.open(file_path, 'rb') as f:
            img_data = await f.read()
        
        with Image.open(io.BytesIO(img_data)) as img:
            width, height = img.size
            return width / height if height > 0 else 0.0
            
    except ImportError:
        raise RuntimeError("未安装Pillow库，无法获取图片尺寸")
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except Exception as e:
        raise RuntimeError(f"获取图片宽高比时出错: {str(e)}")

def parse_cq_codes(text):
    """
    解析文本中的CQ码并提取其类型和参数
    
    CQ码是一种特殊的文本格式，用于表示富文本内容（如图片、表情等），
    格式为：[CQ:type,param1=value1,param2=value2,...]
    
    Args:
        text (str): 包含CQ码的文本字符串
        
    Returns:
        list: 包含解析后CQ码信息的字典列表
              每个字典包含：
              - "CQ": CQ码的类型（如 "image", "face" 等）
              - 其他键值对: CQ码的参数和对应的值
              
    Example:
        >>> text = "[CQ:image,file=example.jpg,url=http://example.com] [CQ:face,id=123]"
        >>> parse_cq_codes(text)
        [
            {"CQ": "image", "file": "example.jpg", "url": "http://example.com"},
            {"CQ": "face", "id": "123"}
        ]
        
    Note:
        - 函数能够正确处理参数值中包含逗号的情况
        - 会自动去除键和值两端的空白字符
        - 如果CQ码没有参数，则只返回包含类型的字典
    """
    # 匹配所有[CQ:...]结构的文本
    cq_matches = re.findall(r'\[CQ:([^\]]+)\]', text)
    result = []
    
    for content in cq_matches:
        # 分割类型和参数部分
        parts = content.split(',', 1)
        cq_type = parts[0].strip()
        cq_dict = {"CQ": cq_type}
        
        # 如果存在参数部分
        if len(parts) > 1:
            params_str = parts[1]
            # 解析键值对（处理值中含逗号的情况）
            current_key = None
            for segment in re.split(r',\s*(?=[^=]+=)', params_str):
                if '=' in segment:
                    # 遇到新键值对
                    key, value = segment.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    cq_dict[key] = value
                    current_key = key
                elif current_key is not None:
                    # 值中含逗号的情况，追加到上一个值
                    cq_dict[current_key] += ',' + segment
        
        result.append(cq_dict)
    
    return result

def generate_cq_code(cq_dict_list: list[dict]) -> list[str]:
    """
    根据提供的CQ码参数字典列表生成CQ码字符串列表
    
    Args:
        cq_dict_list (list): 包含CQ码参数的字典列表，每个字典必须包含"CQ"键表示类型
        
    Returns:
        list: CQ码字符串列表
        
    Example:
        >>> cq_list = [
                {"CQ": "image", "file": "example.jpg", "url": "http://example.com"},
                {"CQ": "face", "id": "123"}
            ]
        >>> generate_cq_code(cq_list)
        [
            "[CQ:image,file=example.jpg,url=http://example.com]",
            "[CQ:face,id=123]"
        ]
    """
    result = []
    for cq_dict in cq_dict_list:
        # 确保包含CQ类型
        if "CQ" not in cq_dict:
            continue
            
        cq_type = cq_dict["CQ"]
        parts = [f"CQ:{cq_type}"]
        
        # 添加参数部分（排除CQ键）
        for key, value in cq_dict.items():
            if key == "CQ":
                continue
            # 处理值中的特殊字符（逗号不需要转义，但右方括号需要转义）
            value = str(value).replace(']', '\\]')
            parts.append(f"{key}={value}")
        
        result.append(f"[{','.join(parts)}]")
    
    return result

def replace_cq_codes(text: str, replace_func):
    """
    替换文本中的CQ码为新的文本，当替换函数返回None时保留原始CQ码
    
    Args:
        text (str): 原始文本
        replace_func (function): 替换函数，接受CQ参数字典，返回替换后的文本或None
        
    Returns:
        str: 替换后的文本
        
    Example:
        >>> text = "Hello [CQ:face,id=123] World"
        >>> def my_replacer(cq_dict):
                if cq_dict['CQ'] == 'face':
                    return f"(表情#{cq_dict['id']})"
                # 其他类型返回None表示保留
        >>> replace_cq_codes(text, my_replacer)
        "Hello (表情#123) World"
        
        >>> def remove_at(cq_dict):
                if cq_dict['CQ'] == 'at':
                    return ""  # 删除@消息
        >>> text = "请[CQ:at,qq=123]查看"
        >>> replace_cq_codes(text, remove_at)
        "请查看"
    """
    pattern = r'(\[CQ:[^\]]+\])'
    
    def replace_match(match):
        full_match = match.group(0)
        cq_str = full_match[4:-1]  # 去掉开头的"[CQ:"和结尾的"]"
        parts = cq_str.split(',', 1)
        cq_dict = {"CQ": parts[0].strip()}
        
        if len(parts) > 1:
            for segment in re.split(r',\s*(?=[^=]+=)', parts[1]):
                if '=' in segment:
                    key, value = segment.split('=', 1)
                    cq_dict[key.strip()] = value.strip().replace('\\]', ']')
        
        # 调用替换函数，如果返回None则保留原始CQ码
        replacement = replace_func(cq_dict)
        return replacement if replacement is not None else full_match
    
    return re.sub(pattern, replace_match, text)


def _unescape_cq_value(value: str) -> str:
    """反转义CQ码参数值中的特殊字符"""
    return value.replace('&#44;', ',') \
                .replace('&#91;', '[') \
                .replace('&#93;', ']') \
                .replace('&amp;', '&')

def _escape_cq_value(value: Any) -> str:
    """转义CQ码参数值"""
    return str(value).replace('&', '&amp;') \
                     .replace('[', '&#91;') \
                     .replace(']', '&#93;') \
                     .replace(',', '&#44;')

def parse_single_cq(cq_code_str: str) -> Dict[str, str]:
    """
    解析单个CQ码字符串为字典
    输入: "[CQ:at,qq=123456]"
    输出: {'CQ': 'at', 'qq': '123456'}
    """
    # 去除首尾的 [ ]
    content = cq_code_str[1:-1]
    
    # 分割类型和参数部分
    parts = content.split(',', 1)
    cq_type = parts[0].split(':')[1].strip() # 取出 CQ: 后面的类型
    cq_dict = {"CQ": cq_type}
    
    if len(parts) > 1:
        params_str = parts[1]
        current_key = None
        # 使用正则分割参数，逻辑同你提供的 parse_cq_codes
        for segment in re.split(r',\s*(?=[^=]+=)', params_str):
            if '=' in segment:
                key, val = segment.split('=', 1)
                key = key.strip()
                val = val.strip()
                cq_dict[key] = _unescape_cq_value(val)
                current_key = key
            elif current_key is not None:
                # 处理值中包含逗号的情况
                cq_dict[current_key] += ',' + _unescape_cq_value(segment)
                
    return cq_dict

def generate_cq_string(type_: str, data: Dict[str, Any]) -> str:
    """
    根据类型和数据生成CQ码字符串
    """
    parts = [f"CQ:{type_}"]
    for key, value in data.items():
        # 过滤掉不需要的字段或二进制数据
        if value is None or key == 'type': 
            continue
        parts.append(f"{key}={_escape_cq_value(value)}")
    return f"[{','.join(parts)}]"

# ==========================================
# 核心功能函数
# ==========================================

def _is_cq_type_allowed(
    cq_type: str,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None
) -> bool:
    """
    判断 CQ 码类型是否被允许。
    
    过滤规则：
    - 如果同时提供白名单和黑名单，白名单优先级更高
    - 仅提供白名单时，只有白名单中的类型被允许
    - 仅提供黑名单时，黑名单中的类型被过滤
    - 都不提供时，所有类型被允许
    
    Args:
        cq_type: CQ 码类型（如 'at', 'image', 'face' 等）
        whitelist: CQ 码类型白名单，仅处理列表中的类型
        blacklist: CQ 码类型黑名单，过滤列表中的类型
        
    Returns:
        bool: True 表示该类型被允许，False 表示被过滤
    """
    # 白名单优先
    if whitelist is not None:
        return cq_type in whitelist
    
    # 黑名单过滤
    if blacklist is not None:
        return cq_type not in blacklist
    
    # 都不提供，全部允许
    return True


async def text_to_message(
    text: str,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None
) -> "Message":
    """
    将文本信息转换为消息对象，并转义CQ码。
    
    支持通过白名单和黑名单过滤 CQ 码类型。被过滤的 CQ 码将作为纯文本保留。
    
    Args:
        text: 包含 CQ 码的文本字符串
        whitelist: CQ 码类型白名单，仅处理列表中的类型。
            例如 ['at', 'image'] 表示只解析 at 和 image 类型的 CQ 码
        blacklist: CQ 码类型黑名单，过滤列表中的类型。
            例如 ['face', 'record'] 表示不解析 face 和 record 类型的 CQ 码
            
    Returns:
        Message: 转换后的消息对象
        
    Note:
        - 如果同时提供白名单和黑名单，白名单优先级更高
        - 被过滤或解析失败的 CQ 码将保留为纯文本形式
        
    Example:
        >>> # 只解析 at 和 image
        >>> await text_to_message(text, whitelist=['at', 'image'])
        >>> # 过滤 face 表情
        >>> await text_to_message(text, blacklist=['face'])
    """
    # 1. 使用 re.split 保留分隔符模式，这样列表会变成 [文本, CQ码, 文本, CQ码...]
    # 匹配模式：[CQ:...]
    pattern = r'(\[CQ:[^\]]+\])'
    parts = re.split(pattern, text)
    
    segments = []
    
    for part in parts:
        if not part:
            continue
            
        # 2. 判断是否为 CQ 码
        if part.startswith('[CQ:') and part.endswith(']'):
            try:
                cq_data = parse_single_cq(part)
                cq_type = cq_data.pop('CQ')  # 取出类型，剩下的是参数
                
                # 检查 CQ 类型是否被允许
                if not _is_cq_type_allowed(cq_type, whitelist, blacklist):
                    # 不在允许范围内，作为纯文本保留
                    segments.append(MessageSegment.text(part))
                    continue
                
                # 3. 使用映射表处理不同类型的 Segment (工厂模式)
                # 注意：这里假设 MessageSegment 有对应的方法，且参数名匹配
                # 如果参数名不匹配，需要单独处理
                if cq_type == 'at':
                    qq_id = cq_data.get('qq', '')
                    if qq_id:
                        segments.append(MessageSegment.at(qq_id))
                elif cq_type == 'face':
                    face_id = cq_data.get('id', '')
                    if face_id and face_id.isdigit():
                        segments.append(MessageSegment.face(int(face_id)))
                    else:
                        segments.append(MessageSegment.text(part))
                elif cq_type == 'image':
                    image_url = cq_data.get('url') or cq_data.get('file', '')
                    if image_url:
                        segments.append(MessageSegment.image(file=image_url))
                    else:
                        segments.append(MessageSegment.text(part))
                elif cq_type == 'record':
                    segments.append(MessageSegment.record(file=cq_data.get('file', '')))
                elif cq_type == 'video':
                    segments.append(MessageSegment.video(file=cq_data.get('file', '')))
                elif cq_type == 'reply':
                    if 'id' in cq_data:
                        segments.append(MessageSegment.reply(int(cq_data['id'])))
                elif cq_type == 'json':
                    segments.append(MessageSegment.json(cq_data.get('data', '')))
                else:
                    # 未知类型，尝试作为通用类型处理或保留原文本
                    # segments.append(MessageSegment(type=cq_type, data=cq_data))
                    segments.append(MessageSegment.text(part))
                    
            except Exception:
                # 解析失败（如 id 不是数字），回退为纯文本
                segments.append(MessageSegment.text(part))
        else:
            # 4. 普通文本
            segments.append(MessageSegment.text(part))

    return Message(segments)


async def message_to_text(
    message: "Message", 
    event: Optional["Event"] = None, 
    bot: Optional["Bot"] = None, 
    message_id: Optional[int] = None,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None
) -> str:
    """
    将消息对象转换为文本信息，并转义CQ码。
    
    支持通过白名单和黑名单过滤 CQ 码类型。被过滤的消息段将被忽略（不输出）。
    
    Args:
        message: 消息对象
        event: 事件对象，用于备用获取回复消息
        bot: Bot 对象，用于获取回复消息详情
        message_id: 消息 ID，用于获取回复消息
        whitelist: CQ 码类型白名单，仅输出列表中的类型。
            例如 ['text', 'at'] 表示只输出文本和 at 类型
        blacklist: CQ 码类型黑名单，过滤列表中的类型。
            例如 ['image', 'record'] 表示不输出图片和语音
            
    Returns:
        str: 转换后的文本字符串
        
    Note:
        - 此函数为 async，因为获取回复消息涉及网络请求
        - 如果同时提供白名单和黑名单，白名单优先级更高
        - 'text' 类型默认总是被输出（除非在黑名单中明确指定）
        - reply 类型的过滤会影响回复消息的输出
        
    Example:
        >>> # 只输出文本和 at
        >>> await message_to_text(message, whitelist=['text', 'at'])
        >>> # 过滤图片和语音
        >>> await message_to_text(message, blacklist=['image', 'record'])
    """
    result_parts = []
    
    # 1. 处理回复消息 (Reply)
    # 逻辑：优先通过 bot+message_id 获取，其次通过 event 描述获取
    reply_cq = ""
    
    # 检查 reply 类型是否被允许
    if _is_cq_type_allowed('reply', whitelist, blacklist):
        if message_id and bot:
            try:
                # 必须 await
                reply_msg = await bot.get_msg(message_id=message_id)
                raw_msg = reply_msg.get("raw_message", "")
                # 尝试从原始消息中提取 reply 结构
                # 注意：OneBot 实现中 reply 通常作为 segment 存在，但也可能在 raw_message 中
                match = re.search(r'\[CQ:reply,id=(-?\d+)\]', raw_msg)
                if match:
                    reply_cq = match.group(0)
            except Exception:
                pass  # 获取失败忽略
                
        elif event:
            # 备用方案：从事件描述字符串中提取
            raw_msg = str(event.get_event_description())
            # 这里的正则根据具体框架的 toString 实现可能需要调整
            match = re.search(r'\[reply:id=(-?\d+)\]', raw_msg)
            if match:
                reply_id = match.group(1)
                reply_cq = f"[CQ:reply,id={reply_id}]"

    if reply_cq:
        result_parts.append(reply_cq)

    # 2. 遍历消息段并转换
    for segment in message:
        segment_type = segment.type
        
        # 检查类型是否被允许
        if not _is_cq_type_allowed(segment_type, whitelist, blacklist):
            continue  # 跳过不允许的类型
        
        if segment_type == "text":
            result_parts.append(segment.data["text"])
        else:
            # 直接利用 segment.data 生成 CQ 码
            # 某些特殊的 segment 可能需要清洗 data 字段
            data = segment.data.copy()
            
            # 针对特定类型的字段清理 (可选)
            if segment_type in ["image", "record", "video", "file"]:
                # 确保只保留 file 和 file_size 等关键字段，或者全部保留
                pass 
                
            cq_code = generate_cq_string(segment_type, data)
            result_parts.append(cq_code)

    return ''.join(result_parts)


# ==========================================
# 消息格式化函数（按模板格式化消息列表）
# ==========================================

from datetime import datetime, time
from typing import List, Union, TYPE_CHECKING, Any


_CHAT_LOG_PREFIX_PATTERN = re.compile(
    r"^\[\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*.+?\([^\)]*\):\s*"
)
"""用于匹配聊天记录格式前缀的正则。

示例前缀：
    [02-04 21:27:07] 天天(3652078196):
    [02-04 21:27:07] 昵称(用户ID, 消息ID):
"""


def strip_chat_log_prefix(text: str) -> str:
    """剥离聊天记录格式前缀，避免被当作正文发送。

    部分模型会“复读”输入上下文的格式（例如："[MM-DD HH:MM:SS] 昵称(qq):正文"），
    这会导致 QQ 端看到像是程序自动加了时间戳/昵称的前缀。实际上该前缀通常来自模型输出。

    本函数用于在发送前做一次保守清洗：
    - 仅当文本**以** "[MM-DD HH:MM:SS]" 开头且包含 "昵称(...):" 结构时才剥离。
    - 不匹配则原样返回。

    Args:
        text: 原始文本。

    Returns:
        清洗后的文本。
    """
    stripped, _hit = strip_chat_log_prefix_with_hit(text)
    return stripped


def strip_chat_log_prefix_with_hit(text: str) -> tuple[str, bool]:
    """剥离聊天记录格式前缀，并返回是否命中。

    Args:
        text: 原始文本。

    Returns:
        tuple[str, bool]: (清洗后的文本, 是否命中并发生剥离)。
    """
    if not text:
        return "", False

    match = _CHAT_LOG_PREFIX_PATTERN.match(text)
    if not match:
        return text, False

    return text[match.end():].lstrip(), True

if TYPE_CHECKING:
    from .CacheManager import UserCacheManager
    from .model import GroupMessage


def truncate_message(text: str, max_length: int, suffix: str = "...") -> str:
    """
    截断超长消息文本。
    
    Args:
        text: 原始消息文本
        max_length: 最大允许长度（字符数）。0 或负数表示不限制
        suffix: 截断后的后缀，默认为 "..."
        
    Returns:
        str: 截断后的文本。如果原文本未超长则返回原文本
        
    Example:
        >>> truncate_message("这是一条很长的消息", 5)
        "这是一条很..."
    """
    if max_length <= 0 or len(text) <= max_length:
        return text
    return text[:max_length] + suffix


async def format_messages_to_text_list(
    messages: List["GroupMessage"],
    template: str = "[[$time_M]-[$time_d] [$time_h]:[$time_m]:[$time_s]] [$user_name]([$user_id]):[$message]",
    max_message_length: int = 0,
    *,
    bot: Optional[Bot] = None,
    cache: Optional["UserCacheManager"] = None,
    self_id: Optional[int] = None,
) -> List[str]:
    """将消息列表按模板格式化为纯文本列表。

    说明：本函数不依赖消息对象中的 `user_name` 字段。若提供缓存管理器，
    将优先通过缓存系统获取群名片/昵称；否则使用 `user_id` 作为占位。

    Args:
        messages: 消息对象列表。
        template: 格式化模板，支持占位符：
            - ``[$time_Y]``/``[$time_M]``/``[$time_d]``/``[$time_h]``/``[$time_m]``/``[$time_s]``: 时间。
            - ``[$user_id]``: 用户 QQ 号。
            - ``[$user_name]``: 用户显示名（通过缓存解析，不读取 `msg.user_name`）。
            - ``[$group_id]``: 群号。
            - ``[$message_id]``: 消息 ID（撤回则显示“已撤回”）。
            - ``[$message]``: 消息文本。
        max_message_length: 单条消息最大长度（字符数），0 表示不限制。
        bot: OneBot Bot 实例（用于缓存查询）。
        cache: 用户缓存管理器（用于解析显示名）。
        self_id: 机器人用户 ID（若提供且命中，则在显示名后追加“(我)”）。

    Returns:
        格式化后的文本列表。
    """
    result = []
    
    for msg in messages:
        # 解析时间
        dt = datetime.fromtimestamp(msg.send_time)
        
        # 获取消息内容（通过 message_to_text 处理 CQ 码）
        # 如果 content 是 Message 对象则转换，否则直接使用字符串
        if hasattr(msg.content, '__iter__') and not isinstance(msg.content, str):
            # 是 Message 对象
            message_text = await message_to_text(msg.content)
        else:
            # 已经是字符串
            message_text = str(msg.content)
        
        # 截断超长消息
        message_text = truncate_message(message_text, max_message_length)

        display_name = str(msg.user_id)
        if cache is not None and bot is not None:
            try:
                if getattr(msg, "group_id", 0):
                    display_name = await cache.get_group_member_name(
                        bot, int(msg.group_id), int(msg.user_id)
                    )
                else:
                    display_name = await cache.get_user_name(bot, int(msg.user_id))
            except Exception:
                display_name = str(msg.user_id)

        if self_id is not None and int(msg.user_id) == int(self_id):
            display_name = f"{display_name}(我)"
        
        # 替换模板中的占位符
        formatted = template
        formatted = formatted.replace("[$time_Y]", dt.strftime("%Y"))
        formatted = formatted.replace("[$time_M]", dt.strftime("%m"))
        formatted = formatted.replace("[$time_d]", dt.strftime("%d"))
        formatted = formatted.replace("[$time_h]", dt.strftime("%H"))
        formatted = formatted.replace("[$time_m]", dt.strftime("%M"))
        formatted = formatted.replace("[$time_s]", dt.strftime("%S"))
        formatted = formatted.replace("[$user_id]", str(msg.user_id))
        formatted = formatted.replace("[$user_name]", display_name)
        formatted = formatted.replace("[$group_id]", str(msg.group_id))
        # 如果消息被撤回，将 message_id 解析为 "已撤回"
        formatted = formatted.replace(
            "[$message_id]", 
            "已撤回" if msg.is_withdrawn else str(msg.message_id)
        )
        formatted = formatted.replace("[$message]", message_text)
        
        result.append(formatted)
    
    return result


async def format_messages_to_text(
    messages: List["GroupMessage"],
    template: str = "[[$time_M]-[$time_d] [$time_h]:[$time_m]:[$time_s]] [$user_name]([$user_id]):[$message]",
    separator: str = "\n",
    max_message_length: int = 0,
    *,
    bot: Optional[Bot] = None,
    cache: Optional["UserCacheManager"] = None,
    self_id: Optional[int] = None,
) -> str:
    """
    将消息列表按照模板格式化为单个纯文本字符串。
    
    此函数用于将多条消息记录格式化为统一的文本格式，方便用于日志记录、
    上下文展示或发送给AI模型作为聊天记录。
    
    Args:
        messages: GroupMessage 对象列表
        template: 格式化模板字符串，支持以下占位符：
            - [$time_Y]: 年份（4位）
            - [$time_M]: 月份（2位）
            - [$time_d]: 日期（2位）
            - [$time_h]: 小时（2位，24小时制）
            - [$time_m]: 分钟（2位）
            - [$time_s]: 秒（2位）
            - [$user_id]: 用户QQ号
            - [$user_name]: 用户昵称
            - [$group_id]: 群号
            - [$message_id]: 消息ID
            - [$message]: 消息内容（已通过 message_to_text 处理）
        separator: 消息之间的分隔符，默认为换行符
        max_message_length: 单条消息内容的最大长度（字符数），
            超过此长度的消息会被截断并用 "..." 代替。
            0 或负数表示不限制长度。
            
    Returns:
        str: 格式化后的纯文本字符串
        
    Example:
        >>> messages = [GroupMessage(...), GroupMessage(...)]
        >>> template = "[[$time_h]:[$time_m]] [$user_name]: [$message]"
        >>> await format_messages_to_text(messages, template, max_message_length=100)
        "[14:30] 张三: 你好\n[14:31] 李四: 世界"
        
    Note:
        - 模板字符串可在 config_group.json 中配置，字段名为 "message_format_placeholder"
        - max_message_length 可在 config_group.json 的 chat_model 中配置
    """
    text_list = await format_messages_to_text_list(
        messages,
        template,
        max_message_length,
        bot=bot,
        cache=cache,
        self_id=self_id,
    )
    return separator.join(text_list)


# ==========================================
# Bot 消息操作工具函数
# ==========================================

async def delete_message(bot: Bot, message_id: int, delay: int = 0) -> Dict[str, Any]:
    """
    撤回消息。
    
    通过调用 LLOneBot API 撤回指定消息。注意：撤回他人消息需要管理员权限，
    且只能撤回2分钟内的消息。
    
    Args:
        bot: Bot 实例，用于调用 API
        message_id: 要撤回的消息 ID
        delay: 延迟撤回时间（秒），范围 0-60，默认为 0（立即撤回）
        
    Returns:
        Dict[str, Any]: API 返回结果，包含 status、retcode、message 等字段
        
    Raises:
        Exception: 当 API 调用失败时抛出异常
        ValueError: 当 delay 参数超出范围时抛出异常
        
    Example:
        >>> result = await delete_message(bot, 12345678)
        >>> print(result)
        {"status": "ok", "retcode": 0, "message": "", "wording": ""}
    """
    return await bot.call_api("delete_msg", message_id=message_id, delay=delay)


async def set_msg_emoji_like(
    bot: Bot, 
    message_id: int, 
    emoji_id: int,
    blocking: bool = True,
    timeout: float = 5
) -> Dict[str, Any] | asyncio.Task:
    """
    对消息进行表情回应（表情贴）。
    
    通过调用 LLOneBot API 对指定消息添加表情回应。
    注意：只支持群聊消息。
    
    Args:
        bot: Bot 实例，用于调用 API
        message_id: 要回应的消息 ID
        emoji_id: 表情 ID，QQ 表情对应的数字编号
        blocking: 是否堵塞
        timeout: 调用超时
        
    Returns:
        Dict[str, Any]: API 返回结果
        
    Raises:
        Exception: 当 API 调用失败时抛出异常
        
    Example:
        >>> # 对消息添加"赞"表情回应
        >>> result = await set_msg_emoji_like(bot, 12345678, 76)
    """

    async def call_with_timeout(coro: Any, timeout: float) -> Any:
        """在指定超时时间内执行协程。

        Args:
            coro: 要执行的协程对象。
            timeout: 超时时间（秒）。

        Returns:
            Any: 协程执行结果。

        Raises:
            asyncio.TimeoutError: 执行超时。
            Exception: 执行过程中发生的其他异常。
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError as e:
            logger.warning("发送表情贴超时")
            raise e
        except Exception as e:
            logger.exception(f"发送表情贴失败: {e}")
            raise e

    coro = bot.call_api(
            "set_msg_emoji_like", 
            message_id=message_id, 
            emoji_id=emoji_id
        )

    if blocking:
        return await call_with_timeout(coro, timeout)

    task = asyncio.create_task(call_with_timeout(coro, timeout))

    def _consume_task_exception(done_task: asyncio.Task) -> None:
        """回收后台表情贴任务的异常，避免未处理任务告警。"""

        try:
            done_task.result()
        except Exception:
            return

    task.add_done_callback(_consume_task_exception)
    return task


async def group_poke(
    bot: Bot, 
    group_id: int, 
    user_id: int
) -> Dict[str, Any]:
    """
    在群聊中戳一戳指定用户（双击头像）。
    
    通过调用 LLOneBot API 在群聊中对指定用户发送戳一戳消息。
    
    Args:
        bot: Bot 实例，用于调用 API
        group_id: 群号
        user_id: 要戳的用户 QQ 号
        
    Returns:
        Dict[str, Any]: API 返回结果
        
    Raises:
        Exception: 当 API 调用失败时抛出异常
        
    Example:
        >>> result = await group_poke(bot, 123456789, 987654321)
    """
    return await bot.call_api(
        "group_poke", 
        group_id=group_id, 
        user_id=user_id
    )

async def send_like(
    bot:Bot,
    user_id:int,
    times:int=10
):
    """
    发送点赞
    
    通过调用 LLOneBot API 发送点赞给指定用户。
    
    参数:
        bot: Bot 实例，用于调用 API
        user_id: 要点赞的用户 QQ 号
        times: 点赞次数，默认为10次
        
    返回:
        Dict[str, Any]: API 返回结果
        
    异常:
        Exception: 当 API 调用失败时抛出异常
        
    示例:
        >>> result = await send_like(bot, 123456789, times=5)
    """
    return await bot.call_api(
        "send_like",
        user_id=user_id,
        times=times
    )
