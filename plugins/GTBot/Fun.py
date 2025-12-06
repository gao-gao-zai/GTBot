import hashlib
from sympy import im
import aiofiles
import re
from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from typing import Optional, Dict, Any
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

def generate_cq_code(cq_dict_list):
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

async def text_to_message(text: str) -> "Message":
    """
    将文本信息转换为消息对象，并转义CQ码
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
                cq_type = cq_data.pop('CQ') # 取出类型，剩下的是参数
                
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
    message_id: Optional[int] = None
) -> str:
    """
    将消息对象转换为文本信息，并转义CQ码
    
    注意：此函数已改为 async，因为获取回复消息涉及网络请求
    """
    result_parts = []
    
    # 1. 处理回复消息 (Reply)
    # 逻辑：优先通过 bot+message_id 获取，其次通过 event 描述获取
    reply_cq = ""
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
            pass # 获取失败忽略
            
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
        if segment.type == "text":
            result_parts.append(segment.data["text"])
        else:
            # 直接利用 segment.data 生成 CQ 码
            # 某些特殊的 segment 可能需要清洗 data 字段
            data = segment.data.copy()
            
            # 针对特定类型的字段清理 (可选)
            if segment.type in ["image", "record", "video", "file"]:
                # 确保只保留 file 和 file_size 等关键字段，或者全部保留
                pass 
                
            cq_code = generate_cq_string(segment.type, data)
            result_parts.append(cq_code)

    return ''.join(result_parts)


# ==========================================
# 消息格式化函数（按模板格式化消息列表）
# ==========================================

from datetime import datetime
from typing import List, Union, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import GroupMessage


async def format_messages_to_text_list(
    messages: List["GroupMessage"],
    template: str = "[[$time_M]-[$time_d] [$time_h]:[$time_m]:[$time_s]] [$user_name]([$user_id]):[$message]"
) -> List[str]:
    """
    将消息列表按照模板格式化为纯文本列表
    
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
            - [$message]: 消息内容（CQ码已转义）
            
    Returns:
        List[str]: 格式化后的文本列表，每条消息对应一个元素
        
    Example:
        >>> messages = [GroupMessage(...), GroupMessage(...)]
        >>> template = "[[$time_h]:[$time_m]] [$user_name]: [$message]"
        >>> await format_messages_to_text_list(messages, template)
        ["[14:30] 张三: 你好", "[14:31] 李四: 世界"]
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
        
        # 替换模板中的占位符
        formatted = template
        formatted = formatted.replace("[$time_Y]", dt.strftime("%Y"))
        formatted = formatted.replace("[$time_M]", dt.strftime("%m"))
        formatted = formatted.replace("[$time_d]", dt.strftime("%d"))
        formatted = formatted.replace("[$time_h]", dt.strftime("%H"))
        formatted = formatted.replace("[$time_m]", dt.strftime("%M"))
        formatted = formatted.replace("[$time_s]", dt.strftime("%S"))
        formatted = formatted.replace("[$user_id]", str(msg.user_id))
        formatted = formatted.replace("[$user_name]", msg.user_name)
        formatted = formatted.replace("[$group_id]", str(msg.group_id))
        formatted = formatted.replace("[$message_id]", str(msg.message_id))
        formatted = formatted.replace("[$message]", message_text)
        
        result.append(formatted)
    
    return result


async def format_messages_to_text(
    messages: List["GroupMessage"],
    template: str = "[[$time_M]-[$time_d] [$time_h]:[$time_m]:[$time_s]] [$user_name]([$user_id]):[$message]",
    separator: str = "\n"
) -> str:
    """
    将消息列表按照模板格式化为单个纯文本字符串
    
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
            
    Returns:
        str: 格式化后的纯文本字符串
        
    Example:
        >>> messages = [GroupMessage(...), GroupMessage(...)]
        >>> template = "[[$time_h]:[$time_m]] [$user_name]: [$message]"
        >>> await format_messages_to_text(messages, template)
        "[14:30] 张三: 你好\n[14:31] 李四: 世界"
        
    Note:
        模板字符串可在 config_group.json 中配置，字段名为 "message_format_placeholder"
    """
    text_list = await format_messages_to_text_list(messages, template)
    return separator.join(text_list)