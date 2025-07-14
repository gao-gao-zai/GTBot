import asyncio
import re
from typing import Any, Dict, Optional
from fastapi import HTTPException
import sys
import aiohttp
import aiofiles
# from pydub import AudioSegment
import nonebot
import os
import base64
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, MessageSegment, GroupMessageEvent, Message
from nonebot.exception import MatcherException
from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot import get_driver, on_command
import asyncio
import json
import asyncio
import hashlib
import re

dir_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")




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
    
    return result  # 添加了缺失的return语句



class toolbox:
    
    
    
    @staticmethod
    async def send_image(command, image_path: str, end: bool = False):
        try:
            # 检查支持的图片格式
            supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            _, ext = os.path.splitext(image_path)
            if ext.lower() not in supported_extensions:
                await command.send("不支持的图片格式。")
                return

            # 使用文件路径发送图片
            img_segment = MessageSegment.image(image_path)
            if end:
                await command.finish(img_segment, at_sender=True)
            else:
                await command.send(img_segment)

        except MatcherException:
            raise  # 确保 MatcherException 能够正常传递，以便框架处理结束逻辑

        except Exception as e:
            # 这里处理所有非 MatcherException 类型的异常
            await command.send(f"发送图片时出错: {str(e)}")


    @staticmethod
    async def send_voice(command, voice_path: str, end: bool = False):
        """
        使用 onebotv11适配器发送语音
        参数：
        - command: 命令对象
        - voice_path: 语音文件路径
        返回：
        - None
        """
        
        try:
            message = MessageSegment.record(voice_path)
            if end:
                await command.finish(message, at_sender=True)
            else:
                await command.send(message)
        except Exception as e:
            await command.send(f"发送语音时出错: {str(e)}")


    @staticmethod
    async def get_qqname(qq: int, bot: Bot, event: Event, group: Optional[bool] = None, no_cache: bool = False) -> str:
        """
        获取QQ昵称(异步函数)
        ---
        参数:
        - qq: int 目标qq的qq号
        - bot: bot 
        - event: Event
        - group: bool 是否获取群昵称，不填则默认尝试获取群昵称
        - no_cache: 是否不使用缓存（使用缓存可能更新不及时，但响应更快）
        """
        if group is None:
            name = "未知"
            try:
                if isinstance(event, GroupMessageEvent):
                    try:
                        group_id = int(event.get_session_id().split('_')[1])
                        name = (await bot.get_group_member_info(group_id=group_id, user_id=qq, no_cache=no_cache))["card"]
                    except:
                        name = (await bot.get_stranger_info(user_id=qq, no_cache=no_cache))["nickname"]
                else:
                    name = (await bot.get_stranger_info(user_id=qq, no_cache=no_cache))["nickname"]
            except:
                pass
        elif group == True:
            name = "未知"
            try:
                group_id = int(event.get_session_id().split('_')[1])
                name = (await bot.get_group_member_info(group_id=group_id, user_id=qq, no_cache=no_cache))["card"]
            except:
                pass
        else:
            name = "未知"
            try:
                name = (await bot.get_stranger_info(user_id=qq, no_cache=no_cache))["nickname"]
            except:
                pass
        return name

    @staticmethod
    async def get_group_name(group_id: int, bot: Bot) -> str:
        """
        获取群名(异步函数)
        ---
        参数:
        - group_id: int 目标群的群号
        - bot: bot 
        """
        name = str(group_id)
        try:
            name = (await bot.get_group_info(group_id=group_id))["group_name"]
        except:
            pass
        return name

    @staticmethod
    async def text_to_message(text: str) -> Message:
        """
        将文本信息转换为消息对象，并转义CQ码
        """
        # 正则表达式来匹配各种消息段格式
        pattern = r'\[CQ:(\w+),([^\]]+)\]'
        
        segments = []
        last_end = 0
        
        # 遍历所有匹配项
        for match in re.finditer(pattern, text):
            # 添加前一个匹配和当前匹配之间的普通文本
            start, end = match.span()
            segments.append(text[last_end:start])
            
            # 解析消息类型和参数
            segment_type = match.group(1)
            params = match.group(2)
            param_dict = {param.split('=')[0]: param.split('=')[1] for param in params.split(',')}

            # 根据类型添加不同的消息段
            if segment_type == 'at':
                segments.append(MessageSegment.at(param_dict['qq']))
            elif segment_type == 'face':
                # 将字符串类型的 id 转换为整数
                face_id = int(param_dict['id'])
                segments.append(MessageSegment.face(face_id))
            elif segment_type == 'image':
                segments.append(MessageSegment.image(param_dict['file']))
            elif segment_type == 'record':
                segments.append(MessageSegment.record(param_dict['file']))
            elif segment_type == 'video':
                segments.append(MessageSegment.video(param_dict['file']))
            elif segment_type == 'reply':
                try: # 防止无法将id解析为int
                    reply_id = int(param_dict['id'])
                    segments.append(MessageSegment.reply(reply_id))
                except:
                    pass
            else:
                # 对于未知的类型，可以简单地添加文本表示或者忽略
                segments.append(MessageSegment.text(f"[CQ:{segment_type},{params}]"))
            
            last_end = end

        
        
        # 添加最后一个匹配之后的文本
        segments.append(text[last_end:])
        
        # 创建消息对象
        return Message(segments)
    
    


    @staticmethod
    def message_to_text(message: Message, event: Event | None = None, bot: Bot | None = None, message_id: int | None = None) -> str:
        """
        将消息对象转换为文本信息，并转义CQ码\n
        
        没有 event | bot+messageid 则不解析回复(引用)消息
        ---
        messageid > event
        
        message: 消息对象\n
        event: 事件对象，用于获取用户回复
        bot: bot对象，用于获取用户回复
        message_id: 消息id，用于获取用户回复
        """
        segments = []
        # 遍历消息中的每个段落
        for segment in message:
            segment_type = segment.type
            data = segment.data
            
            if segment_type == "text":
                segments.append(data["text"])
            elif segment_type == "at":
                segments.append(f"[CQ:at,qq={data['qq']}]")
            elif segment_type == "face":
                segments.append(f"[CQ:face,id={data['id']}]")
            elif segment_type in ["image", "record", "video"]:
                # 对于 image、record、video 类型，仅包含文件标识，忽略url
                if 'file' in data:
                    segments.append(f"[CQ:{segment_type},file={data['file']}, file_size={data['file_size']}]")
                else:
                    # 如果没有文件信息，则仅显示类型
                    segments.append(f"[CQ:{segment_type}]")
            elif segment_type == "file":
                segments.append(f"[CQ:file,file={data['file']}, file_size={data['file_size']}]")
            elif segment_type == "reply":
                segments.append(f"[CQ:reply,id={data['id']}]")
            else:
                # 对于未知的类型，可以简单地添加文本表示或者忽略
                segments.append(f"[CQ:{segment_type}]")

        # 处理回复消息：
        if message_id and bot:
            reply_msg = asyncio.run(bot.get_msg(message_id=message_id))
            raw_msg = reply_msg["raw_message"]
            reply_match = re.search(r'\[reply:id=(-?\d+)\]', raw_msg)
            if reply_match:
                reply_id = reply_match.group(1)
                segments.insert(0, f"[CQ:reply,id={reply_id}]")
        elif event:
            # 获取原始消息文本
            raw_msg = str(event.get_event_description())
            # 使用正则表达式查找回复消息的ID
            reply_match = re.search(r'\[reply:id=(-?\d+)\]', raw_msg)
            if reply_match:
                reply_id = reply_match.group(1)
                segments.insert(0, f"[CQ:reply,id={reply_id}]")
        
        
        # 将所有处理过的段落合并为一个字符串
        return ''.join(segments)

    

    @staticmethod
    def CQAT_to_qq(text: str) -> str:
        """
        将[CQ:at,qq=123456]转换为123456
        """
        pattern = r'\[CQ:at,qq=(\d+)\]'
        ma = re.search(pattern, text)
        while ma:
            text = text.replace(ma.group(), ma.group(1))
            ma = re.search(pattern, text)
        return text
        

    @staticmethod
    def get_CQreply_id(text: str) -> int | None:
        """
        将[CQ:reply,id=123456]转换为123456
        """

        # 修改正则表达式以匹配可能的负数ID
        match = re.search(r"\[CQ:reply,id=(-?\d+)]", text)
        if match:
            return int(match.group(1))  # 返回第一个匹配到的ID
        return None  # 如果没有匹配到，返回None

    @staticmethod
    def get_parameter_list(text: str, quantity: int, split: list[str] | str = ' ', annex: bool = True) -> list[str]:
        """
        将字符串转换为参数列表.
        ---
        参数:
        - text: 要分割的字符串
        - quantity: 分割数量
        - split: 分割符，默认为空格
        
        返回:
        - list[str]: 参数列表
        """
        if isinstance(split, str):
            split = [split]

        while text[0] in split:
            text = text[1:]
        
        text = re.sub(r'\s{2,}', ' ', text) # 删除连续空格

        for s in split: # 替换为统一分隔符
            text = text.replace(s, '<$split>')
        text_list = text.split('<$split>')
        if len(text_list) > quantity and annex: # 如果参数数量超过指定数量，则合并最后几个参数
            text_list[-1] = ' '.join(text_list[-(quantity - 1):])
            text_list = text_list[:quantity]
        text_list = text_list[:quantity]
        return text_list

class onebot_api():
    """

    OneBot API封装类，提供了一系列方法来管理QQ机器人的功能。

    send_like: 发送好友赞。\n
    set_group_kick: 将用户踢出群聊。\n
    set_group_ban: 禁言群成员。\n
    set_group_whole_ban: 群全员禁言。\n
    set_group_admin: 设置或取消群管理员。\n
    set_group_anonymous: 群匿名功能开关。\n
    set_group_special_title: 设置群内成员专属头衔。\n
    set_friend_add_request: 处理加好友请求。\n
    set_group_add_request: 处理加群请求或邀请请求。\n
    get_stranger_info: 获取陌生人信息。\n
    get_group_member_info: 获取群成员信息。\n
    get_group_info: 获取群信息。\n
    get_group_list: 获取群列表。\n
    get_friend_list: 获取好友列表。\n


    """
    def __init__(self, bot: Bot):
        self.bot = bot
    
    async def send_like(self, qq: int, times: int = 10) -> None:
        """
        发送好友赞。
        ---
        参数:
        - qq: 好友 QQ 号。
        - times: 赞的次数，默认为 10 次。
        """
        await self.bot.send_like(user_id=qq, times=times)
    
    async def set_group_kick(self, group_id: int, user_id: int, reject_add_request: bool = False) -> None:
        """
        将用户踢出群聊。
        ---
        参数:
        - group_id: 群号。
        - user_id: 要踢出的用户 QQ 号。
        - reject_add_request: 是否拒绝此用户的加群请求，默认为 False。
        """
        await self.bot.set_group_kick(group_id=group_id, user_id=user_id, reject_add_request=reject_add_request)
    
    async def set_group_ban(self, group_id: int, user_id: int, duration: int = 1800):
        """
        禁言群成员。
        ---
        参数:
        - group_id: 群号。
        - user_id: 要禁言的成员 QQ 号。
        - duration: 禁言时长，单位为秒，默认为 30 分钟。
        """
        await self.bot.set_group_ban(group_id=group_id, user_id=user_id, duration=duration)
    
    async def set_group_whole_ban(self, group_id: int, enable: bool = True):
        """
        群全员禁言。
        ---
        参数:
        - group_id: 群号。
        - enable: 是否开启全员禁言，默认为 True。
        """
        await self.bot.set_group_whole_ban(group_id=group_id, enable=enable)
    
    async def set_group_admin(self, group_id: int, user_id: int, enable: bool = True):
        """
        设置或取消群管理员。
        ---
        参数:
        - group_id: 群号。
        - user_id: 要设置或取消管理员的成员 QQ 号。
        - enable: 是否设置为管理员，默认为 True。
        """
        await self.bot.set_group_admin(group_id=group_id, user_id=user_id, enable=enable)

    async def set_group_card(self, group_id: int, user_id: int, card: str = ""):
        """
        设置群名片。
        ---
        参数:
        - group_id: 群号。
        - user_id: 成员 QQ 号。
        - card: 群名片内容，默认为空字符串。
        """
        await self.bot.set_group_card(group_id=group_id, user_id=user_id, card=card)

    async def set_group_name(self, group_id: int, group_name: str):
        """
        设置群名。
        ---
        参数:
        - group_id: 群号。
        - group_name: 新群名。
        """
        await self.bot.set_group_name(group_id=group_id, group_name=group_name)
    
    async def set_group_leave(self, group_id: int, is_dismiss: bool = False):
        """
        退出群聊。
        ---
        参数:
        - group_id: 群号。
        - is_dismiss: 是否解散群聊，默认为 False。
        """
        await self.bot.set_group_leave(group_id=group_id, is_dismiss=is_dismiss)

    async def set_group_special_title(self, group_id: int, user_id: int, special_title: str = " ", duration: int = -1):
        """
        设置群成员专属头衔。
        ---
        参数:
        - group_id: 群号。
        - user_id: 成员 QQ 号。
        - special_title: 专属头衔内容，默认为空字符串。
        - duration: 专属头衔有效期，单位为秒，默认为 -1（永久）。
        """
        await self.bot.set_group_special_title(group_id=group_id, user_id=user_id, special_title=special_title, duration=duration)
    
    async def set_friend_add_request(self, flag: str, approve: bool = True, remark: str = ""):
        """
        处理加好友请求。
        ---
        参数:
        - flag: 加好友请求的 flag。
        - approve: 是否同意请求，默认为 True。
        - remark: 添加后的好友备注，默认为空字符串。
        """
        await self.bot.set_friend_add_request(flag=flag, approve=approve, remark=remark)

    async def set_group_add_request(self, flag: str, sub_type: str, approve: bool = True, reason: str = ""):
        """
        处理加群请求或邀请。
        ---
        参数:
        - flag: 加群请求或邀请的 flag。
        - sub_type: 请求类型，可选值为 "add" 或 "invite"。
        - approve: 是否同意请求或邀请，默认为 True。
        - reason: 拒绝理由，默认为空字符串。
        """
        await self.bot.set_group_add_request(flag=flag, sub_type=sub_type, approve=approve, reason=reason)
    
    async def get_group_member_list(self, group_id: int) -> list:
        """
        获取群成员列表。
        ---
        参数:
        - group_id: 群号。
        返回:
        - List[GroupMemberInfo]: 群成员列表。
        """
        return await self.bot.get_group_member_list(group_id=group_id)

    async def get_group_member_info(self, group_id: int, user_id: int, no_cache: bool = False) -> dict:
        """
        获取群成员信息。
        ---
        参数:
        - group_id: 群号。
        - user_id: 群成员 QQ 号。
        - no_cache: 是否不使用缓存，默认为 False。
        返回:
        - GroupMemberInfo: 群成员信息。
        """
        return await self.bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=no_cache)

    async def get_group_list(self) -> list:
        """
        获取登录号所有的群列表。
        ---
        返回:
        - List[GroupInfo]: 群列表。
        """
        return await self.bot.get_group_list()

    async def get_login_info(self) -> dict:
        """
        获取登录号信息。
        ---
        返回:
        - LoginInfo: 登录号信息。
        """
        return await self.bot.get_login_info()

    async def get_stranger_info(self, user_id: int, no_cache: bool = False) -> dict:
        """
        获取陌生人信息。
        ---
        参数:
        - user_id: 陌生人 QQ 号。
        - no_cache: 是否不使用缓存，默认为 False。
        返回:
        - UserInfo: 陌生人信息。
        """
        return await self.bot.get_stranger_info(user_id=user_id, no_cache=no_cache)

    async def get_friend_list(self) -> list:
        """
        获取好友列表。
        """
        return await self.bot.get_friend_list()

    async def get_group_info(self, group_id: int, no_cache: bool = False) -> dict:
        """
        获取群信息。
        ---
        参数:
        - group_id: 群号。
        - no_cache: 是否不使用缓存，默认为 False。
        返回:
        - GroupInfo: 群信息。
        """
        return await self.bot.get_group_info(group_id=group_id, no_cache=no_cache)

    async def get_group_honor_info(self, group_id: int, type: str = 'all') -> dict:
        """
        获取群荣誉信息。
        ---
        参数:
        - group_id: 群号。
        - type: 要获取的群荣誉类型，可以是 'talkative', 'performer', 'legend', 'strong_newbie', 'emotion' 或 'all'。
        返回:
        - dict[str, Any]: 群荣誉信息。
        """
        return await self.bot.get_group_honor_info(group_id=group_id, type=type)

    async def get_record(self, file: str, out_format: str) -> dict:
        """
        获取语音。
        ---
        参数:
        - file: 收到的语音文件名。
        - out_format: 要转换到的格式。
        返回:
        - dict[str, Any]: 语音文件信息。
        """
        return await self.bot.get_record(file=file, out_format=out_format)

    async def get_image(self, file: str) -> dict:
        """
        获取图片。
        ---
        参数:
        - file: 收到的图片文件名。
        返回:
        - dict[str, Any]: 图片文件信息。
        """
        return await self.bot.get_image(file=file)

    async def get_msg(self, message_id: int) -> dict:
        """
        获取消息。
        ---
        参数:
        - message_id: 消息 ID。
        返回:
        - dict[str, Any]: 消息内容。
        """
        return await self.bot.get_msg(message_id=message_id)

    async def group_pock(self, group_id: int, user_id: int):
        """
        群聊戳一戳
        Args:
            bot (Bot): Bot对象
            group_id (int): 群号
            user_id (int): 用户QQ号
        """
        await self.bot.call_api("group_poke", group_id=group_id, user_id=user_id)

    async def set_msg_emoji_like(self, msg_id: int, emoji_id: int):
        """
        发送表情回应消息
        Args:
            msg_id (int): 消息ID
            emoji_id (int): 表情ID, 详见https://bot.q.qq.com/wiki/develop/api-v2/openapi/emoji/model.html#EmojiType
        """
        await self.bot.call_api("set_msg_emoji_like", message_id = msg_id, emoji_id=str(emoji_id))
    async def unset_msg_emoji_like(self, msg_id: int, emoji_id: int):
        """
        移除表情回应消息
        Args:
            msg_id (int): 消息ID
            emoji_id (int): 表情ID, 详见https://bot.q.qq.com/wiki/develop/api-v2/openapi/emoji/model.html#EmojiType
        """
        await self.bot.call_api("unset_msg_emoji_like", message_id = msg_id, emoji_id=str(emoji_id))

class rule:
 


    @staticmethod
    async def is_group_admin(ev: Event, bot: Bot, info: dict | None = None) -> bool:
        """
        检查是否为群管理员
        """
        if isinstance(ev, GroupMessageEvent):
            group_id = int(ev.group_id)
            user_id = int(ev.get_user_id())
            if info is None:
                info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=True)
            return info["role"] == "admin"
        else:
            return True
    
    @staticmethod
    async def is_group_owner(ev: Event, bot: Bot, info: dict | None = None) -> bool:
        """
        检查是否为群主
        """
        if isinstance(ev, GroupMessageEvent):
            group_id = int(ev.group_id)
            user_id = int(ev.get_user_id())
            if info is None:
                info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=True)
            return info["role"] == "owner"
        else:
            return True
        
    @staticmethod
    async def is_group_admin_or_owner(ev: Event, bot: Bot) -> bool:
        """
        检查是否为群管理员或群主
        """
        if isinstance(ev, GroupMessageEvent):
            group_id = int(ev.group_id)
            user_id = int(ev.get_user_id())
            info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=True)
            return await rule.is_group_admin(ev, bot, info) or await rule.is_group_owner(ev, bot, info)
        else:
            return True

    @staticmethod
    async def self_is_group_admin(ev: Event, bot: Bot, info: dict | None = None) -> bool:
        """
        检查机器人是否为群管理员
        """
        if isinstance(ev, GroupMessageEvent):
            group_id = int(ev.group_id)
            user_id = int(bot.self_id)
            if info is None:
                info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=True)
            return info["role"] == "admin"
        else:
            return True
    
    @staticmethod
    async def self_is_group_owner(ev: Event, bot: Bot, info: dict | None = None) -> bool:
        """
        检查机器人是否为群主
        """
        if isinstance(ev, GroupMessageEvent):
            group_id = int(ev.group_id)
            user_id = int(bot.self_id)
            if info is None:
                info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=True)
            return info["role"] == "owner"
        else:
            return True
            
    @staticmethod
    async def self_is_group_admin_or_owner(ev: Event, bot: Bot) -> bool:
        """
        检查机器人是否为群管理员或群主
        """
        if isinstance(ev, GroupMessageEvent):
            group_id = int(ev.group_id)
            user_id = int(bot.self_id)
            info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=True)
            return await rule.self_is_group_admin(ev, bot, info) or await rule.self_is_group_owner(ev, bot, info)
        else:
            return True

if __name__ == "__main__":
    print(toolbox.CQAT_to_qq("你好[CQ:at,qq=123456789] [CQ:at,qq=987654321]"))