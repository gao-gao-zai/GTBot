from __future__ import annotations

from asyncio import create_task, sleep
from math import log
from typing import List, Union

from langchain.tools import ToolRuntime, tool
from nonebot import logger
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, PrivateMessageEvent

from ..shared import fun as Fun
from .context import GroupChatContext
from .group_queue import MessageTask, group_message_queue_manager
from .private_queue import PrivateMessageTask, private_message_queue_manager
from .queue_payload import prepare_queue_messages


@tool("send_group_message")
async def send_group_message_tool(
	message: str | List[str],
	runtime: ToolRuntime[GroupChatContext],
	group_id: int | None = None,
	interval: float = 0.2,
) -> str:
	"""向指定群组发送消息。

	使用生产者-消费者模型，确保发送给同一个群的消息按顺序发送，不会并行发送。
	不同群组之间的消息可以并行发送。

	Args:
		message (str | List[str]): 要发送的消息内容，可以是单条消息或消息列表。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
		group_id (int | None): 目标群组 ID。不填则自动获取当前的聊群 ID。
		interval (float): 发送多条消息时的间隔时间（秒），默认为 0.2。

	Returns:
		str: 发送结果信息。

	Note:
		- 使用消息队列确保同一群组的消息按顺序发送。
		- 消息会被加入队列后异步发送，不会阻塞调用。
	"""
	if group_id is None:
		group_id = runtime.context.group_id
	if group_id is None:
		return "当前会话不是群聊，send_group_message 不可用"

	messages: List[str] = [message] if isinstance(message, str) else message
	logger.info(f"工具调用: 向群组 {group_id} 发送 {len(messages)} 条消息（已加入队列）")
	prepared_messages = await prepare_queue_messages(
		messages,
		scope=f"群组 {group_id}",
	)

	task = MessageTask(messages=prepared_messages, group_id=group_id, interval=interval)
	await group_message_queue_manager.enqueue(
		task,
		bot=runtime.context.bot,
		message_manager=runtime.context.message_manager,
		cache=runtime.context.cache,
	)

	return f"消息已提交发送至群组 {group_id}（共 {len(messages)} 条）"


@tool("send_private_message")
async def send_private_message_tool(
	message: str | List[str],
	runtime: ToolRuntime[GroupChatContext],
	user_id: int | None = None,
	interval: float = 0.2,
) -> str:
	"""向当前私聊会话发送消息。

	Args:
		message (str | List[str]): 要发送的消息内容，可以是一条消息或消息列表。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。
		user_id (int | None): 可选目标用户 ID；若填写，则只能等于当前私聊对象。
		interval (float): 多条消息之间的发送间隔秒数。

	Returns:
		str: 发送结果摘要。
	"""
	chat_type = getattr(runtime.context, "chat_type", None)
	if chat_type != "private":
		return "当前会话不是私聊，send_private_message 不可用"

	session_id = str(getattr(runtime.context, "session_id", "") or "").strip()
	peer_user_id = 0
	if session_id.startswith("private:"):
		try:
			peer_user_id = int(session_id.split(":", 1)[1])
		except ValueError:
			peer_user_id = 0
	if peer_user_id <= 0:
		peer_user_id = int(getattr(runtime.context, "user_id", 0) or 0)
	if peer_user_id <= 0:
		return "当前私聊会话缺少有效 user_id，无法发送消息"

	target_user_id = int(user_id) if user_id is not None else peer_user_id
	if target_user_id != peer_user_id:
		return "send_private_message 只能发送到当前私聊对象"

	messages: List[str] = [message] if isinstance(message, str) else message
	logger.info(f"工具调用: 向私聊用户 {target_user_id} 发送 {len(messages)} 条消息（已加入队列）")
	prepared_messages = await prepare_queue_messages(
		messages,
		scope=f"session private:{target_user_id}",
	)

	task = PrivateMessageTask(
		messages=prepared_messages,
		user_id=target_user_id,
		interval=interval,
		session_id=f"private:{target_user_id}",
	)
	await private_message_queue_manager.enqueue(
		task,
		bot=runtime.context.bot,
		message_manager=runtime.context.message_manager,
		cache=runtime.context.cache,
	)

	return f"消息已提交发送至私聊用户 {target_user_id}（共 {len(messages)} 条）"


@tool("delete_message")
async def delete_message_tool(
	message_id: int,
	runtime: ToolRuntime[GroupChatContext],
	delay: int = 0,
) -> str:
	"""撤回指定的消息。

	Args:
		message_id (int): 要撤回的消息 ID。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
		delay (int): 延迟撤回时间（秒），范围 0-60，默认为 0（立即撤回）。

	Returns:
		str: 撤回结果信息。

	Note:
		- 如果自己是管理员或群主则撤回无限制。
		- 否则只能撤回自己的消息，且时间限制为 2 分钟内。
		- 延迟撤回时间最大为 60 秒。
		- 延迟撤回会在后台进行，不会堵塞主流程。
	"""
	if not isinstance(delay, int) or delay < 0 or delay > 60:
		return f"撤回消息 {message_id} 失败: 延迟撤回时间必须在 0-60 秒之间"

	logger.info(f"工具调用: 撤回消息 {message_id}（延迟 {delay} 秒）")

	async def delete_message_async() -> None:
		"""异步撤回消息，支持延迟撤回。"""
		try:
			if delay > 0:
				await sleep(delay)

			await Fun.delete_message(runtime.context.bot, message_id, delay=0)
			msg_mg = runtime.context.message_manager
			await msg_mg.mark_message_withdrawn(message_id)
			logger.info(f"消息 {message_id} 已成功撤回")
		except Exception as e:
			logger.error(f"撤回消息 {message_id} 失败: {str(e)}")

	if delay == 0:
		try:
			await delete_message_async()
			return f"消息 {message_id} 已成功撤回"
		except Exception as e:
			return f"撤回消息 {message_id} 失败: {str(e)}"

	create_task(delete_message_async())
	return f"消息 {message_id} 将在 {delay} 秒后撤回"


@tool("emoji_reaction")
async def emoji_reaction_tool(
	message_id: int,
	emoji_id: int,
	runtime: ToolRuntime[GroupChatContext],
) -> str:
	"""对消息进行表情回应（表情贴）。

	Args:
		message_id (int): 要回应的消息 ID。
		emoji_id (int): 表情 ID，QQ 表情对应的数字编号。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。

	Returns:
		str: 回应结果信息。

	Note:
		只支持群聊消息。
	"""
	logger.info(f"工具调用: 对消息 {message_id} 添加表情回应 {emoji_id}")
	if getattr(runtime.context, "group_id", None) is None:
		return "当前会话不是群聊，emoji_reaction 不可用"
	try:
		await Fun.set_msg_emoji_like(runtime.context.bot, message_id, emoji_id)
		return f"已对消息 {message_id} 添加表情回应（表情ID: {emoji_id}）"
	except Exception as e:
		return f"表情回应失败: {str(e)}"


@tool("poke_user")
async def poke_user_tool(
	user_id: int,
	runtime: ToolRuntime[GroupChatContext],
	group_id: int | None = None,
) -> str:
	"""在群聊中戳一戳指定用户（双击头像效果）。

	Args:
		user_id (int): 要戳的用户 QQ 号。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
		group_id (int | None): 群号。不填则使用当前群组。

	Returns:
		str: 戳一戳结果信息。
	"""
	if group_id is None:
		group_id = runtime.context.group_id
	if group_id is None:
		return "当前会话不是群聊，poke_user 不可用"

	logger.info(f"工具调用: 在群组 {group_id} 戳一戳用户 {user_id}")
	try:
		await Fun.group_poke(runtime.context.bot, group_id, user_id)
		return f"已在群组 {group_id} 戳了用户 {user_id}"
	except Exception as e:
		return f"戳一戳失败: {str(e)}"


@tool("send_like")
async def send_like_tool(
	user_id: int,
	runtime: ToolRuntime[GroupChatContext],
) -> str:
	"""给指定用户发送点赞。

	Args:
		user_id (int): 要点赞的用户 QQ 号。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。

	Returns:
		str: 点赞结果信息。

	Note:
		一般每天只允许给同一个好友点赞 10 次，超出的会失败。
	"""
	logger.info(f"工具调用: 给用户 {user_id} 发送点赞")
	try:
		await Fun.send_like(runtime.context.bot, user_id)
		return f"已给用户 {user_id} 发送点赞"
	except Exception as e:
		return f"发送点赞失败: {str(e)}"


