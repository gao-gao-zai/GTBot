from __future__ import annotations

from asyncio import create_task, sleep
from typing import List, Union

from langchain.tools import ToolRuntime, tool
from nonebot import logger

from . import Fun
from .GroupChatContext import GroupChatContext
from .GroupMessageQueueManager import MessageTask, group_message_queue_manager


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

	messages: List[str] = [message] if isinstance(message, str) else message
	logger.info(f"工具调用: 向群组 {group_id} 发送 {len(messages)} 条消息（已加入队列）")

	task = MessageTask(messages=messages, group_id=group_id, interval=interval)
	await group_message_queue_manager.enqueue(
		task,
		bot=runtime.context.bot,
		message_manager=runtime.context.message_manager,
		cache=runtime.context.cache,
	)

	return f"消息已提交发送至群组 {group_id}（共 {len(messages)} 条）"


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


@tool("get_user_profile")
async def get_user_profile_tool(
	user_id: int,
	runtime: ToolRuntime[GroupChatContext],
) -> dict[int, str] | str:
	"""获取指定用户的画像描述。

	Args:
		user_id (int): 要获取画像的用户 QQ 号。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。

	Returns:
		dict[int, str] | str: 用户画像描述字典（键为描述索引，值为描述内容）。
			如果用户未设置画像，返回提示字符串。
	"""
	logger.info(f"工具调用: 获取用户 {user_id} 的画像描述")
	try:
		profile_manager = runtime.context.profile_manager
		description = await profile_manager.user.get_user_descriptions_with_index(user_id)
		if description:
			return description
		return f"用户 {user_id} 尚未设置画像描述。"
	except Exception as e:
		return f"获取用户画像失败: {str(e)}"


@tool("add_user_profile")
async def add_user_profile_tool(
	user_id: int,
	description: str,
	runtime: ToolRuntime[GroupChatContext],
) -> str:
	"""为用户添加画像描述。

	Args:
		user_id (int): 用户 QQ 号。
		description (str): 要添加的画像描述内容。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。

	Returns:
		str: 操作结果信息。
	"""
	logger.info(f"工具调用: 为用户 {user_id} 添加画像描述")
	try:
		profile_manager = runtime.context.profile_manager
		await profile_manager.user.add_user_profile(user_id, description)
		return f"已为用户 {user_id} 添加画像描述: {description}"
	except ValueError as e:
		return f"添加用户画像失败: {str(e)}"
	except Exception as e:
		return f"添加用户画像失败: {str(e)}"


@tool("edit_user_profile")
async def edit_user_profile_tool(
	user_id: int,
	index: int,
	new_description: str,
	runtime: ToolRuntime[GroupChatContext],
) -> str:
	"""编辑用户指定序号的画像描述。

	Args:
		user_id (int): 用户 QQ 号。
		index (int): 要编辑的描述序号（从 1 开始）。
		new_description (str): 新的描述内容。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。

	Returns:
		str: 操作结果信息。
	"""
	logger.info(f"工具调用: 编辑用户 {user_id} 第 {index} 条画像描述")
	try:
		profile_manager = runtime.context.profile_manager
		await profile_manager.user.edit_user_description_by_index(
			user_id, index, new_description
		)
		return f"已编辑用户 {user_id} 第 {index} 条画像描述为: {new_description}"
	except ValueError as e:
		return f"编辑用户画像失败: {str(e)}"
	except Exception as e:
		return f"编辑用户画像失败: {str(e)}"


@tool("delete_user_profile")
async def delete_user_profile_tool(
	user_id: int,
	indices: Union[int, list[int]],
	runtime: ToolRuntime[GroupChatContext],
) -> str:
	"""删除用户指定序号的画像描述。

	Args:
		user_id (int): 用户 QQ 号。
		indices (int | list[int]): 要删除的描述序号（从 1 开始），可以是单个整数或整数列表。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。

	Returns:
		str: 操作结果信息。

	Note:
		删除后其他描述的序号会自动递减。例如删除序号 2 后，
		原来的序号 3 会变成 2，序号 4 会变成 3，依此类推。
		若需要继续操作，请重新调用获取用户画像工具以确认新序号。
	"""
	logger.info(f"工具调用: 删除用户 {user_id} 的画像描述（序号: {indices}）")
	try:
		profile_manager = runtime.context.profile_manager
		await profile_manager.user.delete_user_description_by_index(user_id, indices)
		return f"已删除用户 {user_id} 的指定画像描述。"
	except ValueError as e:
		return f"删除用户画像失败: {str(e)}"
	except Exception as e:
		return f"删除用户画像失败: {str(e)}"


@tool("get_group_profile")
async def get_group_profile_tool(
	group_id: int,
	runtime: ToolRuntime[GroupChatContext],
) -> dict[int, str] | str:
	"""获取指定群聊的画像描述。

	Args:
		group_id (int): 群号，不填则使用当前群组。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。

	Returns:
		dict[int, str] | str: 群聊画像描述字典（键为描述索引，值为描述内容）。
			如果群聊未设置画像，返回提示字符串。
	"""
	if group_id is None:
		group_id = runtime.context.group_id

	logger.info(f"工具调用: 获取群聊 {group_id} 的画像描述")
	try:
		profile_manager = runtime.context.profile_manager
		description = await profile_manager.group.get_group_descriptions_with_index(group_id)
		if description:
			return description
		return f"群聊 {group_id} 尚未设置画像描述。"
	except Exception as e:
		return f"获取群聊画像失败: {str(e)}"


@tool("add_group_profile")
async def add_group_profile_tool(
	description: str,
	runtime: ToolRuntime[GroupChatContext],
	group_id: int | None = None,
) -> str:
	"""为群聊添加画像描述。

	Args:
		description (str): 要添加的画像描述内容。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
		group_id (int | None): 群号，不填则使用当前群组。

	Returns:
		str: 操作结果信息。
	"""
	if group_id is None:
		group_id = runtime.context.group_id

	logger.info(f"工具调用: 为群聊 {group_id} 添加画像描述")
	try:
		profile_manager = runtime.context.profile_manager
		await profile_manager.group.add_group_profile(group_id, description)
		return f"已为群聊 {group_id} 添加画像描述: {description}"
	except ValueError as e:
		return f"添加群聊画像失败: {str(e)}"
	except Exception as e:
		return f"添加群聊画像失败: {str(e)}"


@tool("edit_group_profile")
async def edit_group_profile_tool(
	index: int,
	new_description: str,
	runtime: ToolRuntime[GroupChatContext],
	group_id: int | None = None,
) -> str:
	"""编辑群聊指定序号的画像描述。

	Args:
		index (int): 要编辑的描述序号（从 1 开始）。
		new_description (str): 新的描述内容。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
		group_id (int | None): 群号，不填则使用当前群组。

	Returns:
		str: 操作结果信息。
	"""
	if group_id is None:
		group_id = runtime.context.group_id

	logger.info(f"工具调用: 编辑群聊 {group_id} 第 {index} 条画像描述")
	try:
		profile_manager = runtime.context.profile_manager
		await profile_manager.group.edit_group_description_by_index(
			group_id, index, new_description
		)
		return f"已编辑群聊 {group_id} 第 {index} 条画像描述为: {new_description}"
	except ValueError as e:
		return f"编辑群聊画像失败: {str(e)}"
	except Exception as e:
		return f"编辑群聊画像失败: {str(e)}"


@tool("delete_group_profile")
async def delete_group_profile_tool(
	indices: Union[int, list[int]],
	runtime: ToolRuntime[GroupChatContext],
	group_id: int | None = None,
) -> str:
	"""删除群聊指定序号的画像描述。

	Args:
		indices (int | list[int]): 要删除的描述序号（从 1 开始），可以是单个整数或整数列表。
		runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
		group_id (int | None): 群号，不填则使用当前群组。

	Returns:
		str: 操作结果信息。

	Note:
		删除后其他描述的序号会自动递减。例如删除序号 2 后，
		原来的序号 3 会变成 2，序号 4 会变成 3，依此类推。
		若需要继续操作，请重新调用获取群聊画像工具以确认新序号。
	"""
	if group_id is None:
		group_id = runtime.context.group_id

	logger.info(f"工具调用: 删除群聊 {group_id} 的画像描述（序号: {indices}）")
	try:
		profile_manager = runtime.context.profile_manager
		await profile_manager.group.delete_group_description_by_index(group_id, indices)
		return f"已删除群聊 {group_id} 的指定画像描述。"
	except ValueError as e:
		return f"删除群聊画像失败: {str(e)}"
	except Exception as e:
		return f"删除群聊画像失败: {str(e)}"
