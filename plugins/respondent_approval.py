# group_join_verifier.py

import httpx
import time
from typing import Dict, Any, Optional

from nonebot import on_request
from nonebot.log import logger
from nonebot.adapters.onebot.v11 import Bot, GroupRequestEvent

# --- 配置区域 ---
# 请在此处修改为你的实际配置

TARGET_GROUP_ID: int = 723708097
API_BASE_URL: str = "https://wq.gaozaiya.cloudns.org"
API_KEY: str = "your_api_key_here"

# --- 配置区域结束 ---


group_request_handler = on_request(priority=5, block=True)

@group_request_handler.handle()
async def handle_group_request(bot: Bot, event: GroupRequestEvent) -> None:
    """处理加群请求，通过API验证识别码"""
    if event.group_id != TARGET_GROUP_ID:
        return

    # --- START: 修正类型错误 ---
    # 1. 将类型声明为 Optional[str]，因为它可能为 None
    comment: Optional[str] = event.comment
    
    # 2. 增加一个前置检查，处理 comment 为 None 或空字符串 "" 的情况
    #    这个 `if not comment:` 同时覆盖了 None 和 "" 两种情况
    if not comment:
        await event.reject(bot, reason="入群答案（识别码）不能为空。")
        logger.warning(f"用户 {event.user_id} 申请加入群 {event.group_id}，但答案为空。")
        return
    # --- END: 修正类型错误 ---
    # 经过上面的检查，类型检查器知道在此之后的 `comment` 变量一定是 `str` 类型

    # 从 comment 中解析出真正的答案
    identifier: str = ""
    if "\n答案：" in comment:
        try:
            identifier = comment.split("\n答案：", 1)[1].strip()
        except IndexError:
            logger.warning(f"无法从入群信息 '{comment}' 中解析答案，将尝试使用完整信息。")
            identifier = comment.strip()
    else:
        identifier = comment.strip()

    if not identifier:
        await event.reject(bot, reason="入群答案（识别码）不能为空。")
        logger.warning(f"用户 {event.user_id} 申请加入群 {event.group_id}，解析出的识别码为空。")
        return

    logger.info(f"收到用户 {event.user_id} 加入群 {event.group_id} 的请求，解析出的识别码为: '{identifier}'")

    headers: Dict[str, str] = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    async with httpx.AsyncClient() as client:
        try:
            get_url: str = f"{API_BASE_URL}/api/respondent/{identifier}"
            response: httpx.Response = await client.get(get_url, headers=headers)

            if response.status_code == 200:
                user_data: Dict[str, Any] = response.json()
                request_qq: str = str(event.user_id)

                # 1. 检查识别码是否过期
                expire_ts_str: Optional[str] = user_data.get("expire_time_timestamp")
                if expire_ts_str:
                    try:
                        expire_timestamp: float = float(expire_ts_str)
                        current_timestamp: float = time.time()
                        
                        if current_timestamp > expire_timestamp:
                            reason = f"验证失败：识别码 '{identifier}' 已于 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expire_timestamp))} 过期。"
                            logger.warning(reason)
                            await event.reject(bot, reason="验证失败：您的识别码已过期。")
                            await update_user_info(
                                client,
                                identifier,
                                reviewed=True,
                                approved=False,
                                notes=reason
                            )
                            return
                    except (ValueError, TypeError):
                        logger.error(f"无法解析识别码 '{identifier}' 的过期时间戳: '{expire_ts_str}'，请检查API返回数据格式。")
                        await event.reject(bot, reason="系统验证出错（时间戳格式错误），请联系管理员。")
                        return

                # 2. 检查QQ号是否匹配或首次绑定
                api_qq: Optional[str] = user_data.get("qq_number")

                if api_qq and api_qq != request_qq:
                    reason = f"验证失败：该识别码已被QQ号({api_qq})绑定，与您的申请QQ号({request_qq})不符。"
                    logger.warning(reason)
                    await event.reject(bot, reason="验证失败：该识别码已被其他QQ号使用。")
                    await update_user_info(
                        client, 
                        identifier, 
                        reviewed=True, 
                        approved=False, 
                        notes=reason
                    )
                else:
                    logger.info(f"验证成功: 识别码 {identifier} 与申请QQ {request_qq} 匹配或首次绑定。同意入群。")
                    await event.approve(bot)
                    await update_user_info(
                        client, 
                        identifier, 
                        reviewed=True, 
                        approved=True, 
                        notes=f"QQ号 {request_qq} 验证通过，已自动同意入群。",
                        qq_number=request_qq
                    )

            elif response.status_code == 404:
                reason = "识别码错误，请确保识别码正确有效，无空格等特殊字符"
                logger.warning(f"用户 {event.user_id} 使用了无效的识别码: {identifier}")
                await event.reject(bot, reason=reason)
            
            elif response.status_code == 401:
                logger.error("API 密钥无效或未提供，请检查配置。")
                await event.reject(bot, reason="系统验证出错，请联系管理员。")

            else:
                logger.error(f"API 请求失败，状态码: {response.status_code}, 响应: {response.text}")
                await event.reject(bot, reason="系统验证出错，请联系管理员。")

        except httpx.RequestError as e:
            logger.error(f"网络请求错误: {e}")
            await event.reject(bot, reason="无法连接到验证服务器，请联系管理员。")


async def update_user_info(
    client: httpx.AsyncClient, 
    identifier: str, 
    reviewed: bool, 
    approved: bool, 
    notes: str,
    qq_number: Optional[str] = None
) -> None:
    """
    辅助函数，用于调用 PUT API 更新用户信息（包括状态和QQ号）。
    """
    put_url: str = f"{API_BASE_URL}/api/respondent/{identifier}"
    payload: Dict[str, Any] = {
        "reviewed": reviewed,
        "approved": approved,
        "notes": notes
    }
    if qq_number is not None:
        payload["qq_number"] = qq_number

    headers: Dict[str, str] = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    try:
        response: httpx.Response = await client.put(put_url, json=payload, headers=headers)
        if response.status_code == 200:
            logger.info(f"成功更新识别码 {identifier} 的信息: {payload}")
        else:
            logger.error(f"更新识别码 {identifier} 信息失败，状态码: {response.status_code}, 响应: {response.text}")
    except httpx.RequestError as e:
        logger.error(f"更新状态时网络请求错误: {e}")

