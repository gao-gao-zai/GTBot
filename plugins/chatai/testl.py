import asyncio
from typing import Dict
from random import randint

# 模拟异步获取用户名的函数
async def get_username(user_id: int) -> str:
    """模拟根据ID获取用户名，实际中可能是数据库查询或API调用"""
    await asyncio.sleep(1+ randint(0, 10)*0.1)  # 模拟耗时操作
    return f"user_{user_id}"  # 返回模拟的用户名

async def fetch_all_users(user_ids: list[int]) -> Dict[int, str]:
    """并发获取所有用户的用户名并返回映射字典"""
    # 创建所有任务
    tasks = [asyncio.create_task(get_username(uid)) for uid in user_ids]
    
    # 等待所有任务完成
    usernames = await asyncio.gather(*tasks)
    
    # 创建ID到用户名的映射字典
    return dict(zip(user_ids, usernames))

async def main():
    user_ids = [101, 202, 303, 404, 505]  # 示例ID列表
    
    print("开始获取用户名...")
    start_time = asyncio.get_event_loop().time()
    
    user_map = await fetch_all_users(user_ids)
    
    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"获取完成，耗时: {elapsed:.2f}秒")
    print("结果:", user_map)

# 运行主程序
asyncio.run(main())