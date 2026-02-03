#!/usr/bin/env python3
"""
GTBot 数据目录配置测试脚本

这个脚本用于测试新添加的数据目录配置是否正常工作
"""

import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ConfigManager import TotalConfiguration
    
    print("=== GTBot 数据目录配置测试 ===\n")
    
    # 1. 初始化配置
    print("1. 初始化配置...")
    config = TotalConfiguration.init()
    print("✓ 配置初始化成功")
    
    # 2. 获取数据目录路径
    print("\n2. 获取数据目录路径...")
    data_dir = config.get_data_dir_path()
    print(f"✓ 数据目录路径: {data_dir}")
    print(f"  - 绝对路径: {data_dir.resolve()}")
    print(f"  - 目录存在: {'✓' if data_dir.exists() else '✗'}")
    print(f"  - 是否为目录: {'✓' if data_dir.is_dir() else '✗'}")
    
    # 3. 测试数据库路径配置（如果model模块可用）
    print("\n3. 测试数据库路径配置...")
    try:
        from plugins.GTBot.DBmodel import DB_PATH, ASYNC_DB_URL, DATA_DIR
        print(f"✓ 模型中的数据目录: {DATA_DIR}")
        print(f"✓ 数据库文件路径: {DB_PATH}")
        print(f"✓ 异步数据库URL: {ASYNC_DB_URL}")
        
        # 4. 验证路径一致性
        print("\n4. 验证路径一致性...")
        if DATA_DIR == data_dir:
            print("✓ 配置和模型中的数据目录路径一致")
        else:
            print(f"✗ 路径不一致 - 配置: {data_dir}, 模型: {DATA_DIR}")
    except ImportError as e:
        print(f"⚠ 跳过model模块测试（缺少依赖）: {e}")
        # 直接验证路径配置
        print("\n4. 手动验证数据库路径配置...")
        from plugins.GTBot.constants import DEFAULT_DB_FILENAME

        expected_db_path = data_dir / DEFAULT_DB_FILENAME
        print(f"✓ 预期数据库路径: {expected_db_path}")
    
    # 5. 显示其他配置信息
    print("\n5. 其他配置信息...")
    print(f"  - 当前配置组: {config.get_current_group_name()}")
    print(f"  - 可用配置组: {', '.join(config.get_available_config_groups())}")
    print(f"  - 配置文件路径: {config.get_config_file_path()}")
    
    print("\n=== 测试完成 ===")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()