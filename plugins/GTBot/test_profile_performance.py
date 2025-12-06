"""用户画像功能性能测试脚本。

测试内容:
    - 添加用户画像性能
    - 获取用户画像性能
    - 编辑用户画像性能
    - 删除用户画像性能
    - 批量操作性能
    - 并发操作性能
"""

import asyncio
import time
import json
import random
import string
import statistics
from pathlib import Path
from typing import Callable, Awaitable, Any
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, INTEGER, select, delete


# ============================================================================
# 数据库模型定义（独立定义，避免依赖复杂的项目配置）
# ============================================================================
class Base(DeclarativeBase):
    pass


class UserProfileModel(Base):
    """用户画像模型 (ORM)"""
    __tablename__ = "test_user_profiles"
    user_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    data: Mapped[str] = mapped_column(Text, default="[]")


class GroupProfileModel(Base):
    """群画像模型 (ORM)"""
    __tablename__ = "test_group_profiles"
    group_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    data: Mapped[str] = mapped_column(Text, default="[]")


# ============================================================================
# 测试用 UserProfileManager（简化版）
# ============================================================================
@dataclass
class UserProfileConfig:
    """用户画像配置"""
    max_descriptions: int = 50
    max_description_char_length: int = 500


config = UserProfileConfig()


class UserProfileManager:
    """用户资料管理器，负责用户资料的增删改查操作。"""
    
    def __init__(self, engine: AsyncEngine) -> None:
        """初始化用户资料管理器。
        
        Args:
            engine: 异步数据库引擎。
        """
        self._engine: AsyncEngine = engine
        self._session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """创建数据库表。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """删除数据库表。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    def _validate_descriptions(self, descriptions: list[str]) -> None:
        """验证描述列表的有效性。
        
        Args:
            descriptions: 描述列表。
            
        Raises:
            ValueError: 当描述超过长度或数量限制时抛出。
        """
        for desc_item in descriptions:
            if len(desc_item) > config.max_description_char_length:
                raise ValueError(
                    f"单条描述长度超过限制（{len(desc_item)} > {config.max_description_char_length}）"
                )
        if len(descriptions) > config.max_descriptions:
            raise ValueError(
                f"用户画像描述条数超过限制（{len(descriptions)} > {config.max_descriptions}）"
            )

    def _parse_description_data(self, data_str: str, user_id: int) -> list[str]:
        """解析描述数据字符串。
        
        Args:
            data_str: JSON格式的描述数据字符串。
            user_id: 用户ID，用于日志记录。
            
        Returns:
            解析后的描述列表。
        """
        try:
            data = json.loads(data_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    async def get_user_profile(self, user_id: int) -> dict | None:
        """获取用户资料。
        
        Args:
            user_id: 用户ID。
            
        Returns:
            用户资料字典，如果不存在则返回None。
        """
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            if user_model:
                return {
                    "user_id": user_model.user_id,
                    "description": self._parse_description_data(user_model.data, user_id)
                }
            return None

    async def get_user_descriptions_with_index(self, user_id: int) -> dict[int, str] | None:
        """获取带有序号的用户描述。
        
        Args:
            user_id: 用户ID。
            
        Returns:
            带序号的描述字典，键为序号（从1开始），值为描述内容。
            如果用户不存在则返回None。
        """
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            if user_model:
                descriptions = self._parse_description_data(user_model.data, user_id)
                return {i + 1: desc_item for i, desc_item in enumerate(descriptions)}
            return None

    async def add_user_profile(self, user_id: int, description: list[str] | str) -> None:
        """添加用户资料。
        
        Args:
            user_id: 用户ID。
            description: 描述内容，可以是字符串或字符串列表。
            
        Raises:
            ValueError: 当描述超过长度或数量限制时抛出。
        """
        descriptions = [description] if isinstance(description, str) else description
        self._validate_descriptions(descriptions)
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if user_model:
                existing_data = self._parse_description_data(user_model.data, user_id)
                existing_data.extend(descriptions)
                self._validate_descriptions(existing_data)
                user_model.data = json.dumps(existing_data)
            else:
                session.add(UserProfileModel(user_id=user_id, data=json.dumps(descriptions)))
            
            await session.commit()

    async def edit_user_description_by_index(self, user_id: int, index: int, new_description: str) -> None:
        """编辑用户对应序号的描述。
        
        Args:
            user_id: 用户ID。
            index: 描述序号（从1开始）。
            new_description: 新的描述内容。
            
        Raises:
            ValueError: 当用户不存在、序号无效或描述超过长度限制时抛出。
        """
        if len(new_description) > config.max_description_char_length:
            raise ValueError(
                f"描述长度超过限制（{len(new_description)} > {config.max_description_char_length}）"
            )
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if not user_model:
                raise ValueError(f"用户 {user_id} 的资料不存在")
            
            descriptions = self._parse_description_data(user_model.data, user_id)
            
            if index < 1 or index > len(descriptions):
                raise ValueError(f"序号 {index} 无效，有效范围为 1-{len(descriptions)}")
            
            descriptions[index - 1] = new_description
            user_model.data = json.dumps(descriptions)
            await session.commit()

    async def delete_user_description_by_index(self, user_id: int, indices: int | list[int]) -> None:
        """按序号删除用户的指定描述。
        
        Args:
            user_id: 用户ID。
            indices: 要删除的描述序号（从1开始），可以是单个整数或整数列表。
            
        Raises:
            ValueError: 当用户不存在、序号无效或删除后无描述时抛出。
        """
        delete_indices = [indices] if isinstance(indices, int) else indices
        
        if not all(isinstance(idx, int) and idx > 0 for idx in delete_indices):
            raise ValueError("所有序号必须为正整数")
        
        delete_indices = sorted(set(delete_indices), reverse=True)
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if not user_model:
                raise ValueError(f"用户 {user_id} 的资料不存在")
            
            descriptions = self._parse_description_data(user_model.data, user_id)
            
            for idx in delete_indices:
                if idx < 1 or idx > len(descriptions):
                    raise ValueError(f"序号 {idx} 无效，有效范围为 1-{len(descriptions)}")
            
            for idx in delete_indices:
                descriptions.pop(idx - 1)
            
            user_model.data = json.dumps(descriptions)
            await session.commit()

    async def delete_user_profile(self, user_id: int) -> None:
        """删除用户的所有描述。
        
        Args:
            user_id: 用户ID。
        """
        async with self._session_maker() as session:
            await session.execute(
                delete(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            await session.commit()


# ============================================================================
# 性能测试工具类
# ============================================================================
@dataclass
class PerformanceResult:
    """性能测试结果。"""
    name: str
    """测试名称"""
    total_time: float
    """总耗时（秒）"""
    iterations: int
    """迭代次数"""
    times: list[float] = field(default_factory=list)
    """每次迭代的耗时列表"""
    
    @property
    def avg_time(self) -> float:
        """平均耗时（秒）"""
        return self.total_time / self.iterations if self.iterations > 0 else 0
    
    @property
    def avg_time_ms(self) -> float:
        """平均耗时（毫秒）"""
        return self.avg_time * 1000
    
    @property
    def min_time_ms(self) -> float:
        """最小耗时（毫秒）"""
        return min(self.times) * 1000 if self.times else 0
    
    @property
    def max_time_ms(self) -> float:
        """最大耗时（毫秒）"""
        return max(self.times) * 1000 if self.times else 0
    
    @property
    def std_dev_ms(self) -> float:
        """标准差（毫秒）"""
        return statistics.stdev(self.times) * 1000 if len(self.times) > 1 else 0
    
    @property
    def ops_per_sec(self) -> float:
        """每秒操作数"""
        return self.iterations / self.total_time if self.total_time > 0 else 0


class PerformanceTester:
    """性能测试器。"""
    
    def __init__(self, manager: UserProfileManager):
        """初始化性能测试器。
        
        Args:
            manager: 用户画像管理器实例。
        """
        self.manager = manager
        self.results: list[PerformanceResult] = []
    
    async def run_test(
        self,
        name: str,
        test_func: Callable[[], Awaitable[Any]],
        iterations: int = 100,
        warmup: int = 5
    ) -> PerformanceResult:
        """运行单个性能测试。
        
        Args:
            name: 测试名称。
            test_func: 测试函数。
            iterations: 迭代次数。
            warmup: 预热次数。
            
        Returns:
            性能测试结果。
        """
        # 预热
        for _ in range(warmup):
            await test_func()
        
        # 正式测试
        times: list[float] = []
        start_total = time.perf_counter()
        
        for _ in range(iterations):
            start = time.perf_counter()
            await test_func()
            end = time.perf_counter()
            times.append(end - start)
        
        end_total = time.perf_counter()
        total_time = end_total - start_total
        
        result = PerformanceResult(
            name=name,
            total_time=total_time,
            iterations=iterations,
            times=times
        )
        self.results.append(result)
        return result
    
    def print_results(self) -> None:
        """打印所有测试结果。"""
        print("\n" + "=" * 80)
        print("性能测试结果".center(80))
        print("=" * 80)
        
        print(f"\n{'测试名称':<30} {'平均(ms)':<12} {'最小(ms)':<12} {'最大(ms)':<12} {'标准差(ms)':<12} {'ops/s':<10}")
        print("-" * 88)
        
        for result in self.results:
            print(
                f"{result.name:<30} "
                f"{result.avg_time_ms:<12.3f} "
                f"{result.min_time_ms:<12.3f} "
                f"{result.max_time_ms:<12.3f} "
                f"{result.std_dev_ms:<12.3f} "
                f"{result.ops_per_sec:<10.1f}"
            )
        
        print("-" * 88)
        print()


def generate_random_description(length: int = 50) -> str:
    """生成随机描述文本。
    
    Args:
        length: 描述长度。
        
    Returns:
        随机描述文本。
    """
    return ''.join(random.choices(string.ascii_letters + string.digits + " ", k=length))


# ============================================================================
# 性能测试用例
# ============================================================================
async def run_performance_tests():
    """运行所有性能测试。"""
    # 创建测试数据库
    test_db_path = Path(__file__).parent / "data" / "test_profile_performance.db"
    test_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 删除旧的测试数据库
    if test_db_path.exists():
        test_db_path.unlink()
    
    # 同时删除 WAL 相关文件
    wal_path = test_db_path.with_suffix(".db-wal")
    shm_path = test_db_path.with_suffix(".db-shm")
    if wal_path.exists():
        wal_path.unlink()
    if shm_path.exists():
        shm_path.unlink()
    
    db_url = f"sqlite+aiosqlite:///{test_db_path}"
    engine = create_async_engine(db_url, echo=False)
    
    # 启用 WAL 模式
    from sqlalchemy import event
    
    def set_sqlite_pragma(dbapi_conn, connection_record):
        """设置 SQLite WAL 模式。"""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()
    
    event.listen(engine.sync_engine, "connect", set_sqlite_pragma)
    
    manager = UserProfileManager(engine)
    await manager.create_tables()
    
    tester = PerformanceTester(manager)
    
    print("\n开始用户画像功能性能测试...")
    print(f"测试数据库: {test_db_path}")
    
    # ========================================================================
    # 测试 1: 添加用户画像（新用户）
    # ========================================================================
    user_id_counter = [1000000]
    
    async def test_add_new_user():
        user_id_counter[0] += 1
        await manager.add_user_profile(user_id_counter[0], generate_random_description(100))
    
    print("\n[1/8] 测试添加用户画像（新用户）...")
    await tester.run_test("添加画像(新用户)", test_add_new_user, iterations=100)
    
    # ========================================================================
    # 测试 2: 添加用户画像（已有用户追加）
    # ========================================================================
    test_user_id = 999999
    await manager.add_user_profile(test_user_id, generate_random_description(100))
    
    async def test_add_existing_user():
        await manager.add_user_profile(test_user_id, generate_random_description(100))
    
    print("[2/8] 测试添加用户画像（已有用户追加）...")
    # 限制迭代次数，避免超过描述数量限制
    await tester.run_test("添加画像(追加)", test_add_existing_user, iterations=40, warmup=2)
    
    # ========================================================================
    # 测试 3: 获取用户画像
    # ========================================================================
    async def test_get_user_profile():
        await manager.get_user_profile(test_user_id)
    
    print("[3/8] 测试获取用户画像...")
    await tester.run_test("获取画像(get_user_profile)", test_get_user_profile, iterations=200)
    
    # ========================================================================
    # 测试 4: 获取带索引的用户描述
    # ========================================================================
    async def test_get_descriptions_with_index():
        await manager.get_user_descriptions_with_index(test_user_id)
    
    print("[4/8] 测试获取带索引的用户描述...")
    await tester.run_test("获取画像(带索引)", test_get_descriptions_with_index, iterations=200)
    
    # ========================================================================
    # 测试 5: 编辑用户画像
    # ========================================================================
    async def test_edit_description():
        await manager.edit_user_description_by_index(
            test_user_id, 1, generate_random_description(100)
        )
    
    print("[5/8] 测试编辑用户画像...")
    await tester.run_test("编辑画像", test_edit_description, iterations=100)
    
    # ========================================================================
    # 测试 6: 获取不存在的用户画像
    # ========================================================================
    async def test_get_nonexistent_user():
        await manager.get_user_profile(1)  # 不存在的用户
    
    print("[6/8] 测试获取不存在的用户画像...")
    await tester.run_test("获取画像(不存在)", test_get_nonexistent_user, iterations=200)
    
    # ========================================================================
    # 测试 7: 批量添加用户画像
    # ========================================================================
    batch_user_id_counter = [2000000]
    
    async def test_batch_add():
        batch_user_id_counter[0] += 1
        descriptions = [generate_random_description(50) for _ in range(5)]
        await manager.add_user_profile(batch_user_id_counter[0], descriptions)
    
    print("[7/8] 测试批量添加用户画像（每次5条）...")
    await tester.run_test("批量添加画像(5条)", test_batch_add, iterations=50)
    
    # ========================================================================
    # 测试 8: 并发获取用户画像
    # ========================================================================
    concurrent_user_ids = list(range(1000001, 1000011))
    
    async def test_concurrent_get():
        tasks = [manager.get_user_profile(uid) for uid in concurrent_user_ids]
        await asyncio.gather(*tasks)
    
    print("[8/8] 测试并发获取用户画像（10个并发）...")
    await tester.run_test("并发获取画像(10个)", test_concurrent_get, iterations=50)
    
    # ========================================================================
    # 打印结果
    # ========================================================================
    tester.print_results()
    
    # ========================================================================
    # 额外统计
    # ========================================================================
    print("=" * 80)
    print("额外性能分析".center(80))
    print("=" * 80)
    
    # 测试不同数据量下的性能
    print("\n不同描述数量下的获取性能测试：")
    print(f"{'描述数量':<15} {'平均耗时(ms)':<15} {'ops/s':<15}")
    print("-" * 45)
    
    for desc_count in [1, 5, 10, 20, 30]:
        test_uid = 3000000 + desc_count
        descriptions = [generate_random_description(50) for _ in range(desc_count)]
        await manager.add_user_profile(test_uid, descriptions)
        
        # 测试获取性能
        times: list[float] = []
        for _ in range(50):
            start = time.perf_counter()
            await manager.get_user_profile(test_uid)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_ms = (sum(times) / len(times)) * 1000
        ops = 1000 / avg_ms
        print(f"{desc_count:<15} {avg_ms:<15.3f} {ops:<15.1f}")
    
    print()
    
    # 测试不同描述长度的性能
    print("不同描述长度下的添加性能测试：")
    print(f"{'描述长度(字符)':<15} {'平均耗时(ms)':<15} {'ops/s':<15}")
    print("-" * 45)
    
    for desc_length in [50, 100, 200, 300, 500]:
        test_uid_base = 4000000 + desc_length
        times: list[float] = []
        
        for i in range(30):
            uid = test_uid_base + i
            desc = generate_random_description(desc_length)
            
            start = time.perf_counter()
            await manager.add_user_profile(uid, desc)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_ms = (sum(times) / len(times)) * 1000
        ops = 1000 / avg_ms
        print(f"{desc_length:<15} {avg_ms:<15.3f} {ops:<15.1f}")
    
    print()
    
    # ========================================================================
    # 清理
    # ========================================================================
    await manager.drop_tables()
    await engine.dispose()
    
    # 删除测试数据库
    if test_db_path.exists():
        test_db_path.unlink()
    
    # 删除 WAL 相关文件
    wal_path = test_db_path.with_suffix(".db-wal")
    shm_path = test_db_path.with_suffix(".db-shm")
    if wal_path.exists():
        wal_path.unlink()
    if shm_path.exists():
        shm_path.unlink()
    
    print("=" * 80)
    print("性能测试完成！".center(80))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_performance_tests())
