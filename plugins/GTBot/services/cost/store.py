from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from plugins.GTBot.ConfigManager import total_config

from .models import CostLeaderboardEntry, CostRecord, CostSummary


class CostLedgerStore:
    """负责统一消费账本 SQLite 读写的底层存储对象。

    该类只处理明细表结构初始化、幂等写入和基础查询聚合，不承担权限判断、
    当前请求上下文解析或聊天模型计费规则解释等上层职责。所有写操作都通过同一把
    进程内锁串行化，避免多协程并发写入时出现重复建表或写入竞争。
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """初始化消费账本存储。

        Args:
            db_path: 自定义数据库路径。未传入时使用 GTBot 数据目录下的
                `cost_ledger.sqlite3`。
        """

        data_dir = total_config.get_data_dir_path()
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = Path(db_path) if db_path is not None else data_dir / "cost_ledger.sqlite3"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def db_path(self) -> Path:
        """返回当前账本数据库文件的绝对路径。

        Returns:
            当前账本数据库文件路径。
        """

        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """创建一条配置为 `Row` 工厂的新 SQLite 连接。

        Returns:
            可按列名访问结果的 SQLite 连接。
        """

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化消费账本表结构和索引。

        当前版本仅维护一张明细表，并围绕个人账单、时间范围查询、按群过滤、
        来源筛选和 `response_id` 回溯建立最小必要索引。
        """

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cost_ledger (
                        event_id TEXT PRIMARY KEY,
                        occurred_at REAL NOT NULL,
                        source_type TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        category TEXT NOT NULL,
                        owner_user_id INTEGER NOT NULL,
                        actor_user_id INTEGER NOT NULL,
                        group_id INTEGER,
                        session_id TEXT,
                        response_id TEXT,
                        provider TEXT,
                        model_name TEXT,
                        billing_mode TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        unit_price REAL,
                        amount REAL NOT NULL,
                        currency TEXT NOT NULL,
                        extra_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cost_ledger_owner_time "
                    "ON cost_ledger(owner_user_id, occurred_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cost_ledger_occurred_at "
                    "ON cost_ledger(occurred_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cost_ledger_group_time "
                    "ON cost_ledger(group_id, occurred_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cost_ledger_source_name "
                    "ON cost_ledger(source_name)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cost_ledger_response_id "
                    "ON cost_ledger(response_id)"
                )
                conn.commit()

    def insert_record(self, record: CostRecord) -> bool:
        """以幂等方式插入一条消费记录。

        若 `event_id` 已存在，则视为重复记账并返回 `False`，而不是覆盖原记录。

        Args:
            record: 待持久化的消费明细记录。

        Returns:
            `True` 表示本次成功插入新记录，`False` 表示命中幂等去重。
        """

        payload = (
            record.event_id,
            float(record.occurred_at),
            str(record.source_type),
            str(record.source_name),
            str(record.category),
            int(record.owner_user_id),
            int(record.actor_user_id),
            int(record.group_id) if record.group_id is not None else None,
            record.session_id,
            record.response_id,
            record.provider,
            record.model_name,
            str(record.billing_mode),
            float(record.quantity),
            float(record.unit_price) if record.unit_price is not None else None,
            float(record.amount),
            str(record.currency),
            json.dumps(record.extra, ensure_ascii=False, sort_keys=True),
        )
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO cost_ledger(
                        event_id, occurred_at, source_type, source_name, category,
                        owner_user_id, actor_user_id, group_id, session_id, response_id,
                        provider, model_name, billing_mode, quantity, unit_price,
                        amount, currency, extra_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
                conn.commit()
                return int(cursor.rowcount or 0) > 0

    def list_records(
        self,
        *,
        owner_user_id: int | None = None,
        start_at: float | None = None,
        end_at: float | None = None,
        source_name: str | None = None,
        response_id: str | None = None,
        group_id: int | None = None,
        limit: int = 20,
    ) -> list[CostRecord]:
        """按过滤条件返回消费明细列表。

        Args:
            owner_user_id: 可选；仅返回指定归属用户的账单。
            start_at: 可选；仅返回发生时间大于等于该值的账单。
            end_at: 可选；仅返回发生时间小于该值的账单。
            source_name: 可选；仅返回指定来源的账单。
            response_id: 可选；仅返回指定响应 ID 的账单。
            group_id: 可选；仅返回指定群范围内的账单。
            limit: 最多返回的账单条数，按时间倒序。

        Returns:
            按时间倒序排列的消费明细列表。
        """

        sql = [
            """
            SELECT event_id, occurred_at, source_type, source_name, category,
                   owner_user_id, actor_user_id, group_id, session_id, response_id,
                   provider, model_name, billing_mode, quantity, unit_price,
                   amount, currency, extra_json
            FROM cost_ledger
            WHERE 1=1
            """
        ]
        params: list[Any] = []
        self._append_filters(
            sql=sql,
            params=params,
            owner_user_id=owner_user_id,
            start_at=start_at,
            end_at=end_at,
            source_name=source_name,
            response_id=response_id,
            group_id=group_id,
        )
        sql.append("ORDER BY occurred_at DESC, event_id DESC LIMIT ?")
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def summarize(
        self,
        *,
        owner_user_id: int | None = None,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
    ) -> CostSummary:
        """计算指定过滤条件下的消费总额。

        Args:
            owner_user_id: 可选；仅统计指定归属用户。
            start_at: 可选；时间范围起点，包含该时间。
            end_at: 可选；时间范围终点，不包含该时间。
            group_id: 可选；仅统计指定群范围内的账单。

        Returns:
            包含总金额和账单数量的汇总结果。
        """

        sql = [
            """
            SELECT COALESCE(SUM(amount), 0.0) AS total_amount,
                   COUNT(1) AS record_count
            FROM cost_ledger
            WHERE 1=1
            """
        ]
        params: list[Any] = []
        self._append_filters(
            sql=sql,
            params=params,
            owner_user_id=owner_user_id,
            start_at=start_at,
            end_at=end_at,
            source_name=None,
            response_id=None,
            group_id=group_id,
        )
        with self._connect() as conn:
            row = conn.execute(" ".join(sql), params).fetchone()
        return CostSummary(
            total_amount=float((row["total_amount"] if row is not None else 0.0) or 0.0),
            currency="CNY",
            record_count=int((row["record_count"] if row is not None else 0) or 0),
        )

    def summarize_by_source(
        self,
        *,
        owner_user_id: int,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
    ) -> list[tuple[str, CostSummary]]:
        """按来源名称聚合指定用户的消费总额。

        Args:
            owner_user_id: 需要聚合的归属用户 QQ 用户 ID。
            start_at: 可选；时间范围起点，包含该时间。
            end_at: 可选；时间范围终点，不包含该时间。
            group_id: 可选；仅统计指定群范围内的账单。

        Returns:
            由来源名称和对应汇总结果组成的列表，按总额倒序排列。
        """

        sql = [
            """
            SELECT source_name,
                   COALESCE(SUM(amount), 0.0) AS total_amount,
                   COUNT(1) AS record_count
            FROM cost_ledger
            WHERE owner_user_id = ?
            """
        ]
        params: list[Any] = [int(owner_user_id)]
        self._append_optional_time_and_group_filters(sql=sql, params=params, start_at=start_at, end_at=end_at, group_id=group_id)
        sql.append("GROUP BY source_name ORDER BY total_amount DESC, source_name ASC")
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [
            (
                str(row["source_name"]),
                CostSummary(
                    total_amount=float(row["total_amount"] or 0.0),
                    currency="CNY",
                    record_count=int(row["record_count"] or 0),
                ),
            )
            for row in rows
        ]

    def leaderboard(
        self,
        *,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
        limit: int = 20,
    ) -> list[CostLeaderboardEntry]:
        """生成指定口径下的用户消费排行榜。

        Args:
            start_at: 可选；时间范围起点，包含该时间。
            end_at: 可选；时间范围终点，不包含该时间。
            group_id: 可选；仅统计指定群范围内的账单。
            limit: 最多返回的排行榜条数。

        Returns:
            按总金额倒序排列的排行榜条目列表。
        """

        sql = [
            """
            SELECT owner_user_id,
                   COALESCE(SUM(amount), 0.0) AS total_amount,
                   COUNT(1) AS record_count
            FROM cost_ledger
            WHERE 1=1
            """
        ]
        params: list[Any] = []
        self._append_optional_time_and_group_filters(sql=sql, params=params, start_at=start_at, end_at=end_at, group_id=group_id)
        sql.append(
            "GROUP BY owner_user_id "
            "ORDER BY total_amount DESC, record_count DESC, owner_user_id ASC "
            "LIMIT ?"
        )
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [
            CostLeaderboardEntry(
                owner_user_id=int(row["owner_user_id"]),
                total_amount=float(row["total_amount"] or 0.0),
                currency="CNY",
                record_count=int(row["record_count"] or 0),
            )
            for row in rows
        ]

    def _append_filters(
        self,
        *,
        sql: list[str],
        params: list[Any],
        owner_user_id: int | None,
        start_at: float | None,
        end_at: float | None,
        source_name: str | None,
        response_id: str | None,
        group_id: int | None,
    ) -> None:
        """为消费明细查询追加统一过滤条件。

        Args:
            sql: 待追加 SQL 片段的列表。
            params: 与 SQL 对应的参数列表。
            owner_user_id: 可选归属用户过滤条件。
            start_at: 可选时间范围起点。
            end_at: 可选时间范围终点。
            source_name: 可选来源名称过滤条件。
            response_id: 可选响应 ID 过滤条件。
            group_id: 可选群范围过滤条件。
        """

        if owner_user_id is not None:
            sql.append("AND owner_user_id = ?")
            params.append(int(owner_user_id))
        self._append_optional_time_and_group_filters(
            sql=sql,
            params=params,
            start_at=start_at,
            end_at=end_at,
            group_id=group_id,
        )
        if source_name:
            sql.append("AND source_name = ?")
            params.append(str(source_name))
        if response_id:
            sql.append("AND response_id = ?")
            params.append(str(response_id))

    @staticmethod
    def _append_optional_time_and_group_filters(
        *,
        sql: list[str],
        params: list[Any],
        start_at: float | None,
        end_at: float | None,
        group_id: int | None,
    ) -> None:
        """为查询追加可选的时间范围和群范围过滤条件。

        Args:
            sql: 待追加 SQL 片段的列表。
            params: 与 SQL 对应的参数列表。
            start_at: 可选时间范围起点。
            end_at: 可选时间范围终点。
            group_id: 可选群范围过滤条件。
        """

        if start_at is not None:
            sql.append("AND occurred_at >= ?")
            params.append(float(start_at))
        if end_at is not None:
            sql.append("AND occurred_at < ?")
            params.append(float(end_at))
        if group_id is not None:
            sql.append("AND group_id = ?")
            params.append(int(group_id))

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CostRecord:
        """将 SQLite 行对象转换为消费记录。

        Args:
            row: 原始数据库行对象。

        Returns:
            规整化后的消费记录对象。
        """

        extra_raw = row["extra_json"]
        extra: dict[str, Any]
        if isinstance(extra_raw, str) and extra_raw.strip():
            loaded = json.loads(extra_raw)
            extra = loaded if isinstance(loaded, dict) else {}
        else:
            extra = {}
        return CostRecord(
            event_id=str(row["event_id"]),
            occurred_at=float(row["occurred_at"]),
            source_type=str(row["source_type"]),  # type: ignore[arg-type]
            source_name=str(row["source_name"]),
            category=str(row["category"]),
            owner_user_id=int(row["owner_user_id"]),
            actor_user_id=int(row["actor_user_id"]),
            group_id=int(row["group_id"]) if row["group_id"] is not None else None,
            session_id=str(row["session_id"]) if row["session_id"] is not None else None,
            response_id=str(row["response_id"]) if row["response_id"] is not None else None,
            provider=str(row["provider"]) if row["provider"] is not None else None,
            model_name=str(row["model_name"]) if row["model_name"] is not None else None,
            billing_mode=str(row["billing_mode"]),  # type: ignore[arg-type]
            quantity=float(row["quantity"]),
            unit_price=float(row["unit_price"]) if row["unit_price"] is not None else None,
            amount=float(row["amount"]),
            currency=str(row["currency"]),
            extra=extra,
        )
