from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskStore:
    """Small SQLite store for per-client uploaded recognition tasks."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upload_tasks (
                    id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    params_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    run_dir TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_upload_tasks_client ON upload_tasks(client_id, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_upload_tasks_status ON upload_tasks(status)")

    def mark_interrupted_tasks_failed(self) -> None:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE upload_tasks
                SET status = 'failed',
                    progress = 100,
                    message = '服务重启，任务未完成',
                    updated_at = ?,
                    finished_at = COALESCE(finished_at, ?)
                WHERE status IN ('queued', 'processing')
                """,
                (now, now),
            )

    def create_task(
        self,
        *,
        task_id: str,
        client_id: str,
        original_filename: str,
        stored_path: str,
        file_size_bytes: int,
        params_json: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO upload_tasks (
                    id, client_id, original_filename, stored_path, file_size_bytes,
                    status, progress, message, params_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'queued', 5, '等待处理', ?, ?, ?)
                """,
                (
                    task_id,
                    client_id,
                    original_filename,
                    stored_path,
                    int(file_size_bytes),
                    params_json,
                    now,
                    now,
                ),
            )
        task = self.get_task(task_id)
        if task is None:
            raise RuntimeError("failed to create upload task")
        return task

    def get_task(self, task_id: str, client_id: str | None = None) -> Dict[str, Any] | None:
        sql = "SELECT * FROM upload_tasks WHERE id = ?"
        params: tuple[Any, ...] = (task_id,)
        if client_id is not None:
            sql += " AND client_id = ?"
            params = (task_id, client_id)

        with self._lock, self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def list_tasks(self, client_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM upload_tasks
                WHERE client_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (client_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_task(self, task_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "progress",
            "message",
            "result_json",
            "run_dir",
            "started_at",
            "finished_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = utc_now()

        if updates.get("status") == "processing" and "started_at" not in updates:
            updates["started_at"] = utc_now()

        if updates.get("status") in {"completed", "failed"} and "finished_at" not in updates:
            updates["finished_at"] = utc_now()

        set_sql = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [task_id]

        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE upload_tasks SET {set_sql} WHERE id = ?", params)

    def delete_client_tasks(self, client_id: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM upload_tasks WHERE client_id = ?", (client_id,))
            return int(cur.rowcount or 0)
