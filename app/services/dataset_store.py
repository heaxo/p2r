from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DatasetStore:
    """SQLite persistence for named image datasets and recognition results."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'unrecognized',
                    source TEXT NOT NULL DEFAULT 'manual',
                    expert_importer TEXT NOT NULL DEFAULT 'Procesos',
                    copied_from_id TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    recognized_at TEXT,
                    recognition_started_at TEXT,
                    recognition_finished_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_items (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    row_order INTEGER NOT NULL DEFAULT 0,
                    image_path TEXT NOT NULL DEFAULT '',
                    original_filename TEXT NOT NULL DEFAULT '',
                    plate_no TEXT NOT NULL DEFAULT '',
                    quantity INTEGER,
                    material TEXT NOT NULL DEFAULT '',
                    thickness_mm REAL,
                    dxf_target_size_1_mm REAL,
                    dxf_target_size_2_mm REAL,
                    dxf_target_x_mm REAL,
                    dxf_target_y_mm REAL,
                    paper_source TEXT NOT NULL DEFAULT 'sam2',
                    a4_orientation TEXT NOT NULL DEFAULT 'auto',
                    plate_point_ratio TEXT,
                    paper_point_ratio TEXT,
                    use_plate_perspective INTEGER NOT NULL DEFAULT 0,
                    dxf_notch_fill_enabled INTEGER NOT NULL DEFAULT 0,
                    dxf_notch_fill_max_width_mm REAL NOT NULL DEFAULT 80.0,
                    dxf_notch_fill_max_depth_mm REAL NOT NULL DEFAULT 25.0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    result_json TEXT,
                    run_dir TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_items_dataset ON dataset_items(dataset_id, row_order)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_items_status ON dataset_items(status)")
            dataset_columns = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)").fetchall()}
            if "expert_importer" not in dataset_columns:
                conn.execute("ALTER TABLE datasets ADD COLUMN expert_importer TEXT NOT NULL DEFAULT 'Procesos'")
            item_columns = {row["name"] for row in conn.execute("PRAGMA table_info(dataset_items)").fetchall()}
            if "quantity" not in item_columns:
                conn.execute("ALTER TABLE dataset_items ADD COLUMN quantity INTEGER")
            if "paper_source" not in item_columns:
                conn.execute("ALTER TABLE dataset_items ADD COLUMN paper_source TEXT NOT NULL DEFAULT 'sam2'")
            if "a4_orientation" not in item_columns:
                conn.execute("ALTER TABLE dataset_items ADD COLUMN a4_orientation TEXT NOT NULL DEFAULT 'auto'")
            if "plate_point_ratio" not in item_columns:
                conn.execute("ALTER TABLE dataset_items ADD COLUMN plate_point_ratio TEXT")
            if "paper_point_ratio" not in item_columns:
                conn.execute("ALTER TABLE dataset_items ADD COLUMN paper_point_ratio TEXT")

    def mark_interrupted_work_failed(self) -> None:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE dataset_items
                SET status = 'failed',
                    progress = 100,
                    message = '服务重启，识别未完成',
                    updated_at = ?,
                    finished_at = COALESCE(finished_at, ?)
                WHERE status IN ('queued', 'processing')
                """,
                (now, now),
            )
            conn.execute(
                """
                UPDATE datasets
                SET status = 'failed',
                    last_error = COALESCE(NULLIF(last_error, ''), '服务重启，识别未完成'),
                    updated_at = ?,
                    recognition_finished_at = COALESCE(recognition_finished_at, ?)
                WHERE status = 'recognizing'
                """,
                (now, now),
            )

    def create_dataset(
        self,
        *,
        dataset_id: str,
        name: str,
        source: str,
        copied_from_id: str | None = None,
        expert_importer: str = "Procesos",
    ) -> Dict[str, Any]:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    id, name, status, source, expert_importer, copied_from_id, created_at, updated_at
                )
                VALUES (?, ?, 'unrecognized', ?, ?, ?, ?, ?)
                """,
                (dataset_id, name, source, expert_importer, copied_from_id, now, now),
            )
        row = self.get_dataset(dataset_id)
        if row is None:
            raise RuntimeError("failed to create dataset")
        return row

    def name_exists(self, name: str, exclude_dataset_id: str | None = None) -> bool:
        sql = "SELECT 1 FROM datasets WHERE name = ?"
        params: tuple[Any, ...] = (name,)
        if exclude_dataset_id:
            sql += " AND id <> ?"
            params = (name, exclude_dataset_id)
        with self._lock, self._connect() as conn:
            return conn.execute(sql, params).fetchone() is not None

    def get_dataset(self, dataset_id: str) -> Dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            return dict(row) if row else None

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.*,
                    COUNT(i.id) AS item_count,
                    SUM(CASE WHEN i.status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
                    SUM(CASE WHEN i.status = 'failed' THEN 1 ELSE 0 END) AS failed_count
                FROM datasets d
                LEFT JOIN dataset_items i ON i.dataset_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC, d.id DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def update_dataset(self, dataset_id: str, **fields: Any) -> None:
        allowed = {
            "name",
            "status",
            "expert_importer",
            "last_error",
            "recognized_at",
            "recognition_started_at",
            "recognition_finished_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = utc_now()
        set_sql = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [dataset_id]

        with self._lock, self._connect() as conn:
            cur = conn.execute(f"UPDATE datasets SET {set_sql} WHERE id = ?", params)
            if cur.rowcount == 0:
                raise KeyError(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            if cur.rowcount == 0:
                raise KeyError(dataset_id)

    def next_item_order(self, dataset_id: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(row_order), 0) + 1 AS next_order FROM dataset_items WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            return int(row["next_order"] if row else 1)

    def add_item(self, **fields: Any) -> Dict[str, Any]:
        now = utc_now()
        values = {
            "id": fields["id"],
            "dataset_id": fields["dataset_id"],
            "row_order": int(fields.get("row_order") or 0),
            "image_path": fields.get("image_path") or "",
            "original_filename": fields.get("original_filename") or "",
            "plate_no": fields.get("plate_no") or "",
            "quantity": fields.get("quantity"),
            "material": fields.get("material") or "",
            "thickness_mm": fields.get("thickness_mm"),
            "dxf_target_size_1_mm": fields.get("dxf_target_size_1_mm"),
            "dxf_target_size_2_mm": fields.get("dxf_target_size_2_mm"),
            "dxf_target_x_mm": fields.get("dxf_target_x_mm"),
            "dxf_target_y_mm": fields.get("dxf_target_y_mm"),
            "paper_source": fields.get("paper_source") or "sam2",
            "a4_orientation": fields.get("a4_orientation") or "auto",
            "plate_point_ratio": fields.get("plate_point_ratio"),
            "paper_point_ratio": fields.get("paper_point_ratio"),
            "use_plate_perspective": 1 if fields.get("use_plate_perspective") else 0,
            "dxf_notch_fill_enabled": 1 if fields.get("dxf_notch_fill_enabled") else 0,
            "dxf_notch_fill_max_width_mm": float(fields.get("dxf_notch_fill_max_width_mm") or 80.0),
            "dxf_notch_fill_max_depth_mm": float(fields.get("dxf_notch_fill_max_depth_mm") or 25.0),
            "created_at": now,
            "updated_at": now,
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)

        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO dataset_items ({columns}) VALUES ({placeholders})",
                list(values.values()),
            )
            conn.execute("UPDATE datasets SET status = 'unrecognized', updated_at = ? WHERE id = ?", (now, fields["dataset_id"]))
        row = self.get_item(fields["id"])
        if row is None:
            raise RuntimeError("failed to create dataset item")
        return row

    def get_item(self, item_id: str, dataset_id: str | None = None) -> Dict[str, Any] | None:
        sql = "SELECT * FROM dataset_items WHERE id = ?"
        params: tuple[Any, ...] = (item_id,)
        if dataset_id:
            sql += " AND dataset_id = ?"
            params = (item_id, dataset_id)
        with self._lock, self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def list_items(self, dataset_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM dataset_items
                WHERE dataset_id = ?
                ORDER BY row_order ASC, created_at ASC, id ASC
                """,
                (dataset_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_item(self, item_id: str, dataset_id: str, **fields: Any) -> None:
        allowed = {
            "row_order",
            "image_path",
            "original_filename",
            "plate_no",
            "quantity",
            "material",
            "thickness_mm",
            "dxf_target_size_1_mm",
            "dxf_target_size_2_mm",
            "dxf_target_x_mm",
            "dxf_target_y_mm",
            "paper_source",
            "a4_orientation",
            "plate_point_ratio",
            "paper_point_ratio",
            "use_plate_perspective",
            "dxf_notch_fill_enabled",
            "dxf_notch_fill_max_width_mm",
            "dxf_notch_fill_max_depth_mm",
            "status",
            "progress",
            "message",
            "result_json",
            "run_dir",
            "started_at",
            "finished_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return

        for key in ("use_plate_perspective", "dxf_notch_fill_enabled"):
            if key in updates:
                updates[key] = 1 if updates[key] else 0

        updates["updated_at"] = utc_now()
        if updates.get("status") == "processing" and "started_at" not in updates:
            updates["started_at"] = utc_now()
        if updates.get("status") in {"completed", "failed"} and "finished_at" not in updates:
            updates["finished_at"] = utc_now()

        set_sql = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [item_id, dataset_id]

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE dataset_items SET {set_sql} WHERE id = ? AND dataset_id = ?",
                params,
            )
            if cur.rowcount == 0:
                raise KeyError(item_id)
            conn.execute(
                "UPDATE datasets SET updated_at = ?, status = CASE WHEN status = 'recognized' THEN 'unrecognized' ELSE status END WHERE id = ?",
                (utc_now(), dataset_id),
            )

    def delete_item(self, item_id: str, dataset_id: str) -> None:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM dataset_items WHERE id = ? AND dataset_id = ?", (item_id, dataset_id))
            if cur.rowcount == 0:
                raise KeyError(item_id)
            conn.execute("UPDATE datasets SET status = 'unrecognized', updated_at = ? WHERE id = ?", (utc_now(), dataset_id))

    def begin_recognition(self, dataset_id: str) -> int:
        now = utc_now()
        with self._lock, self._connect() as conn:
            item_count = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM dataset_items
                WHERE dataset_id = ?
                  AND status != 'completed'
                """,
                (dataset_id,),
            ).fetchone()["count"]
            if int(item_count or 0) == 0:
                raise ValueError("数据集中没有未识别的数据")
            conn.execute(
                """
                UPDATE datasets
                SET status = 'recognizing',
                    last_error = NULL,
                    recognition_started_at = ?,
                    recognition_finished_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, dataset_id),
            )
            conn.execute(
                """
                UPDATE dataset_items
                SET status = 'queued',
                    progress = 5,
                    message = '等待处理',
                    result_json = NULL,
                    run_dir = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = ?
                WHERE dataset_id = ?
                  AND status != 'completed'
                """,
                (now, dataset_id),
            )
            return int(item_count or 0)

    def finish_recognition(self, dataset_id: str, *, failed_count: int, last_error: str | None = None) -> None:
        now = utc_now()
        status = "failed" if failed_count else "recognized"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE datasets
                SET status = ?,
                    last_error = ?,
                    recognized_at = CASE WHEN ? = 'recognized' THEN ? ELSE recognized_at END,
                    recognition_finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (status, last_error, status, now, now, now, dataset_id),
            )

    def clear_item_recognition(self, item_id: str, dataset_id: str) -> int:
        now = utc_now()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE dataset_items
                SET status = 'pending',
                    progress = 0,
                    message = '',
                    result_json = NULL,
                    run_dir = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = ?
                WHERE id = ?
                  AND dataset_id = ?
                  AND status = 'completed'
                """,
                (now, item_id, dataset_id),
            )
            if cur.rowcount == 0:
                exists = conn.execute(
                    "SELECT 1 FROM dataset_items WHERE id = ? AND dataset_id = ?",
                    (item_id, dataset_id),
                ).fetchone()
                if exists is None:
                    raise KeyError(item_id)
            if cur.rowcount:
                conn.execute(
                    """
                    UPDATE datasets
                    SET status = 'unrecognized',
                        recognized_at = NULL,
                        recognition_finished_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, dataset_id),
                )
            return int(cur.rowcount or 0)

    def clear_dataset_recognition(self, dataset_id: str) -> int:
        now = utc_now()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE dataset_items
                SET status = 'pending',
                    progress = 0,
                    message = '',
                    result_json = NULL,
                    run_dir = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = ?
                WHERE dataset_id = ?
                  AND status = 'completed'
                """,
                (now, dataset_id),
            )
            dataset_cur = conn.execute(
                """
                UPDATE datasets
                SET status = 'unrecognized',
                    recognized_at = NULL,
                    recognition_finished_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, dataset_id),
            )
            if dataset_cur.rowcount == 0:
                raise KeyError(dataset_id)
            return int(cur.rowcount or 0)
