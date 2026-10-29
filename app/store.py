"""SQLite 任务存储 + 保留期自动清理。

一张 jobs 表记任务状态，data/uploads/ 放原始文件。
默认保留 24 小时：进程启动时清一次，之后后台线程每 30 分钟清一次，
记录和上传的原文件一起删——比赛"隐私说明"里承诺的，就是这里真的在做。

.env 里可调：
    RETENTION_HOURS          保留小时数，默认 24
    CLEAN_INTERVAL_SECONDS   后台清理间隔，默认 1800
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("store")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


RETENTION_HOURS = _env_float("RETENTION_HOURS", 24.0)
CLEAN_INTERVAL = _env_float("CLEAN_INTERVAL_SECONDS", 1800.0)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    task_id     TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    ext         TEXT NOT NULL DEFAULT '',
    size        INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'queued',
    progress    INTEGER NOT NULL DEFAULT 0,
    message     TEXT NOT NULL DEFAULT '',
    text        TEXT,
    summary     TEXT,
    keywords    TEXT,
    n_chars     INTEGER NOT NULL DEFAULT 0,
    n_summary   INTEGER NOT NULL DEFAULT 0,
    elapsed     REAL NOT NULL DEFAULT 0,
    error       TEXT,
    engine      TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
"""

# update() 只认这些列：字段名来自内部调用，白名单再兜一层
UPDATABLE = frozenset(
    {
        "status",
        "progress",
        "message",
        "text",
        "summary",
        "keywords",
        "n_chars",
        "n_summary",
        "elapsed",
        "error",
        "engine",
    }
)


def _now() -> str:
    return datetime.now().strftime(TIME_FORMAT)


class Store:
    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        upload_dir: str | Path = UPLOAD_DIR,
        retention_hours: float = RETENTION_HOURS,
        text_limit: int = 200_000,
    ) -> None:
        self.db_path = Path(db_path)
        self.upload_dir = Path(upload_dir)
        self.retention_hours = float(retention_hours)
        self.text_limit = int(text_limit)
        self.last_purge: float = 0.0
        self._janitor: threading.Thread | None = None
        self._janitor_stop = threading.Event()
        self._janitor_lock = threading.Lock()

    # ------------------------------ 生命周期 ------------------------------

    def _conn(self) -> sqlite3.Connection:
        """每次操作新开连接：请求线程和后台线程各用各的，省掉线程安全问题。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.commit()

    def _migrate(self, conn) -> None:
        """老库升级：主键 id → task_id；补 engine 列。都不丢历史数据。"""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if not columns:
            return

        if "task_id" not in columns and "id" in columns:
            try:
                conn.execute("ALTER TABLE jobs RENAME COLUMN id TO task_id")
                conn.commit()
                log.info("旧表主键 id 已重命名为 task_id")
            except Exception:
                log.warning("旧表迁移失败，建议删掉 data/app.db 重新生成")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}

        if "engine" not in columns:
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN engine TEXT")
                conn.commit()
                log.info("jobs 表补齐 engine 列")
            except Exception:
                log.warning("engine 列迁移失败，engine_used 字段可能为空")

    def recover_stale(self) -> int:
        """上次进程被强杀时任务会卡在 queued/running，启动时统一标记失败。"""
        with closing(self._conn()) as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'failed', message = '服务重启，任务中断',"
                " error = '服务重启导致任务中断，请重新上传', updated_at = ?"
                " WHERE status IN ('queued', 'running')",
                (_now(),),
            )
            conn.commit()
            count = cursor.rowcount
        if count:
            log.warning("清理了 %d 个上次没跑完的任务", count)
        return count

    # ------------------------------ 写入 ------------------------------

    def create(self, task_id: str, filename: str, ext: str, size: int) -> None:
        ts = _now()
        with closing(self._conn()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO jobs"
                " (task_id, filename, ext, size, status, message, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, 'queued', '已入队', ?, ?)",
                (task_id, filename, ext, size, ts, ts),
            )
            conn.commit()

    def update(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        unknown = set(fields) - UPDATABLE
        if unknown:
            raise ValueError(f"不认识的字段：{sorted(unknown)}")

        fields["updated_at"] = _now()
        clause = ", ".join(f"{key} = ?" for key in fields)
        with closing(self._conn()) as conn:
            conn.execute(
                f"UPDATE jobs SET {clause} WHERE task_id = ?",
                (*fields.values(), task_id),
            )
            conn.commit()

    def save_job(self, job: Any) -> None:
        """jobs.JobManager 的状态回调：把内存里的任务落到表里。"""
        doc = job.doc or {}
        fields: dict[str, Any] = {
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "text": (job.text or "")[: self.text_limit] or None,
            "summary": json.dumps(doc, ensure_ascii=False) if doc else None,
            "keywords": json.dumps(doc.get("keywords", []), ensure_ascii=False) if doc else None,
            "n_chars": getattr(job, "chars", 0) or len(job.text or ""),
            "n_summary": len(str(doc.get("tldr", ""))) if doc else 0,
            "elapsed": job.elapsed,
            "error": job.error or None,
        }
        if doc:                                   # 只在有结果时写引擎，别把中间态冲掉
            fields["engine"] = str(doc.get("provider", ""))[:40] or None
        self.update(job.id, **fields)

    # ------------------------------ 查询 ------------------------------

    def get(self, task_id: str) -> dict[str, Any] | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(200, int(limit)))
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT task_id, filename, status, progress, n_chars, elapsed, engine, created_at"
                " FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with closing(self._conn()) as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            oldest = conn.execute("SELECT MIN(created_at) FROM jobs").fetchone()[0]
        counts = {row["status"]: row["n"] for row in rows}
        files, size = self._disk_usage()
        return {
            "records": total,
            "by_status": counts,
            "oldest": oldest or "",
            "upload_files": files,
            "upload_bytes": size,
            "retention_hours": self.retention_hours,
            "last_purge": (
                datetime.fromtimestamp(self.last_purge).strftime(TIME_FORMAT) if self.last_purge else ""
            ),
        }

    # ------------------------------ 清理 ------------------------------

    def cutoff(self, hours: float | None = None) -> str:
        hours = self.retention_hours if hours is None else float(hours)
        return (datetime.now() - timedelta(hours=hours)).strftime(TIME_FORMAT)

    def purge_expired(self, hours: float | None = None, dry_run: bool = False) -> dict[str, Any]:
        """删掉超过保留期的记录和原文件，顺手清掉没人认领的孤儿文件。

        返回 {records, files, bytes, cutoff, dry_run}；dry_run=True 只统计不删。
        """
        hours = self.retention_hours if hours is None else float(hours)
        cutoff = self.cutoff(hours)

        with closing(self._conn()) as conn:
            expired = conn.execute(
                "SELECT task_id, ext FROM jobs WHERE created_at < ?", (cutoff,)
            ).fetchall()
            alive = {row["task_id"] for row in conn.execute("SELECT task_id FROM jobs")}

        victims = [(row["task_id"], row["ext"] or "") for row in expired]
        doomed = [self.upload_dir / f"{task_id}{ext}" for task_id, ext in victims]

        # 孤儿文件：记录没了但文件还在（比如建记录那一步就失败了）
        orphans: list[Path] = []
        if self.upload_dir.is_dir():
            for path in self.upload_dir.iterdir():
                if not path.is_file() or path in doomed or path.stem in alive:
                    continue
                try:
                    if time.time() - path.stat().st_mtime < hours * 3600:
                        continue          # 刚上传、还没建记录，给一次机会
                except OSError:
                    continue
                orphans.append(path)

        files = 0
        freed = 0
        for path in doomed + orphans:
            try:
                size = path.stat().st_size
            except OSError:
                continue                  # 文件已经不在了
            files += 1
            freed += size
            if not dry_run:
                path.unlink(missing_ok=True)

        if not dry_run and victims:
            with closing(self._conn()) as conn:
                conn.executemany(
                    "DELETE FROM jobs WHERE task_id = ?", [(task_id,) for task_id, _ in victims]
                )
                conn.commit()

        result = {
            "records": len(victims),
            "files": files,
            "bytes": freed,
            "cutoff": cutoff,
            "dry_run": dry_run,
        }
        if not dry_run:
            self.last_purge = time.time()
            if result["records"] or result["files"]:
                log.info(
                    "清理过期数据：%d 条记录 / %d 个文件 / %.1f KB（截止 %s）",
                    result["records"],
                    result["files"],
                    freed / 1024,
                    cutoff,
                )
        return result

    def _disk_usage(self) -> tuple[int, int]:
        if not self.upload_dir.is_dir():
            return 0, 0
        files = 0
        size = 0
        for path in self.upload_dir.iterdir():
            if not path.is_file():
                continue
            files += 1
            try:
                size += path.stat().st_size
            except OSError:
                pass
        return files, size

    # ------------------------------ 定时清理 ------------------------------

    def start_janitor(self, interval: float | None = None) -> None:
        interval = CLEAN_INTERVAL if interval is None else float(interval)
        with self._janitor_lock:
            if self._janitor is not None and self._janitor.is_alive():
                return
            self._janitor_stop.clear()
            self._janitor = threading.Thread(
                target=self._janitor_loop,
                args=(max(0.1, interval),),
                name="store-janitor",
                daemon=True,
            )
            self._janitor.start()
            log.info("清理线程已启动，每 %.0f 分钟扫一次（保留 %.0f 小时）", interval / 60, self.retention_hours)

    def stop_janitor(self) -> None:
        self._janitor_stop.set()
        thread = self._janitor
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._janitor = None

    def _janitor_loop(self, interval: float) -> None:
        while not self._janitor_stop.wait(interval):
            try:
                self.purge_expired()
            except Exception:                 # 清理失败不能把线程搞死
                log.exception("定时清理失败")


STORE = Store()
