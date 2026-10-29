"""threading 版后台任务队列。

上传后立刻返回 task_id，真正耗时的「抽取 + 摘要」交给后台线程跑，
FastAPI 的事件循环完全不参与，页面不会卡。

用 threading 而不是 asyncio：OCR 是 CPU 密集的同步代码，丢进线程池
（默认 2 个工作线程）比塞进事件循环更简单也更稳。
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import extract as extract_mod
from . import summarize as summarize_mod

log = logging.getLogger("jobs")

Hook = Callable[["Job"], None]

DEFAULT_WORKERS = 2
KEEP_FINISHED = 200            # 内存里最多留这么多已完成任务
STORED_TEXT_LIMIT = 200_000    # 原文只存前 20 万字，别把 SQLite 撑爆


@dataclass
class Job:
    id: str
    filename: str
    ext: str
    size: int
    path: Path
    provider: str | None = None
    status: str = "queued"        # queued / running / done / failed
    stage: str = "queued"         # queued / extract / summarize / done
    progress: int = 0
    message: str = "已入队"
    text: str = ""
    doc: dict[str, Any] | None = None
    error: str = ""
    chars: int = 0                # 去空白后的有效字数，抽取完成时回填
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return round(max(0.0, end - self.created_at), 2)

    @property
    def provider_name(self) -> str:
        return str((self.doc or {}).get("provider", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "ext": self.ext,
            "size": self.size,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "chars": self.chars or len(self.text),
            "elapsed": self.elapsed,
            "provider": self.provider_name,
            "doc": self.doc,
        }


class JobManager:
    def __init__(self, workers: int = DEFAULT_WORKERS) -> None:
        self.workers = max(1, int(workers))
        self._queue: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._threads: list[threading.Thread] = []
        self._start_lock = threading.Lock()
        self._stop = threading.Event()
        self._hook: Hook | None = None

    # ------------------------------ 对外接口 ------------------------------

    def set_hook(self, hook: Hook | None) -> None:
        """状态变化回调。main.py 用它把进度写进 SQLite。"""
        self._hook = hook

    def submit(
        self,
        path: str | Path,
        filename: str | None = None,
        provider: str | None = None,
        job_id: str | None = None,
    ) -> Job:
        path = Path(path)
        job = Job(
            id=job_id or uuid.uuid4().hex,
            filename=filename or path.name,
            ext=path.suffix.lower(),
            size=path.stat().st_size if path.exists() else 0,
            path=path,
            provider=provider,
        )
        with self._lock:
            self._prune_locked()
            self._jobs[job.id] = job

        self._ensure_workers()
        self._emit(job)                       # 先让上层拿到 queued 状态
        self._queue.put(job)
        log.info("入队 %s  %s", job.id[:8], job.filename)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        return job.to_dict() if job else None

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {"queued": 0, "running": 0, "done": 0, "failed": 0}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
        counts["pending"] = self._queue.qsize()
        counts["workers"] = len(self._threads)
        return counts

    def shutdown(self, wait: bool = False) -> None:
        self._stop.set()
        if wait:
            for thread in list(self._threads):
                thread.join(timeout=2.0)

    # ------------------------------ 工作线程 ------------------------------

    def _ensure_workers(self) -> None:
        with self._start_lock:
            if self._threads:
                return
            for index in range(self.workers):
                thread = threading.Thread(
                    target=self._run,
                    name=f"job-worker-{index + 1}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
            log.info("后台工作线程已启动：%d 个", self.workers)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(job)
            except Exception:                 # 兜底：工作线程绝不允许自己死掉
                log.exception("工作线程异常")
            finally:
                self._queue.task_done()

    def _process(self, job: Job) -> None:
        job.status = "running"
        job.started_at = time.time()
        self._set(job, 5, "开始处理", stage="extract")

        try:
            result = extract_mod.extract(
                job.path,
                on_progress=lambda percent, message: self._set(job, percent, message),
            )
            if not result.text.strip():
                raise extract_mod.ExtractError("empty", "没抽出任何文字：可能是空白图片或加密 PDF")

            job.text = result.text[:STORED_TEXT_LIMIT]
            job.chars = result.chars
            self._set(
                job,
                70,
                f"抽取完成（{result.chars} 字），正在生成摘要",
                stage="summarize",
            )

            doc = summarize_mod.summarize(result.text, provider=job.provider)
            job.doc = doc.to_dict()
            job.status = "done"
            job.stage = "done"
            job.progress = 100
            job.message = "完成"
            log.info("完成 %s  %d 字 → %.2fs", job.id[:8], result.chars, job.elapsed)

        except extract_mod.ExtractError as exc:
            job.status = "failed"
            job.stage = "extract"
            job.error = exc.message
            job.message = "失败：抽取阶段"
            log.warning("抽取失败 %s  %s", job.id[:8], exc.message)

        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "失败：处理异常"
            log.exception("任务异常 %s", job.id[:8])

        finally:
            job.finished_at = time.time()
            self._emit(job)                  # 先把原文和结果落库
            job.text = ""                    # 内存里不再留原文：/result 一律从 SQLite 取
            with self._lock:
                self._prune_locked()         # 完成即回收，不必等下一次 submit

    def _set(self, job: Job, percent: int, message: str, stage: str | None = None) -> None:
        with self._lock:
            job.progress = max(0, min(100, int(percent)))
            job.message = message
            if stage:
                job.stage = stage
        self._emit(job)

    # ------------------------------ 内部 ------------------------------

    def _emit(self, job: Job) -> None:
        hook = self._hook
        if hook is None:
            return
        try:
            hook(job)
        except Exception:                     # 落库失败不能影响任务本身
            log.exception("任务状态落库失败 %s", job.id[:8])

    def _prune_locked(self) -> None:
        if len(self._jobs) <= KEEP_FINISHED * 2:
            return
        finished = [j for j in self._jobs.values() if j.status in ("done", "failed")]
        if len(finished) <= KEEP_FINISHED:
            return
        finished.sort(key=lambda j: j.finished_at)
        for job in finished[: len(finished) - KEEP_FINISHED]:
            self._jobs.pop(job.id, None)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MANAGER = JobManager(workers=_env_int("JOB_WORKERS", DEFAULT_WORKERS))
