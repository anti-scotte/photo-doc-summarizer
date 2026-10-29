"""纯本地图文摘要系统 · FastAPI 入口。

本文件只管 HTTP 这一层：
    POST /upload       收文件 → 校验 → 落盘 → 建任务 → 立刻返回 202 + task_id
    GET  /status/{id}  查进度
    GET  /result/{id}  取结构化摘要
    GET  /history      最近处理过的任务（超过保留期就查不到了）
    GET  /info         摘要引擎、任务队列、存储与隐私状态
真正的重活在 app/jobs.py 的后台线程里跑，状态和清理在 app/store.py。

启动：uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .extract import ALLOWED_EXTS, ExtractError, sniff
from .jobs import MANAGER
from .store import STORE
from .summarize import active_engine, provider_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

# ============================== 配置 ==============================

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

MAX_MB = 20
MAX_BYTES = MAX_MB * 1024 * 1024
CHUNK = 1 << 20

# ExtractError.kind → HTTP 状态码
KIND_STATUS = {
    "unsupported": 415,
    "bad_file": 422,
    "empty": 422,
    "missing_dep": 500,
}


class ApiError(Exception):
    """业务异常，统一由 exception_handler 转成 {"ok": false, "error": {...}}"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _as_api_error(exc: ExtractError) -> ApiError:
    return ApiError(KIND_STATUS.get(exc.kind, 400), exc.kind, exc.message)


def _serve_url() -> str:
    """run.bat 可能因为端口占用改到 8001/8002，日志里要打印真实地址。"""
    return f"http://127.0.0.1:{os.getenv('SERVE_PORT') or '8000'}"


# ============================== 应用 ==============================

@asynccontextmanager
async def lifespan(_: FastAPI):
    STORE.init()
    STORE.recover_stale()

    # 启动先清一次：上次进程可能没等到定时线程就退出了
    cleaned = STORE.purge_expired()
    if cleaned["records"] or cleaned["files"]:
        log.info("启动清理：%d 条记录 / %d 个文件", cleaned["records"], cleaned["files"])

    MANAGER.set_hook(STORE.save_job)       # 任务状态变化 → 落库
    STORE.start_janitor()                  # 之后每 30 分钟再扫一次

    log.info("摘要引擎 %s", provider_info()["active"])
    log.info("数据保留 %d 小时，数据目录 %s", STORE.retention_hours, STORE.db_path.parent)
    log.info("准备就绪 → %s", _serve_url())
    yield
    STORE.stop_janitor()
    MANAGER.shutdown(wait=False)
    log.info("服务已退出")


app = FastAPI(
    title="纯本地图文摘要系统",
    version="0.3.0",
    description="上传图片 / PDF / Word，后台线程抽取文本并生成结构化摘要。全部本地运行，不联网，24 小时后自动删除。",
    lifespan=lifespan,
)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.middleware("http")
async def reject_oversized_upload(request: Request, call_next):
    """带 Content-Length 的超大请求，在解析 multipart 之前就挡掉。

    FastAPI 会先把整个请求体收完、再进路由函数：一个 50MB 的文件虽然最终
    也会返回 413，但那趟临时文件已经落过盘了。这里提前拒绝，省掉那次 IO。
    （没有 Content-Length 的 chunked 上传仍然由路由里的流式计数兜底。）
    """
    if request.method == "POST" and request.url.path == "/upload":
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BYTES + 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={
                    "ok": False,
                    "error": {"code": "too_large", "message": f"文件超过 {MAX_MB} MB 上限"},
                },
            )
    return await call_next(request)


# 只放行本机页面跨域调用，绝不能用 "*"。
# 这是个能收文件、能读摘要的本地服务：一旦放开 "*"，你浏览任意网页时，
# 那个网页都能悄悄往 127.0.0.1:8000 传文件并把摘要读走。
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

NO_PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>图文摘要系统</title>
<body style="font-family:system-ui;padding:56px;line-height:1.9;color:#1b2430">
<h2>服务已经起来了</h2>
<p>但还没有前端页面：请把页面放到 <code>web/index.html</code>，刷新本页即可。</p>
<p>想先验证接口，可以打开 <a href="/docs">/docs</a>：
   <code>POST /upload</code> → <code>GET /status/{id}</code> → <code>GET /result/{id}</code></p>
</body></html>"""


# ------------------------------ 路由 ------------------------------

@app.get("/", response_class=HTMLResponse, summary="前端页面")
def index() -> Any:
    page = WEB_DIR / "index.html"
    if page.is_file():
        return FileResponse(str(page))
    return HTMLResponse(NO_PAGE)


@app.post("/upload", summary="上传文件，立刻返回 task_id")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    filename = (file.filename or "").strip()
    if not filename:
        raise ApiError(400, "no_filename", "没有拿到文件名")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        allowed = "  ".join(sorted(ALLOWED_EXTS))
        raise ApiError(415, "unsupported_type", f"不支持 {ext or '无扩展名'}；可用类型：{allowed}")

    task_id = uuid.uuid4().hex
    dest = STORE.upload_dir / f"{task_id}{ext}"
    size = 0
    head = b""

    try:
        STORE.upload_dir.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:                  # 边写边数，不信任 Content-Length
                    raise ApiError(413, "too_large", f"文件超过 {MAX_MB} MB 上限")
                if not head:
                    head = chunk[:8]
                fh.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)                  # 半截文件不许留在盘上
        raise
    finally:
        await file.close()

    if size == 0:
        dest.unlink(missing_ok=True)
        raise ApiError(400, "empty_file", "文件是空的")

    try:
        sniff(head, ext)
    except ExtractError as exc:
        dest.unlink(missing_ok=True)
        raise _as_api_error(exc) from exc

    STORE.create(task_id, filename, ext, size)
    MANAGER.submit(dest, filename=filename, job_id=task_id)   # 立即返回，后台线程接手
    log.info("已接收 %s  %s  %.1f KB", task_id[:8], filename, size / 1024)

    return JSONResponse(
        status_code=202,
        content={
            "ok": True,
            "data": {
                "id": task_id,
                "status": "queued",
                "filename": filename,
                "size": size,
                "engine_used": active_engine(),          # 预计使用的引擎（结果出来后以 /result 为准）
                "status_url": f"/status/{task_id}",
                "result_url": f"/result/{task_id}",
            },
        },
    )


@app.get("/status/{task_id}", summary="查询处理进度")
def task_status(task_id: str) -> dict[str, Any]:
    row = _require_task(task_id)
    return {
        "ok": True,
        "data": {
            "id": row["task_id"],
            "filename": row["filename"],
            "status": row["status"],
            "progress": row["progress"],
            "message": row["message"],
            "chars": row["n_chars"],
            "elapsed": round(row["elapsed"] or 0, 2),
            "error": row["error"] or "",
            "engine_used": _engine_of(row),          # 完成前是预测值，完成后是真实引擎
        },
    }


@app.get("/result/{task_id}", summary="获取结构化摘要")
def task_result(task_id: str) -> dict[str, Any]:
    row = _require_task(task_id)

    if row["status"] == "failed":
        raise ApiError(409, "task_failed", row["error"] or "处理失败")
    if row["status"] != "done":
        raise ApiError(409, "not_ready", f"任务还没完成（{row['status']} {row['progress']}%）")

    try:
        doc = json.loads(row["summary"] or "{}")
    except json.JSONDecodeError:
        doc = {}

    return {
        "ok": True,
        "data": {
            "id": row["task_id"],
            "filename": row["filename"],
            "doc": doc,
            "engine_used": str(doc.get("provider") or _engine_of(row)),
            "source_chars": row["n_chars"],
            "elapsed": round(row["elapsed"] or 0, 2),
            "created_at": row["created_at"],
            "text_preview": (row["text"] or "")[:300],
        },
    }


@app.get("/history", summary="最近处理过的任务")
def history(limit: int = 20) -> dict[str, Any]:
    return {"ok": True, "data": {"items": STORE.list_recent(limit), "retention_hours": STORE.retention_hours}}


@app.get("/info", summary="服务信息与隐私状态")
def info() -> dict[str, Any]:
    return {
        "ok": True,
        "data": {
            "engine_used": active_engine(),          # 前端顶部提示条读这个字段
            "summary": provider_info(),
            "jobs": MANAGER.stats(),
            "store": STORE.stats(),
            "limits": {"max_mb": MAX_MB, "allowed_exts": sorted(ALLOWED_EXTS)},
            "privacy": {
                "retention_hours": STORE.retention_hours,
                "auto_delete": "超过保留期的任务记录与上传原文件会被自动删除",
                "network": "不调用任何外部服务；摘要模型只允许跑在本机",
            },
        },
    }


def _require_task(task_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", task_id or ""):
        raise ApiError(400, "bad_id", "任务 ID 格式不对")
    row = STORE.get(task_id)
    if row is None:
        raise ApiError(404, "task_not_found", "任务不存在，或已超过保留期被自动删除")
    return row


def _engine_of(row: dict[str, Any]) -> str:
    """任务真实使用的引擎；还没跑完就退回"当前可用引擎"。"""
    try:
        recorded = row["engine"]
    except (KeyError, IndexError):
        recorded = None
    return str(recorded) if recorded else active_engine()


# ------------------------------ 异常处理 ------------------------------

@app.exception_handler(ApiError)
async def on_api_error(_: Request, exc: ApiError) -> JSONResponse:
    log.warning("业务异常 %s: %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    detail = "; ".join(
        f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}"
        for err in exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content={"ok": False, "error": {"code": "invalid_request", "message": detail or "请求参数不合法"}},
    )


@app.exception_handler(Exception)
async def on_unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("未处理异常")
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": {"code": "internal_error", "message": "服务器内部错误，详见控制台日志"}},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
