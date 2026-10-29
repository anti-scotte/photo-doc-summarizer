"""文件 → 纯文本。

- 图片（jpg/png/webp/bmp/tif）：Pillow 解码 + EXIF 纠正 → RapidOCR 识别
- PDF：优先 PyMuPDF（能抽文本层，也能把页面渲染成图做 OCR）
       没有 PyMuPDF 时退到 pypdf（只能抽文本层，扫描件会明确报错）
- Word（.docx）：python-docx，正文 + 表格一起取
- txt/md：标准库直读，编码兜底

对上层只暴露两个东西：extract() 和 ExtractError。
"""

from __future__ import annotations

import io
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("extract")

Progress = Callable[[int, str], None]

# 支持的文件类型
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DOC_EXTS = {".pdf", ".docx"}
TEXT_EXTS = {".txt", ".md"}
ALLOWED_EXTS = IMAGE_EXTS | DOC_EXTS | TEXT_EXTS

# 文件头魔数：拦住"扩展名对、内容不对"的假文件
SIGNATURES = {
    ".pdf": (b"%PDF", "PDF"),
    ".docx": (b"PK", "DOCX"),
    ".png": (b"\x89PNG", "PNG"),
    ".jpg": (b"\xff\xd8", "JPEG"),
    ".jpeg": (b"\xff\xd8", "JPEG"),
    ".tif": (b"II*\x00", "TIFF"),
    ".tiff": (b"II*\x00", "TIFF"),
}

OCR_DPI = 200                 # 再高对识别率提升有限，耗时翻倍
SCAN_CHARS_PER_PAGE = 20      # 平均每页少于这么多字 → 判定为扫描件
MAX_PAGES = 200               # 文档页数硬上限
OCR_PAGE_LIMIT = 50           # 扫描件最多 OCR 这么多页，其余跳过


class ExtractError(Exception):
    """抽取阶段的失败。kind 供上层映射 HTTP 状态码。"""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass
class Extracted:
    text: str
    kind: str                 # text / docx / image / pdf-text / pdf-ocr
    pages: int = 0
    used_ocr: bool = False

    @property
    def chars(self) -> int:
        return len(re.sub(r"\s+", "", self.text))


def sniff(head: bytes, ext: str) -> None:
    """上传落盘后立刻校验文件头，早失败比晚失败好。"""
    sig = SIGNATURES.get(ext)
    if sig and head and not head.startswith(sig[0]):
        raise ExtractError("unsupported", f"文件内容不像 {sig[1]}，请确认扩展名是否正确")


def extract(path: Path, on_progress: Progress | None = None) -> Extracted:
    """按扩展名分发。任何失败都抛 ExtractError，不泄露底层异常类型给前端。"""
    if not path.is_file():
        raise ExtractError("bad_file", "文件不存在或已被清理")

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ExtractError("unsupported", f"不支持 {ext or '无扩展名'}；可用类型：{' '.join(sorted(ALLOWED_EXTS))}")

    if ext in TEXT_EXTS:
        return Extracted(_tidy(_read_plain(path)), "text")
    if ext == ".docx":
        return Extracted(_tidy(_read_docx(path)), "docx")
    if ext in IMAGE_EXTS:
        if on_progress:
            on_progress(20, "正在识别图片文字")
        return Extracted(_tidy(_read_image(path)), "image", pages=1, used_ocr=True)
    return _read_pdf(path, on_progress)


# ------------------------------ 纯文本 ------------------------------

def _read_plain(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractError("bad_file", "文件编码无法识别，请另存为 UTF-8 后重试")


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# ------------------------------ Word ------------------------------

def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ExtractError("missing_dep", "缺少 python-docx，请重新运行 run.bat") from exc

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise ExtractError("bad_file", f"这个 .docx 打不开：{exc}") from exc

    lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]

    # 表格不能丢：按行拼回文本，单元格之间用 | 分隔
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
    return "\n".join(lines)


# ------------------------------ 图片 ------------------------------

def _read_image(path: Path) -> str:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ExtractError("missing_dep", "缺少 pillow，请重新运行 run.bat") from exc

    try:
        with Image.open(str(path)) as img:
            img = ImageOps.exif_transpose(img)   # 手机竖拍不纠正，OCR 出来全是乱码
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            png = buffer.getvalue()
    except Exception as exc:
        raise ExtractError("bad_file", f"这张图打不开：{exc}") from exc
    return "\n".join(ocr(png))


# ------------------------------ PDF ------------------------------

def _read_pdf(path: Path, on_progress: Progress | None) -> Extracted:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _pdf_with_pypdf(path, on_progress)
    return _pdf_with_fitz(fitz, path, on_progress)


def _pdf_with_fitz(fitz: Any, path: Path, on_progress: Progress | None) -> Extracted:
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise ExtractError("bad_file", f"这个 PDF 打不开：{exc}") from exc

    with doc:
        pages = doc.page_count
        if pages <= 0:
            raise ExtractError("bad_file", "PDF 里没有页面")
        if pages > MAX_PAGES:
            raise ExtractError("unsupported", f"超过 {MAX_PAGES} 页，请拆分后再上传")

        text = "\n".join((page.get_text("text") or "") for page in doc)

        # 有文本层就直接用，别白白跑一遍 OCR
        if len(re.sub(r"\s+", "", text)) >= SCAN_CHARS_PER_PAGE * pages:
            return Extracted(_tidy(text), "pdf-text", pages=pages)

        if on_progress:
            on_progress(15, f"没有文本层（{pages} 页），改用 OCR")

        todo = min(pages, OCR_PAGE_LIMIT)
        lines: list[str] = []
        for index, page in enumerate(doc, 1):
            if index > todo:
                break
            png = page.get_pixmap(dpi=OCR_DPI).tobytes("png")
            lines.extend(ocr(png))
            if on_progress and (index == todo or index % 3 == 0):
                on_progress(15 + int(50 * index / todo), f"OCR 第 {index}/{todo} 页")

        if pages > OCR_PAGE_LIMIT:
            lines.append(f"（本 PDF 共 {pages} 页，仅识别前 {OCR_PAGE_LIMIT} 页）")
    return Extracted(_tidy("\n".join(lines)), "pdf-ocr", pages=pages, used_ocr=True)


def _pdf_with_pypdf(path: Path, on_progress: Progress | None) -> Extracted:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractError("missing_dep", "缺少 pymupdf（也没有 pypdf），请重新运行 run.bat") from exc

    if on_progress:
        on_progress(15, "读取 PDF 文本层")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ExtractError("bad_file", f"这个 PDF 打不开：{exc}") from exc

    pages = len(reader.pages)
    if pages > MAX_PAGES:
        raise ExtractError("unsupported", f"超过 {MAX_PAGES} 页，请拆分后再上传")

    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not re.sub(r"\s+", "", text):
        raise ExtractError(
            "empty",
            "这个 PDF 是扫描件（没有文本层），pypdf 无法把它转成图片；请安装 pymupdf 后重试",
        )
    return Extracted(_tidy(text), "pdf-text", pages=pages)


# ------------------------------ OCR ------------------------------

_ENGINE: Any = None
_ENGINE_LOCK = threading.Lock()


def _get_engine() -> Any:
    """OCR 模型只加载一次，多线程共用；第一次约 1-2 秒。"""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise ExtractError("missing_dep", "缺少 rapidocr-onnxruntime，请重新运行 run.bat") from exc
            log.info("加载 OCR 模型（首次约 1-2 秒）")
            _ENGINE = RapidOCR()
    return _ENGINE


def ocr(source: bytes) -> list[str]:
    """识别一段图片字节，返回逐行文本。"""
    engine = _get_engine()
    try:
        result, _ = engine(source)
    except Exception as exc:
        raise ExtractError("bad_file", f"OCR 识别失败：{exc}") from exc

    if result is None:
        return []

    # 新版 rapidocr 返回带 .txts 的对象
    if hasattr(result, "txts"):
        return [str(t).strip() for t in result.txts if str(t).strip()]

    lines: list[str] = []
    for item in result:
        text = ""
        if isinstance(item, dict):
            text = str(item.get("txt") or item.get("text") or "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[1])
        text = text.strip()
        if text:
            lines.append(text)
    return lines
