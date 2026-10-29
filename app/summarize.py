"""文字 → 结构化摘要（五栏：讲了什么 / 结论 / 关键数字 / 待办 / 存疑）。

三个引擎，同一套接口（Summarizer 协议）：

- ExtractiveSummarizer  纯离线，jieba 分词 + 词频打分 + MMR 选句。
                        五栏靠规则切分：数字用正则抠、待办/存疑看标记词、缺口写进存疑。
- OllamaSummarizer      调用**本机** Ollama（http://127.0.0.1:11434），让模型直接产出五栏。
                        代码里有硬闸：host 不是本机地址就直接拒绝，绝不上传外部服务。
- MockSummarizer        不连任何服务，输出结构固定、内容来自原文的假模型结果，用于无模型时演示。

对上层只暴露 summarize() 和 SummaryDoc。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

log = logging.getLogger("summarize")

# 送模型的字数上限：本地小模型上下文吃不下整本 PDF，超了截断并标注
MAX_MODEL_CHARS = 5000
SENTENCE_LIMIT = 6
SENTENCE_RATIO = 0.35          # 短文档别把整篇当摘要
KEYWORD_LIMIT = 8
MIN_SENT_LEN = 10
MAX_SENT_LEN = 200
DUPLICATE_LIMIT = 0.8
SECTION_ITEM_LIMIT = 4

# 只允许这些 host（比赛要求：纯本地，不上传外部服务）
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"


@dataclass
class SummaryDoc:
    """前端直接渲染的结构。五栏固定，前端不用猜字段。"""
    title: str = "未命名文档"
    tldr: str = ""
    what: list[str] = field(default_factory=list)          # 讲了什么
    conclusion: list[str] = field(default_factory=list)    # 结论
    numbers: list[str] = field(default_factory=list)       # 关键数字
    todos: list[str] = field(default_factory=list)         # 待办
    doubts: list[str] = field(default_factory=list)        # 存疑
    keywords: list[str] = field(default_factory=list)
    doc_type: str = "通用文档"
    provider: str = "extractive"
    chars: int = 0
    elapsed: float = 0.0
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Summarizer(Protocol):
    name: str

    def summarize(self, text: str) -> SummaryDoc: ...


# ============================== 安全闸 ==============================

def ensure_local(base_url: str) -> None:
    """模型服务必须在本机。这不是配置项，是比赛的硬约束。"""
    host = (urlparse(base_url).hostname or "").lower()
    if host not in LOCAL_HOSTS:
        raise RuntimeError(
            f"拒绝连接 {host or '未知地址'}：比赛要求纯本地处理，只允许 "
            f"{' / '.join(sorted(LOCAL_HOSTS))}。文本不会离开这台机器。"
        )


# ============================== 文本工具 ==============================

_SPLIT = re.compile(r"[。！？!?；;\r\n]+")
_CLAUSE = re.compile(r"[^，,；;。！？!?]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*|\d+(?:\.\d+)?%?")

# 数字：阿拉伯数字 + 中文数字。
# 中文数字必须带"第/百分之"前缀或量词后缀才认，否则"统一""一起"会被误抓。
_CN = "零○一二三四五六七八九十百千万亿两半"
_NUMBER = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|％|万元|亿元|万|亿|元|小时|分钟|秒|天|周|月|年|个|台|人|次|倍|页|项|条|吨|件|GB|MB|K)?"
    rf"|百分之[{_CN}十点]+"
    rf"|第[{_CN}十]+"
    rf"|[{_CN}十]+(?:个|台|人|次|倍|成|小时|天|周|月|年|项|条|份|步|轮|批)"
    r"|一半|半数|翻倍|翻番"
)
_CN_NUM_CHARS = re.compile(rf"[{_CN}]")

_STOP = set("的了和是在有我你他她它们这那也就都而与及或被把对从到为以于其因此所以如果但是")
_LEAD_STOP = "的了在从对为把被和与及而就都也"
_TAIL_STOP = "的了着过到和与及而就都也时后从向在对为把被以于"

CONCLUSION_MARKERS = (
    "结论", "因此", "所以", "可见", "表明", "说明", "综上", "总体", "意味着", "由此可见", "从而",
)

# 待办：谁在什么时候做什么。词表越宽，离线模式下漏掉的越少
TODO_MARKERS = (
    "需要", "须", "务必", "建议", "下一步", "计划", "将于", "负责", "要求", "待办",
    "请在", "请于", "应于", "应在", "应当在", "提交", "补齐", "报送", "落实",
    "安排", "截止", "限期", "推进", "整改", "复核", "下周一", "本周内", "月底前",
)

# 存疑分强弱两档：强信号先占位，弱信号（可能/也许/预计）只在有空位时补进来，
# 否则"预计增长 20%"这种正常结论句会被误判成存疑。
STRONG_DOUBT_MARKERS = (
    "待确认", "待核实", "待定", "尚未", "暂缺", "未明确", "存疑", "风险", "不确定",
    "尚不清楚", "有待", "暂未",
)
WEAK_DOUBT_MARKERS = (
    "可能", "也许", "预计", "或将", "是否", "估计", "大概", "需进一步",
)
DOUBT_MARKERS = STRONG_DOUBT_MARKERS + WEAK_DOUBT_MARKERS

DOC_TYPES = (
    ("合同", ("合同", "协议", "甲方", "乙方", "条款", "签署")),
    ("报告", ("报告", "复盘", "总结", "季度", "月度", "年度", "分析")),
    ("通知", ("通知", "公告", "须知", "安排", "请于")),
    ("论文", ("论文", "研究", "实验", "样本", "方法", "结论")),
    ("新闻", ("记者", "报道", "发布会", "讯")),
)


def _fallback_tokens(text: str) -> list[str]:
    """没有 jieba 时的降级分词：英文/数字按词切，中文切成二元组。

    不能直接把整段中文当一个 token——那样 7 字以上的句子会被长度过滤掉，关键词全空。
    """
    out: list[str] = []
    for run in _CJK.findall(text):
        if len(run) <= 4:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    out.extend(_LATIN.findall(text))
    return out


def tokenize(text: str) -> list[str]:
    tokens: Iterable[str]
    try:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        tokens = jieba.lcut(text)
    except Exception:                      # 没有 jieba 也能降级跑
        tokens = _fallback_tokens(text)

    out: list[str] = []
    for word in tokens:
        raw = word.strip()
        low = raw.lower()
        if len(low) < 2:
            continue
        if len(low) > 7 and not low.isascii():   # 中文长串多半是"在第三季度提升到"这类噪声
            continue
        if len(low) > 24 or low in _STOP:
            continue
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", low):
            continue
        out.append(raw)                          # 保留原样大小写，关键词里 OEE 不该变成 oee
    return out


def _split_sentences(text: str) -> list[str]:
    raw = [s.strip() for s in _SPLIT.split(re.sub(r"[ \t\u3000]+", " ", text))]
    kept = []
    for sentence in raw:
        if not (MIN_SENT_LEN <= len(sentence) <= MAX_SENT_LEN):
            continue
        if " | " in sentence:                    # Word 表格行，不当摘要句
            continue
        kept.append(sentence)
    return kept


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _guess_title(text: str, sentences: list[str]) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if 4 <= len(first_line) <= 40:
        return first_line
    if sentences:
        return sentences[0][:40]
    return first_line[:40] or "未命名文档"


def _guess_type(text: str) -> str:
    head = text[:1200]
    for name, words in DOC_TYPES:
        if any(word in head for word in words):
            return name
    return "通用文档"


def _is_clean(word: str) -> bool:
    return bool(word) and word[0] not in _LEAD_STOP and word[-1] not in _TAIL_STOP


def top_keywords(freq: Counter, limit: int = KEYWORD_LIMIT) -> list[str]:
    words = [w for w, _ in freq.most_common(300) if len(w) <= 12 and _is_clean(w)]
    return words[:limit]


def textrank_weights(text: str, top_k: int = 80) -> dict[str, float]:
    """用 jieba 的 TextRank 给**词**打权重，再用它给句子打分。

    jieba.analyse.textrank 输出的是词而不是句子，所以这里做一层转换：
    一句话由多少高权重词构成，它就越"核心"。相比裸词频，TextRank 靠共现图
    的中心性排序，能把"设备综合效率"这类真正的主语词排在"进行""相关"前面。
    没有 jieba 或 TextRank 异常时返回空字典，调用方自动退回词频。
    """
    try:
        from jieba import analyse

        pairs = analyse.textrank(
            text,
            topK=top_k,
            withWeight=True,
            allowPOS=("ns", "n", "vn", "v", "nr", "nt", "nz"),
        )
    except Exception:
        return {}

    weights: dict[str, float] = {}
    for item in pairs or []:
        try:
            word, weight = str(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if word and weight > 0:
            weights[word] = weight
    return weights


def pick_numbers(sentences: list[str], limit: int = 5) -> list[str]:
    """抽出带数字的小句，不做任何换算——每个片段都能在原文里原样找到。

    按标点切成小句再取，而不是按字符窗口截：窗口会把 "OEE" 切成 "EE"，
    也会让同一句抠出三段互相重叠的碎片。
    """
    out: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        taken = 0
        for clause in _CLAUSE.findall(sentence):
            clause = clause.strip()
            if not clause or len(clause) > 60:
                continue
            if not any(len(t.group(0).strip()) >= 2 for t in _NUMBER.finditer(clause)):
                continue
            if clause in seen:
                continue
            seen.add(clause)
            out.append(clause)
            taken += 1
            if len(out) >= limit:
                return out
            if taken >= 2:                       # 同一句最多取两段，别让一行数字占满
                break
    return out


def _match(sentences: list[str], markers: tuple[str, ...], limit: int) -> list[str]:
    hits = []
    for sentence in sentences:
        if any(marker in sentence for marker in markers):
            hits.append(sentence)
            if len(hits) >= limit:
                break
    return hits


# ============================== 引擎一：抽取式 ==============================

class ExtractiveSummarizer:
    """不依赖任何模型：足够快、每句话都能回查原文。"""

    name = "extractive"

    def summarize(self, text: str) -> SummaryDoc:
        sentences = _split_sentences(text)
        title = _guess_title(text, sentences)
        doc_type = _guess_type(text)

        if not sentences:
            body = text.strip()[:MAX_SENT_LEN]
            return SummaryDoc(
                title=title,
                tldr=body,
                what=[body] if body else [],
                numbers=pick_numbers([text]),
                keywords=top_keywords(Counter(tokenize(text))),
                doc_type=doc_type,
                provider=self.name,
                chars=len(text),
            )

        # 权重来源：优先 TextRank 词权重，拿不到再退回词频（TF 加权）
        joined = " ".join(sentences)
        freq = Counter(tokenize(joined))
        tr = textrank_weights(joined)
        if tr:
            floor = max(tr.values()) * 0.02          # 没进 TextRank 的词给个地板分，别让整句归零
            weights = {w: tr.get(w, 0.0) or floor * min(c, 5) / 5 for w, c in freq.items()}
        else:
            weights = {w: float(c) for w, c in freq.items()}

        scored: list[tuple[float, int, str, set[str]]] = []
        for index, sentence in enumerate(sentences):
            words = tokenize(sentence)
            if not words:
                continue
            score = sum(weights.get(w, 0.0) for w in words) / (len(words) ** 0.5)
            scored.append((score, index, sentence, set(words)))

        if not scored:                                  # 全是没有实词的口水句
            return SummaryDoc(
                title=title,
                tldr=sentences[0],
                what=sentences[:3],
                numbers=pick_numbers(sentences),
                keywords=[],
                doc_type=doc_type,
                provider=self.name,
                chars=len(text),
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        target = max(2, min(SENTENCE_LIMIT, int(len(sentences) * SENTENCE_RATIO + 0.5)))

        # MMR 思路：本身分越高越好，跟已选句子越像扣得越多
        picked: list[tuple[float, int, str, set[str]]] = []
        pool = list(scored)
        while pool and len(picked) < target:
            best_at, best_value = -1, float("-inf")
            for i, cand in enumerate(pool):
                sim = max((_jaccard(cand[3], kept[3]) for kept in picked), default=0.0)
                if sim > DUPLICATE_LIMIT:
                    continue
                value = cand[0] * (1.0 - sim)
                if value > best_value:
                    best_at, best_value = i, value
            if best_at < 0:
                break
            picked.append(pool.pop(best_at))

        picked.sort(key=lambda item: item[1])
        what = [item[2] for item in picked]

        # ---- 其余四栏：全部由正文规则切出来，不编内容 ----
        conclusion = [s for s in _match(sentences, CONCLUSION_MARKERS, 2) if s not in what]
        numbers = pick_numbers(sentences)
        todos = _match(sentences, TODO_MARKERS, SECTION_ITEM_LIMIT)

        # 存疑分档：强信号（待确认/尚未/风险）先占位，弱信号（可能/预计）有空位才补
        strong = [s for s in _match(sentences, STRONG_DOUBT_MARKERS, 2) if s not in todos]
        weak = [
            s
            for s in _match(sentences, WEAK_DOUBT_MARKERS, 2)
            if s not in todos and s not in strong
        ]
        doubts = strong + weak

        if not conclusion:
            doubts.append("正文里没有明确的结论性表述，判断需要结合原文再确认")
        if not numbers:
            doubts.append("正文里没有出现可核实的数字，涉及量化的说法建议回原文核对")
        if len(text) < 200:
            doubts.append("原文较短，摘要可能没有覆盖全部信息")

        # 关键词优先用 TextRank 排序，拿不到再退回词频
        if tr:
            keywords = [w for w in sorted(tr, key=lambda k: -tr[k]) if len(w) <= 12][:KEYWORD_LIMIT]
        else:
            keywords = top_keywords(freq)

        return SummaryDoc(
            title=title,
            tldr=scored[0][2],
            what=what,
            conclusion=conclusion[:3],
            numbers=numbers,
            todos=todos,
            doubts=doubts[:4],
            keywords=keywords,
            doc_type=doc_type,
            provider=self.name,
            chars=len(text),
        )


# ============================== 引擎二：本机 Ollama ==============================

SYSTEM_PROMPT = (
    "你是中文文档摘要助手。只输出一个 JSON 对象，不要解释，不要 Markdown 代码块。\n"
    "字段说明（除 title/tldr/doc_type 外都是字符串数组）：\n"
    '  "title": 不超过 20 字的标题；\n'
    '  "tldr": 不超过 60 字的一句话总结；\n'
    '  "what": 讲了什么，1-3 条，每条不超过 30 字；\n'
    '  "conclusion": 结论，1-3 条，每条不超过 30 字，没有就给空数组；\n'
    '  "numbers": 关键数字，0-5 条，必须原样抄自正文并带单位，不许换算或推算；\n'
    '  "todos": 待办，0-4 条，写清"谁在什么时候做什么"；\n'
    '  "doubts": 存疑，0-3 条，只写正文确实缺失或含糊的地方，没有就给空数组；\n'
    '  "keywords": 3-8 个关键词；\n'
    '  "doc_type": 只能是 合同/报告/通知/论文/新闻/通用文档 之一。\n'
    "硬规则：只依据正文，不许编造数字、人名或结论；数组宁可为空也不要凑数。"
)


def _repair_json(text: str) -> str:
    """模型被 max_tokens 截断时，把没闭合的引号和括号补上。

    只是尽力而为：补完还解析不了就抛错，由上一层降级到抽取式。
    """
    out: list[str] = []
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
            out.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                out.append(ch)
        else:
            out.append(ch)

    fixed = "".join(out).rstrip()
    if in_str:
        fixed += '"'
    fixed = re.sub(r"[,:]\s*$", "", fixed)
    while stack:
        fixed += stack.pop()
    return fixed


def _loads_json(raw: str) -> dict[str, Any]:
    """把模型输出解析成 dict：去代码围栏 → 补截断 → 多重兜底。

    绝不返回半成品：解析不出来就抛 ValueError，由 summarize() 降级，
    这样前端永远拿到的是合法五栏 JSON。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("模型没有返回内容")
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    candidates: list[str] = []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])   # 掐掉前后多余的说明文字
    candidates.append(text)                        # 原样再试一次

    for candidate in candidates:
        for attempt in (candidate, _repair_json(candidate)):
            try:
                data = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(data, dict):
                return data
    raise ValueError("模型返回的不是合法 JSON 对象")


def _as_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _as_list(value: Any, limit: int, item_limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        item = _as_text(item, item_limit)
        if item:
            out.append(item)
    return out[:limit]


def _list_models(base_url: str, timeout: float = 1.0) -> list[str]:
    """问一下本机 Ollama 装了哪些模型。"""
    try:
        with urlopen(f"{base_url}/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return [str(m.get("name")) for m in data.get("models", []) if m.get("name")]


# 通用对话模型优先；coder / 代码模型垫底（写 JSON 没问题，中文归纳偏弱）
MODEL_PREFERENCE = (
    "qwen2.5:7b-instruct", "qwen2.5:7b", "qwen2.5:3b-instruct", "qwen2.5:3b",
    "qwen2.5:1.5b-instruct", "qwen2.5:1.5b", "qwen2.5",
    "qwen2", "qwen", "llama3", "llama", "glm4", "glm", "gemma", "mistral", "phi",
)


def _model_rank(name: str) -> tuple[int, int, str]:
    low = name.lower()
    rank = len(MODEL_PREFERENCE) + 1
    for index, pref in enumerate(MODEL_PREFERENCE):
        if low == pref or low.startswith(pref):
            rank = index
            break
    return (rank + (20 if "coder" in low else 0), len(name), low)


def rank_models(names: list[str]) -> list[str]:
    return sorted(names, key=_model_rank)


# /info、/upload、/status 都会问"现在用哪个引擎"。每次真探测一遍 Ollama 要近 1 秒，
# 这里缓存 15 秒：一次页面加载 + 紧随其后的上传，只付一次探测成本。
_ACTIVE_TTL = 15.0
_active_cache: dict[str, Any] = {"ts": 0.0, "name": "extractive"}


def active_engine(provider: str | None = None) -> str:
    """当前会用的引擎名：ollama / extractive / mock。永不抛异常。"""
    now = time.time()
    if provider is None and _active_cache["ts"] and now - _active_cache["ts"] < _ACTIVE_TTL:
        return str(_active_cache["name"])
    try:
        name = get_summarizer(provider).name
    except Exception:                      # 配置写错、探测失败都算"用不了大模型"
        name = "extractive"
    if provider is None:
        _active_cache["ts"] = now
        _active_cache["name"] = name
    return name


class OllamaSummarizer:
    """调用本机 Ollama。/api/chat + format=json，让模型直接产出五栏。"""

    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 180.0) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        ensure_local(self.base_url)                    # 安全闸：非本机直接拒绝
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", timeout) or timeout)

        installed = _list_models(self.base_url)
        explicit = model or os.getenv("OLLAMA_MODEL")
        if explicit:
            # 指定了就先用它，其余已装模型排队兜底
            self.models = [explicit] + [m for m in rank_models(installed) if m != explicit]
        else:
            self.models = rank_models(installed) or [DEFAULT_MODEL]
        self.model = self.models[0]

    def _post_chat(self, model: str, body: str) -> dict[str, Any]:
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_ctx": 8192},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"以下是待摘要素材：\n\n{body}"},
            ],
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _explain(self, exc: HTTPError, model: str) -> RuntimeError:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "ignore")[:160]
        except Exception:
            pass
        hint = ""
        if exc.code == 404:
            available = _list_models(self.base_url)
            hint = (
                f"；本机已有模型：{', '.join(available)}" if available else "；本机还没有拉取任何模型"
            ) + f"。先执行 ollama pull {model}，或用 OLLAMA_MODEL 指定已装的模型"
        return RuntimeError(f"Ollama 返回 {exc.code} {detail or ''}{hint}".strip())

    def summarize(self, text: str) -> SummaryDoc:
        body = text[:MAX_MODEL_CHARS]
        warning = "" if text == body else f"正文超过 {MAX_MODEL_CHARS} 字，只送了前 {MAX_MODEL_CHARS} 字给模型"

        raw: dict[str, Any] | None = None
        used_model = self.model
        last_error: Exception | None = None

        # 候选模型逐个试：模型不存在（404）就换下一个，超时/连接失败则直接上抛给上层降级
        for model in self.models:
            try:
                raw = self._post_chat(model, body)
                used_model = model
                break
            except HTTPError as exc:
                if exc.code == 404 and len(self.models) > 1 and model != self.models[-1]:
                    log.info("本机没有模型 %s，换下一个候选", model)
                    last_error = exc
                    continue
                raise self._explain(exc, model) from exc

        if raw is None:
            raise RuntimeError(f"没有可用的模型（已尝试 {', '.join(self.models)}）") from last_error

        content = (raw.get("message") or {}).get("content", "")
        data = _loads_json(content)

        return SummaryDoc(
            title=_as_text(data.get("title"), 40) or _guess_title(text, _split_sentences(text)),
            tldr=_as_text(data.get("tldr"), 120),
            what=_as_list(data.get("what"), 3, 60),
            conclusion=_as_list(data.get("conclusion"), 3, 60),
            numbers=_as_list(data.get("numbers"), 5, 60),
            todos=_as_list(data.get("todos"), SECTION_ITEM_LIMIT, 60),
            doubts=_as_list(data.get("doubts"), 3, 60),
            keywords=_as_list(data.get("keywords"), KEYWORD_LIMIT + 2, 24),
            doc_type=_as_text(data.get("doc_type"), 10) or "通用文档",
            provider=f"{self.name}:{used_model}",
            chars=len(text),
            warning=warning,
        )


# ============================== 引擎三：Mock ==============================

class MockSummarizer:
    """不连服务、不编内容：用抽取式结果套上模型输出的结构，用来演示"模型已接入"。

    刻意不返回虚构的人名/数字——答辩时被问"这句哪来的"，还能指回原文。
    """

    name = "mock"

    def __init__(self, latency: float | None = None) -> None:
        if latency is None:
            latency = float(os.getenv("MOCK_LATENCY", "0.6"))
        self.latency = max(0.0, latency)

    def summarize(self, text: str) -> SummaryDoc:
        if self.latency:
            time.sleep(self.latency)           # 模拟模型推理耗时，方便看进度条

        doc = ExtractiveSummarizer().summarize(text)
        doc.provider = self.name
        doc.warning = "当前是 Mock 引擎：结构同正式模型，内容由抽取式生成，未调用任何模型"
        return doc


# ============================== 工厂 ==============================

def _ollama_alive(base_url: str, timeout: float = 0.8) -> bool:
    """本机 Ollama 且至少装了一个模型，才算真的可用。"""
    if not ensure_local_ok(base_url):
        return False
    return bool(_list_models(base_url, timeout))


def get_summarizer(provider: str | None = None) -> Summarizer:
    """SUMMARY_PROVIDER = auto | extractive | ollama | mock（默认 auto）。"""
    choice = (provider or os.getenv("SUMMARY_PROVIDER", "auto") or "auto").strip().lower()

    if choice == "extractive":
        return ExtractiveSummarizer()
    if choice == "mock":
        return MockSummarizer()

    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    if choice in {"ollama", "auto", ""}:
        ensure_local(base_url)                                   # auto 也要过安全闸
        if choice == "ollama" or _ollama_alive(base_url):
            return OllamaSummarizer(base_url=base_url)
        return ExtractiveSummarizer()

    log.warning("未知的 SUMMARY_PROVIDER=%s，回退抽取式", choice)
    return ExtractiveSummarizer()


def summarize(text: str, provider: str | None = None) -> SummaryDoc:
    """入口。**永远返回一个结果**，任何故障都降级到本地抽取式。

    这里的 try 故意放得很宽：连"引擎构造不出来"（比如 OLLAMA_BASE_URL 指到
    非本机地址、被安全闸拒绝）也必须降级，不能把异常抛给任务队列。
    评委电脑上没装 Ollama，走的就是这条路。
    """
    started = time.perf_counter()
    warning = ""
    engine: Summarizer | None = None

    try:
        engine = get_summarizer(provider)      # 可能在探测 / 安全闸这一步就失败
        doc = engine.summarize(text)
    except Exception as exc:                   # noqa: BLE001 —— 兜底就是它的职责
        name = getattr(engine, "name", None) or provider or "auto"
        if name in ("auto", ""):
            name = "本机大模型"
        log.warning("摘要引擎 %s 不可用，已回退本地轻量提取引擎：%s", name, exc)
        warning = f"{name} 不可用，已回退本地轻量提取引擎：{exc}"
        try:
            doc = ExtractiveSummarizer().summarize(text)
        except Exception:                      # 连兜底都挂了，也绝不能崩
            log.exception("抽取式引擎异常，返回最小可用结果")
            head = text.strip()[:MAX_SENT_LEN]
            doc = SummaryDoc(
                title=_guess_title(text, []),
                tldr=head,
                what=[head] if head else [],
                provider="fallback",
                chars=len(text),
            )
        doc.warning = warning

    doc.elapsed = round(time.perf_counter() - started, 2)
    return doc


def ensure_local_ok(base_url: str) -> bool:
    try:
        ensure_local(base_url)
        return True
    except RuntimeError:
        return False


def provider_info() -> dict[str, Any]:
    """给前端显示"现在用的是哪个引擎"，隐私声明要能被验证。"""
    choice = (os.getenv("SUMMARY_PROVIDER", "auto") or "auto").lower()
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    local_ok = ensure_local_ok(base_url)

    try:
        engine = get_summarizer(choice)            # 只探测这一次
        active = engine.name
        model = getattr(engine, "model", "") or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
    except Exception:                              # 配置写错也不能让 /info 挂掉
        active, model = "extractive", DEFAULT_MODEL

    return {
        "provider_setting": choice,
        "active": active,
        "engine_used": active,
        "ollama_base_url": base_url if local_ok else "(已阻止：非本机地址)",
        "ollama_model": model,
        "ollama_reachable": active == "ollama",
        "network_policy": "只允许本机模型服务，文本不出机器",
    }
