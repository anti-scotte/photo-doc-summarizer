"""纯本地图文摘要系统 · 应用包。

模块划分：
    extract.py    文件 → 纯文本（OCR / PDF / DOCX / 纯文本）
    summarize.py  纯文本 → 五栏结构化摘要（抽取式 / 本机 Ollama / Mock）
    jobs.py       threading 后台任务队列
    store.py      SQLite 任务存储 + 24 小时保留期清理
    main.py       FastAPI 入口：路由、校验、异常处理
"""
