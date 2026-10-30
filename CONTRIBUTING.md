# 贡献指南 · Contributing

感谢愿意改进这个项目！下面是最小但够用的约定。

- [报告问题](#报告问题)
- [提交代码](#提交代码)
- [代码规范](#代码规范)
- [开发环境](#开发环境)
- [提交信息](#提交信息)

---

## 报告问题

提 Issue 前请先搜一下有没有重复。新开 Issue 时请用模板：

- **Bug** → [bug_report](https://github.com/anti-scotte/photo-doc-summarizer/issues/new?template=bug_report.md)
- **新功能** → [feature_request](https://github.com/anti-scotte/photo-doc-summarizer/issues/new?template=feature_request.md)

一个能复现的 Bug 报告至少要有：**操作系统 + Python 版本 + 用的是哪个引擎（提取式还是大模型）+ 完整报错**。
如果和文件解析有关，请说明文件类型和大致页数，**不要上传含个人信息的真实文档**（可以用 `samples/` 里的样例复现）。

## 提交代码

1. Fork 仓库，从 `main` 拉一个分支：`git checkout -b fix/pdf-encoding`
2. 改代码，本地跑通（见下方"开发环境"）
3. 提交前自查：
   ```bash
   python -m compileall app     # 语法检查
   ```
4. 推到你自己的 fork，向 `main` 提 PR，并按 PR 模板填写

PR 请保持**一件事一个 PR**。顺手修了无关的格式问题会让 review 变慢。

## 代码规范

- 遵循 **PEP 8**，4 空格缩进，行宽尽量 ≤ 120
- **必须写类型标注**（本项目全部使用 `from __future__ import annotations` 风格）
- **函数要有中文 docstring**，说明"为什么"而不是"是什么"
- 不要引入重型依赖；新增依赖前请先开 Issue 讨论
- **禁止引入 GPL / AGPL 依赖**（本项目是 MIT，AGPL 会传染）
- 前端保持**零框架**：只用原生 HTML / CSS / JS，不引入构建工具

分层约定（改动请守住边界）：

| 文件 | 职责 | 不该做的事 |
|---|---|---|
| `app/extract.py` | 文件 → 纯文本 | 不碰数据库、不碰 HTTP |
| `app/summarize.py` | 文本 → 五栏 | 不读文件、不写库 |
| `app/jobs.py` | 线程队列 | 不直接操作 SQLite |
| `app/store.py` | SQLite + 清理 | 不做业务判断 |
| `app/main.py` | HTTP 路由 | 不放业务逻辑 |

## 开发环境

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

想验证"没有大模型也能跑"这条路径，设一个环境变量即可：

```bat
set SUMMARY_PROVIDER=extractive
```

## 提交信息

参考 [Conventional Commits](https://www.conventionalcommits.org/)：

```
feat: 增加 PDF 分栏识别
fix: 修复 pypdf 抽中文乱码时未回退的问题
docs: 补充部署说明
refactor: 拆分 extract.py 的 PDF 分支
```

---

提交 PR 即表示你同意以本项目的 [MIT License](LICENSE) 授权你的贡献。
