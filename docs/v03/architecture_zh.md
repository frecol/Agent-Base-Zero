# v0.3 更新详解：会话持久化、长期记忆与上下文压缩

v0.3 在 v0.2 的"对话循环 + 流式输出 + 工具调用"基础上，新增了六大能力：**会话持久化**、**会话恢复**、**长期记忆工具**、**记忆注入**、**Token 估算与进度展示**、**上下文压缩**。所有数据通过纯文件存储（JSON），不引入数据库。

---

## 目录

- [变更总览](#变更总览)
- [新增能力一：会话持久化](#新增能力一会话持久化)
  - [Session ID 生成](#session-id-生成)
  - [存储结构](#存储结构)
  - [原子写入](#原子写入)
- [新增能力二：会话恢复](#新增能力二会话恢复)
  - [/sessions 命令](#sessions-命令)
  - [/resume 命令](#resume-命令)
  - [/new 命令](#new-命令)
- [新增能力三：长期记忆工具](#新增能力三长期记忆工具)
  - [记忆存储结构](#记忆存储结构)
  - [memory_save 工具](#memory_save-工具)
  - [session_search 工具](#session_search-工具)
  - [记忆只在启动时加载](#记忆只在启动时加载)
- [新增能力四：记忆注入](#新增能力四记忆注入)
- [新增能力五：Token 估算与进度展示](#新增能力五token-估算与进度展示)
  - [chars/4 估算算法](#chars4-估算算法)
  - [/status 命令增强](#status-命令增强)
- [新增能力六：上下文压缩](#新增能力六上下文压缩)
  - [自动压缩触发](#自动压缩触发)
  - [压缩算法](#压缩算法)
  - [/compact 手动压缩](#compact-手动压缩)
  - [压缩 Prompt 设计](#压缩-prompt-设计)
- [新增工具](#新增工具)
  - [grep_search — 文本搜索](#grep_search--文本搜索)
  - [fetch_url — 网页抓取](#fetch_url--网页抓取)
  - [tree — 目录树展示](#tree--目录树展示)
  - [find_file — 文件查找](#find_file--文件查找)
  - [file_delete — 文件删除](#file_delete--文件删除)
  - [system_info — 系统信息](#system_info--系统信息)
- [核心集成：\_post_turn 钩子](#核心集成_post_turn-钩子)
- [配置变更](#配置变更)
- [CLI 新增命令](#cli-新增命令)
- [文件变更清单](#文件变更清单)

---

## 变更总览

| 维度 | v0.2 | v0.3 |
|------|------|------|
| 会话 | 内存中，退出即丢失 | 持久化到 `.genesis/sessions/` |
| 记忆 | 无 | 跨会话长期记忆，`.genesis/memory/memory.json` |
| 上下文管理 | 无限制 | 64000 token 上限，自动压缩 |
| CLI 命令 | 7 个 | 新增 4 个（`/sessions`, `/resume`, `/new`, `/compact`） |
| /status | 基本信息 | 显示 Token 用量、Session ID |
| 新增文件 | 0 | 3 个（`session.py`, `memory.py`, `tokens.py`） |

---

## 新增能力一：会话持久化

### Session ID 生成

每次创建 Agent 实例时自动生成唯一 Session ID：

```python
# agent/session.py
def generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = secrets.token_hex(3)  # 6 hex chars
    return f"{timestamp}_{random_part}"
    # 例: "20260427_143025_a3f8b2"
```

### 存储结构

会话文件存储在 `.genesis/sessions/{session_id}.json`：

```json
{
  "session_id": "20260427_143025_a3f8b2",
  "created_at": "2026-04-27T14:30:25",
  "updated_at": "2026-04-27T15:12:03",
  "title": "User asked about Python decorators",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

- `title` 从第一条 user 消息截取前 60 字符自动生成
- `created_at` 首次保存时设定，后续不更新
- `updated_at` 每次保存都更新

### 原子写入

使用"先写临时文件再 rename"的策略，避免崩溃导致数据丢失：

```python
tmp_path = file_path.with_suffix(".tmp")
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
os.replace(tmp_path, file_path)  # 原子操作
```

---

## 新增能力二：会话恢复

### /sessions 命令

列出所有保存的会话，按 `updated_at` 倒序排列：

```
  20260427_143025_a3f8b2  2026-04-27T15:12:03  User asked about Python decorators  (12 msgs)
  20260426_091500_f1e2d3  2026-04-26T09:15:00  Help with git rebase                 (8 msgs)
```

### /resume 命令

恢复指定会话，加载历史消息继续对话：

```
/resume 20260427_143025_a3f8b2
→ Resumed session 20260427_143025_a3f8b2
→ 12 messages loaded
```

### /new 命令

新建会话，生成新的 session_id，清空消息列表（长期记忆保留）：

```
/new
→ New session started: 20260427_160000_b4c5d6
→ Previous session 20260427_143025_a3f8b2 saved
```

---

## 新增能力三：长期记忆工具

### 记忆存储结构

记忆文件存储在 `.genesis/memory/memory.json`：

```json
{
  "entries": [
    {
      "id": "mem_a3f8b2c1",
      "content": "User prefers Python over JavaScript for scripting",
      "created_at": "2026-04-27T14:35:00",
      "updated_at": "2026-04-27T14:35:00"
    }
  ]
}
```

记忆跨会话共享 — 所有会话读写同一个 `memory.json` 文件。

### memory_save 工具

记忆不再是每轮自动提取，而是作为工具（`memory_save`）由 LLM 自主决定何时保存：

```python
# tools/memory_save.py
SCHEMA = {
    "name": "memory_save",
    "description": "Save a durable fact to persistent long-term memory...",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to save..."}
        }
    }
}
```

**System Prompt 中的指引**:
- 保存：用户偏好、环境细节、工具特性、稳定约定
- 不保存：任务进度、会话结果、临时状态（用 `session_search` 回忆）
- 优先保存能减少未来用户纠正/提醒的信息

**设计理由**: 让 LLM 自己决定何时记忆，比每轮强制提取更精确、更节省 token。

### session_search 工具

LLM 可以通过 `session_search` 工具搜索过去的对话记录：

```python
# tools/session_search.py
SCHEMA = {
    "name": "session_search",
    "description": "Search past conversation sessions by keyword...",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword..."},
            "limit": {"type": "integer", "description": "Max results. Default: 10"}
        }
    }
}
```

搜索逻辑：遍历 `.genesis/sessions/` 下所有会话文件，对每条消息的 content 做关键词匹配，返回匹配的消息摘要和所属会话 ID。

### 记忆只在启动时加载

```python
# agent/agent.py
def load_memory(self):
    entries = load_mem()
    self._memory_text = format_memory_for_prompt(entries)
```

`_memory_text` 只在会话启动时加载一次，中途不再更新。这样做是为了：

1. **保持 prompt 缓存命中** — system prompt 部分不变，LLM 提供商（如 DeepSeek）可以复用已缓存的 prefix，减少费用
2. **保持会话稳定** — 避免 mid-session 记忆变化导致 LLM 行为不一致
3. **新记忆下个会话生效** — LLM 通过 `memory_save` 保存的记忆在下次启动时加载

---

## 新增能力四：记忆注入

新会话启动时，`agent.load_memory()` 加载长期记忆并缓存为文本：

```python
# agent/agent.py
def load_memory(self):
    entries = load_mem()
    self._memory_text = format_memory_for_prompt(entries)
```

在 `_build_messages()` 中作为第二个 system 消息注入：

```python
result = [{"role": "system", "content": self.system_prompt}]
if self._memory_text:
    result.append({"role": "system", "content": self._memory_text})
# ... conversation messages
```

格式化后的记忆文本：

```
[Long-term Memory - Information about the user from past conversations]
- User prefers Python over JavaScript for scripting
- User prefers concise explanations
- User's timezone is Asia/Shanghai
```

记忆始终开启，无需配置。

---

## 新增能力五：Token 估算与进度展示

### chars/4 估算算法

采用简单的 chars/4 启发式估算，与 hermes-agent 一致：

```python
_CHARS_PER_TOKEN = 4

def estimate_tokens(messages: list[dict]) -> int:
    total_chars = 0
    for msg in messages:
        for value in msg.values():
            if isinstance(value, str):
                total_chars += len(value)
            elif isinstance(value, list):  # tool_calls
                # 递归计算嵌套结构
    return (total_chars + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN
```

遍历所有 message 的字符串字段（content、role、tool_calls 等），求和后除以 4。

### /status 命令增强

```
/status
  Streaming:    on
  Thinking:     on
  Model:        deepseek-v4-flash
  Session:      20260427_143025_a3f8b2
  Messages:     12
  Token usage:  ~3,200 / 64,000 (5%)
```

---

## 新增能力六：上下文压缩

### 自动压缩触发

当 token 用量达到 `max_context_tokens * compression_threshold`（默认 64000 × 95% = 60800）时自动触发。

### 压缩算法

```
原始消息列表: [msg0, msg1, msg2, ..., msgN-21, msgN-20, ..., msgN]

分割:
  head  = msgs[:3]            ← 保留前 3 条
  middle = msgs[3:-20]        ← 压缩中间部分
  tail  = msgs[-20:]          ← 保留后 20 条

压缩中: LLM 将 middle 生成结构化 summary
压缩后: [head... + summary_msg + tail...]
```

### /compact 手动压缩

用户可通过 `/compact` 主动触发压缩：

- 先检查消息总数是否 > `head_keep + tail_keep`（默认 23）
- 不满足条件则拒绝压缩并提示
- 满足条件则执行压缩，显示压缩前后对比：

```
/compact
→ Compressed: 45 -> 24 messages, tokens ~58,000 -> ~12,000
```

### 压缩 Prompt 设计

使用结构化模板，确保不丢失关键信息：

```
## Goal           — 用户目标（1-2 句）
## Key Actions    — 已执行的操作列表
## Current State  — 当前进展
## Decisions      — 重要技术决策
## Technical Details — 需要精确保留的值
## User Preferences — 用户表达的偏好
```

---

## 新增工具

v0.3 新增 6 个实用工具，补齐代码搜索、网页读取、项目结构浏览、文件查找、文件删除和系统信息等能力。所有工具遵循相同的 `SCHEMA + handler + registry.register()` 模式，放入 `tools/` 目录即自动注册。

### grep_search — 文本搜索

按关键词或正则表达式递归搜索项目中的文件内容，替代 LLM 手动拼接 `grep` 命令。

```python
# tools/grep_search.py
SCHEMA = {
    "name": "grep_search",
    "description": "Search file contents by keyword or regex pattern...",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "搜索模式（支持正则）"},
            "path": {"type": "string", "description": "搜索目录，默认当前目录"},
            "file_pattern": {"type": "string", "description": "文件过滤 glob（如 '*.py'）"},
            "case_insensitive": {"type": "boolean", "description": "是否忽略大小写"},
            "max_results": {"type": "integer", "description": "最大结果数，默认 50"},
        },
        "required": ["pattern"],
    },
}
```

**设计要点：**
- 使用 Python `re` 模块实现正则匹配，无需外部依赖
- 自动跳过 `.git`、`__pycache__`、`node_modules`、`.genesis`、`.venv` 等目录
- 通过前 8KB 采样检测二进制文件并跳过
- 返回匹配行列表，每条包含 `file`、`line_number`、`line`

### fetch_url — 网页抓取

获取指定 URL 的网页内容并提取纯文本。与 `web_search` 配合使用：先搜索找到 URL，再抓取读取全文。

```python
# tools/fetch_url.py
SCHEMA = {
    "name": "fetch_url",
    "description": "Fetch a web page and return its text content...",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的 URL"},
            "max_length": {"type": "integer", "description": "最大返回字符数，默认 10000"},
        },
        "required": ["url"],
    },
}
```

**设计要点：**
- 使用 `urllib.request` 发送请求，设置合理 User-Agent
- 使用 BeautifulSoup（需 `beautifulsoup4` 依赖）剥离 HTML 标签，提取可读纯文本
- 自动去除 `<script>`、`<style>`、`<nav>`、`<footer>` 等非内容标签
- 读取上限 1MB，超出截断并标记 `truncated: true`
- 非 HTML 内容（如纯文本）直接解码返回

### tree — 目录树展示

递归展示项目目录结构，类似 `tree` 命令，比多次调用 `list_dir` 更高效。

```python
# tools/tree.py
SCHEMA = {
    "name": "tree",
    "description": "Display the directory tree structure recursively...",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "根目录路径，默认当前目录"},
            "max_depth": {"type": "integer", "description": "最大遍历深度，默认 3"},
        },
        "required": [],
    },
}
```

**设计要点：**
- 使用 `├──` / `└──` / `│  ` 字符绘制树形结构
- 自动跳过 `.git`、`__pycache__`、`node_modules`、`.genesis`、`.venv` 等目录和隐藏文件
- 总条目数上限 500，超出截断并标记 `truncated: true`
- 目录后缀加 `/` 标识

### find_file — 文件查找

按文件名 glob 模式在整个项目中递归搜索文件。

```python
# tools/find_file.py
SCHEMA = {
    "name": "find_file",
    "description": "Find files by name pattern (glob) in a directory tree...",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob 模式（如 '*.py', '**/test_*.json'）"},
            "path": {"type": "string", "description": "搜索目录，默认当前目录"},
        },
        "required": ["pattern"],
    },
}
```

**设计要点：**
- 使用 `pathlib.Path.rglob()` 递归匹配
- 自动跳过 `.git`、`__pycache__`、`.genesis` 等目录
- 返回匹配文件路径列表和总数

### file_delete — 文件删除

删除指定路径的文件，补齐文件操作能力（已有读、写、编辑）。

```python
# tools/file_delete.py
SCHEMA = {
    "name": "file_delete",
    "description": "Delete a file at the given path...",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要删除的文件路径"},
        },
        "required": ["path"],
    },
}
```

**设计要点：**
- 仅允许删除普通文件，不允许删除目录
- 删除前检查文件存在性，权限不足时返回明确错误

### system_info — 系统信息

获取当前运行环境的关键信息，帮助 LLM 做出合理决策。

```python
# tools/system_info.py
SCHEMA = {
    "name": "system_info",
    "description": "Get current system information including OS, CPU, memory, and disk...",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}
```

**设计要点：**
- 返回操作系统、版本、主机名、Python 版本、CPU 核数、架构、工作目录
- 尽力获取磁盘空间信息（`shutil.disk_usage`），失败则跳过
- 纯标准库实现，无外部依赖

---

## 核心集成：\_post_turn 钩子

`_post_turn()` 是 v0.3 的核心集成点，在每轮对话结束后自动执行：

```python
def _post_turn(self):
    # 1. 压缩检查 — token 超阈值时自动压缩
    if needs_compression(self.messages, ...):
        self.messages = run_memory_check_then_compress(...)

    # 2. 会话保存 — 持久化到文件
    if settings.auto_save:
        save_session(self.session_id, self.messages, ...)
```

**设计要点**:
- 记忆保存由 LLM 通过 `memory_save` 工具自主触发，不在 `_post_turn` 中处理
- 压缩在保存之前，确保保存的是压缩后的状态
- 每步都 try-catch，任何一步失败不影响其他步骤

---

## 配置变更

`agent/config.py` 新增 5 个配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_context_tokens` | 64000 | 最大上下文 token 数 |
| `compression_threshold` | 0.95 | 自动压缩触发阈值 |
| `head_keep` | 3 | 压缩时保留的前 N 条消息 |
| `tail_keep` | 20 | 压缩时保留的后 N 条消息 |
| `auto_save` | True | 是否自动保存会话 |

均可通过 `.env` 覆盖。长期记忆始终开启，无需配置。

---

## CLI 新增命令

| 命令 | 说明 |
|------|------|
| `/sessions` | 列出所有保存的会话 |
| `/resume <id>` | 恢复指定会话继续对话 |
| `/new` | 新建会话（保留长期记忆） |
| `/compact` | 手动压缩对话历史 |

`/status` 命令增强：显示 Session ID、消息数、Token 用量。

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `agent/session.py` | **新增** | 会话持久化与恢复 |
| `agent/memory.py` | **新增** | 长期记忆存储（纯文件 I/O） |
| `agent/tokens.py` | **新增** | Token 估算与上下文压缩 |
| `tools/memory_save.py` | **新增** | 记忆保存工具（LLM 自主调用） |
| `tools/session_search.py` | **新增** | 历史会话搜索工具 |
| `tools/edit_file.py` | **新增** | 通过字符串替换编辑文件 |
| `tools/grep_search.py` | **新增** | 按正则/关键词搜索文件内容 |
| `tools/fetch_url.py` | **新增** | 抓取网页并提取纯文本 |
| `tools/tree.py` | **新增** | 递归展示目录树结构 |
| `tools/find_file.py` | **新增** | 按 glob 模式查找文件 |
| `tools/file_delete.py` | **新增** | 删除指定文件 |
| `tools/system_info.py` | **新增** | 获取系统运行环境信息 |
| `agent/agent.py` | 修改 | 集成 session_id、memory 注入、\_post_turn 钩子 |
| `agent/cli.py` | 修改 | 新增 4 个命令、增强 /status、更新 banner |
| `agent/config.py` | 修改 | 新增 5 个配置项 |
| `.gitignore` | 修改 | 新增 `.genesis/` |
| `pyproject.toml` | 修改 | 版本号更新为 0.3.0 |
