# v0.2 更新详解：流式输出、思维链与联网搜索

v0.2 在 v0.1 的"对话循环 + 工具注册"基础上，新增了三大能力：**流式输出（Streaming）**、**思维链模式（Thinking）** 和 **联网搜索工具**，并优化了 CLI 交互体验。

---

## 目录

- [变更总览](#变更总览)
- [新增能力一：流式输出](#新增能力一流式输出)
  - [StreamEvent 事件模型](#streamevent-事件模型)
  - [run_turn_stream 方法](#run_turn_stream-方法)
  - [CLI 流式渲染](#cli-流式渲染)
- [新增能力二：思维链模式](#新增能力二思维链模式)
  - [DeepSeek Thinking API](#deepseek-thinking-api)
  - [reasoning_content 处理策略](#reasoning_content-处理策略)
- [新增能力三：联网搜索工具](#新增能力三联网搜索工具)
  - [web_search 工具](#web_search-工具)
  - [current_time 工具](#current_time-工具)
- [配置与依赖变更](#配置与依赖变更)
- [CLI 新增命令](#cli-新增命令)
- [System Prompt 优化](#system-prompt-优化)
- [文件变更清单](#文件变更清单)

---

## 变更总览

| 维度 | v0.1 | v0.2 |
|------|------|------|
| 响应模式 | 仅非流式（等待完整响应） | 流式 + 非流式双模式 |
| 思维链 | 无 | 支持 DeepSeek Thinking |
| 工具数量 | 4 个 | 6 个（+web_search、current_time） |
| max_tokens | 4096 | 8192 |
| CLI 命令 | /help /clear /exit /tools | +/stream /think /status |
| 依赖 | openai, pydantic-settings, rich, python-dotenv | +ddgs, tzdata |

---

## 新增能力一：流式输出

v0.1 中 Agent 每次回答都要等 LLM 完整生成后才显示，体验上有明显延迟。v0.2 引入了流式输出，用户可以实时看到 LLM 的逐字生成过程。

### StreamEvent 事件模型

`agent/agent.py` 新增了 `StreamEvent` 数据类，作为流式输出的统一事件载体：

```python
# agent/agent.py

@dataclass
class StreamEvent:
    """A structured event yielded during streaming."""
    type: str  # "thinking" | "content" | "tool_start" | "tool_result" | "done"
    data: dict = field(default_factory=dict)
```

流式输出不再返回一个字符串，而是通过 Generator 逐步 yield 事件对象：

| 事件类型 | 含义 | data 内容 |
|----------|------|-----------|
| `"thinking"` | LLM 的思维过程片段 | `{"text": "..."}` |
| `"content"` | 正式回复文本片段 | `{"text": "..."}` |
| `"done"` | 回合结束 | `{"content": "...", "reasoning": "..."}` |

### run_turn_stream 方法

`Agent` 类新增了 `run_turn_stream()` 方法，与 v0.1 的 `run_turn()` 对应：

```python
# agent/agent.py — run_turn_stream() 核心逻辑（简化）

def run_turn_stream(self, user_input, thinking_enabled=False,
                    on_tool_call=None, on_tool_result=None):
    self.messages.append({"role": "user", "content": user_input})
    tools = self._get_tools()

    for _ in range(self.max_iterations):
        stream = self.client.chat_stream(
            api_messages, tools=tools, thinking_enabled=thinking_enabled
        )

        # 累积器：收集流式片段
        content_chunks = []
        reasoning_chunks = []
        tool_call_accs = {}  # 工具调用需要按 index 累积

        for chunk in stream:
            delta = chunk.choices[0].delta

            # 1. 思维内容（最先到达）
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_chunks.append(delta.reasoning_content)
                yield StreamEvent(type="thinking", data={"text": delta.reasoning_content})

            # 2. 正式回复
            if delta.content:
                content_chunks.append(delta.content)
                yield StreamEvent(type="content", data={"text": delta.content})

            # 3. 工具调用片段（按 index 累积拼接）
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    # ...累积 id、name、arguments...
                    pass

        # 流结束，判断是否有工具调用
        if tool_call_accs:
            # 有工具调用 → 执行并继续循环
            self._execute_tool_calls(...)
            continue
        else:
            # 无工具调用 → 最终回答
            yield StreamEvent(type="done", ...)
            return content
```

关键点：
- 工具调用在流式模式下是**分片传输**的，需要通过 `index` 字段将片段累积拼接成完整的调用
- 流式工具调用完成后，会拼接为 `SimpleNamespace` 对象（通过 `_make_tool_call_namespace()` 辅助函数），复用现有的 `_execute_tool_calls()` 逻辑
- 整体循环逻辑（LLM → 工具 → LLM → ...）与 v0.1 一致

### CLI 流式渲染

`agent/cli.py` 新增了 `_stream_response()` 函数，专门处理流式模式的终端渲染：

```
用户输入后，终端实时显示：

───── Thinking ──────────────────────────────
用户想让我读取 config.py，我需要用 read_file 工具...
─────────────────────────────────────────────

┌─ read_file ─────────┐
│ path='agent/config.py' │
└──────────────────────┘
  Result: {"success": true, "content": "..."}
─────────────────────────────────────────────

───── Response ──────────────────────────────
这是 config.py 的内容：from pydantic_settings...
───── Response Finish ───────────────────────
```

设计要点：
- 使用 `_Phase` 枚举跟踪当前阶段（IDLE → THINKING → CONTENT），避免重复输出标题
- 思维内容使用 ANSI 转义码（`\033[2m`）实现灰色暗淡效果，直接写入 stdout 绕过 Rich 的逐 token 渲染开销
- 工具调用展示改用 `Panel` 组件，比 v0.1 的单行文本更清晰

---

## 新增能力二：思维链模式

v0.2 接入了 DeepSeek 的 Thinking 能力，让 LLM 在给出回答前先进行一段"内心独白"，提升推理质量。

### DeepSeek Thinking API

在 `agent/client.py` 中，`chat()` 和 `chat_stream()` 都新增了 `thinking_enabled` 参数：

```python
# agent/client.py

def chat(self, messages, tools=None, thinking_enabled=False):
    kwargs = {
        "model": self.model,
        "messages": messages,
        "max_tokens": self.max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    if thinking_enabled:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return self.client.chat.completions.create(**kwargs)
```

启用 thinking 后，API 返回的 `delta` 中会包含 `reasoning_content` 字段，包含 LLM 的推理过程文本。

### reasoning_content 处理策略

思维链内容需要保存在 `messages` 历史中，以便多轮工具调用时 LLM 能延续之前的推理。但思维内容通常很长，会增加 API 请求的 token 开销。

`_build_messages()` 新增了条件过滤逻辑：

```python
# agent/agent.py — _build_messages()

def _build_messages(self, thinking_enabled=False):
    result = [{"role": "system", "content": self.system_prompt}]
    for msg in self.messages:
        if not thinking_enabled and "reasoning_content" in msg:
            # 非 thinking 模式：剔除 reasoning_content 节省带宽
            msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
        result.append(msg)
    return result
```

策略：
- **thinking 开启时**：保留 `reasoning_content`，让 LLM 能看到之前的推理过程
- **thinking 关闭时**：剔除 `reasoning_content`，节省 token 开销

---

## 新增能力三：联网搜索工具

v0.2 新增了两个工具，打破了 v0.1 Agent 只能与本地系统交互的限制。

### web_search 工具

`tools/web_search.py` 使用 [DuckDuckGo Search](https://pypi.org/project/ddgs/) 实现网页搜索：

```python
# tools/web_search.py

SCHEMA = {
    "name": "web_search",
    "description": "Search the web using DuckDuckGo. "
                   "Returns a list of results with title, URL, and a short snippet.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {"type": "integer", "description": "Max results (default 5)."},
        },
        "required": ["query"],
    },
}

def handler(args: dict) -> str:
    with DDGS() as ddgs:
        results = [{"title": r["title"], "href": r["href"], "body": r["body"]}
                    for r in ddgs.text(query, max_results=max_results)]
    return json.dumps({"success": True, "results": results})
```

特点：
- 无需 API Key，直接通过 DuckDuckGo 搜索
- 返回标题、链接和摘要
- SCHEMA 的 description 中特别提示 LLM：搜索前需将相对时间词（如"今天"）转为具体日期

### current_time 工具

`tools/current_time.py` 提供当前时间查询，配合 web_search 使用：

```python
# tools/current_time.py

def handler(args: dict) -> str:
    tz_name = args.get("timezone") or settings.default_timezone
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    return json.dumps({
        "success": True,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": tz_name,
        "weekday": now.strftime("%A"),
    })
```

设计意图：LLM 本身不知道当前时间。当用户问"今天有什么新闻"时，Agent 需要先用 `current_time` 获取日期，再用 `web_search` 搜索。两个工具配合形成了一个完整的联网能力链。

默认时区通过配置项 `default_timezone` 设置，默认值为 `Asia/Shanghai`。

---

## 配置与依赖变更

### 新增配置项

`agent/config.py` 新增了三个配置项：

```python
# agent/config.py

class Settings(BaseSettings):
    # ... 原有配置 ...
    max_tokens: int = 8192            # 从 4096 提升到 8192

    # Streaming & Thinking (v0.2)
    stream_enabled: bool = True       # 默认开启流式输出
    thinking_enabled: bool = True     # 默认开启思维链

    # Timezone
    default_timezone: str = "Asia/Shanghai"  # 默认时区
```

所有配置均可通过 `.env` 文件或环境变量覆盖。

### 新增依赖

`pyproject.toml` 新增了两个依赖：

| 依赖 | 版本 | 用途 |
|------|------|------|
| `ddgs` | >=7.0.0 | DuckDuckGo 搜索客户端 |
| `tzdata` | >=2024.1 | 时区数据（Windows 系统需要） |

---

## CLI 新增命令

v0.2 在 CLI 中新增了三个内置命令：

| 命令 | 功能 |
|------|------|
| `/stream` | 切换流式/非流式模式 |
| `/think` | 切换思维链模式开/关 |
| `/status` | 显示当前设置（流式、思维链、模型名） |

`run_cli()` 内部通过局部变量 `stream_enabled` 和 `thinking_enabled` 控制模式切换，每次对话回合根据当前状态选择 `run_turn()` 或 `run_turn_stream()`。

---

## System Prompt 优化

System Prompt 从 v0.1 的简短描述更新为更完整的角色定义：

```
v0.1:
  "You can read files, write files, list directories, and run shell commands.
   Always think step-by-step before using a tool."

v0.2:
  "You assist users with a wide range of tasks including answering questions,
   writing and editing code, analyzing information, creative work...
   You are a CLI AI Agent. Try not to use markdown but simple text
   renderable inside a terminal."
```

主要变化：
- 从"文件操作助手"升级为"通用任务助手"的定位
- 明确指出自己是 CLI Agent，提示 LLM 避免使用复杂 Markdown（终端渲染有限）
- 强调效率与清晰沟通

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `agent/agent.py` | 修改 | 新增 `StreamEvent`、`run_turn_stream()`、`_make_tool_call_namespace()`；`_build_messages()` 增加 thinking 过滤 |
| `agent/cli.py` | 修改 | 新增 `_stream_response()`、`_Phase` 枚举、`/stream` `/think` `/status` 命令；优化工具调用展示 |
| `agent/client.py` | 修改 | `chat()` 和 `chat_stream()` 增加 `thinking_enabled` 参数 |
| `agent/config.py` | 修改 | 新增 `stream_enabled`、`thinking_enabled`、`default_timezone`；`max_tokens` 提升到 8192 |
| `tools/web_search.py` | **新增** | DuckDuckGo 联网搜索工具 |
| `tools/current_time.py` | **新增** | 当前时间查询工具 |
| `pyproject.toml` | 修改 | 版本升级到 0.2.0，新增 ddgs、tzdata 依赖 |
| `.env.example` | 修改 | 新增 v0.2 配置示例 |
| `README.md` / `README_zh.md` | 修改 | 当前进度标记更新为 v0.2 |
