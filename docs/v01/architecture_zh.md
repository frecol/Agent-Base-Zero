# v0.1 架构详解：对话循环与工具系统

本文档深入讲解 Agent-Base-Zero v0.1 的两大核心机制：**对话循环（Agent Loop）** 和 **工具注册与调用（Tool Registry）**。

---

## 目录

- [项目概览](#项目概览)
- [对话循环机制](#对话循环机制)
  - [整体流程](#整体流程)
  - [消息格式](#消息格式)
  - [核心代码走读](#核心代码走读)
- [工具注册与使用机制](#工具注册与使用机制)
  - [注册表：ToolRegistry](#注册表toolregistry)
  - [自动发现机制](#自动发现机制)
  - [调度执行](#调度执行)
- [如何编写一个新工具](#如何编写一个新工具)
- [内置工具一览](#内置工具一览)
- [CLI 交互层](#cli-交互层)
- [总结](#总结)

---

## 项目概览

v0.1 是整个项目最精简的起点，只做三件事：

1. **与 LLM 对话** — 通过 OpenAI 兼容协议调用 DeepSeek API
2. **调用工具** — LLM 可以自主决定何时调用哪个工具
3. **终端交互** — 用户通过命令行与 Agent 对话

### 目录结构

```
Agent-Base-Zero/
├── main.py                  # 入口：启动 CLI
├── agent/
│   ├── config.py            # 配置管理（API Key、模型名等）
│   ├── client.py            # DeepSeek API 客户端封装
│   ├── agent.py             # 核心对话循环
│   └── cli.py               # 终端 CLI 界面
├── tools/
│   ├── __init__.py          # 自动发现并注册所有工具
│   ├── registry.py          # 工具注册表（核心）
│   ├── read_file.py         # 工具：读文件
│   ├── write_file.py        # 工具：写文件
│   ├── list_dir.py          # 工具：列目录
│   └── run_command.py       # 工具：执行命令
└── docs/v01/                # 本文档所在目录
```

整个 v0.1 只有约 **650 行 Python 代码**，适合作为学习 Agent 的第一个台阶。

---

## 对话循环机制

对话循环是 Agent 的心脏。它决定了"用户说一句话，Agent 如何思考、行动、直到给出最终回答"的完整过程。

### 整体流程

```
用户输入
  │
  ▼
┌─────────────────────────────────────┐
│ 1. 将用户消息追加到 messages 列表     │
│ 2. 获取所有已注册工具的定义            │
└──────────────────┬──────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  调用 LLM API       │ ◄──────────────────────┐
        │  (带上对话历史+工具)  │                        │
        └──────────┬──────────┘                        │
                   │                                   │
                   ▼                                   │
          LLM 返回了工具调用？                           │
           ┌─────┴─────┐                              │
           │ 是        │ 否                            │
           ▼           ▼                               │
    执行工具调用    返回最终文本响应                      │
    追加结果到      （结束循环）                          │
    messages 列表                                      │
           │                                           │
           └───────────────────────────────────────────┘
                  （继续循环，让 LLM 看到工具结果）
```

关键点：**这不是一次调用就结束的**。当 LLM 决定使用工具时，工具的执行结果会被追加到对话历史中，然后**再次调用 LLM**，让它根据工具结果决定下一步——可能是继续调用工具，也可能是直接给出最终回答。

为了防止无限循环，设置了最大迭代次数 `MAX_TOOL_ITERATIONS = 50`。

### 消息格式

对话历史 `self.messages` 是一个字典列表，遵循 OpenAI 消息格式。一次包含工具调用的完整对话看起来像这样：

```python
messages = [
    # 系统提示词（不在 messages 中，由 _build_messages() 动态拼入）
    {"role": "system", "content": "You are Genesis, a helpful AI assistant..."}

    # 用户输入
    {"role": "user", "content": "帮我读一下 config.py 的内容"},

    # LLM 决定调用工具（assistant 消息带 tool_calls 字段）
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"agent/config.py\"}"
            }
        }]
    },

    # 工具执行结果（role 为 "tool"，需携带 tool_call_id）
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "{\"success\": true, \"content\": \"from pydantic_settings...\"}"
    },

    # LLM 根据工具结果生成的最终回答
    {"role": "assistant", "content": "这是 config.py 的内容：\n..."}
]
```

注意 `tool` 消息的 `tool_call_id` 必须与对应的 `assistant.tool_calls` 中的 `id` 一致，这样 LLM 才能将结果与对应的工具调用关联起来。

### 核心代码走读

对话循环的核心在 `agent/agent.py` 的 `Agent.run_turn()` 方法中：

```python
# agent/agent.py — Agent.run_turn()

def run_turn(self, user_input, on_tool_call=None, on_tool_result=None) -> str:
    # 第一步：追加用户消息
    self.messages.append({"role": "user", "content": user_input})
    tools = self._get_tools()

    # 第二步：循环调用 LLM
    for _ in range(self.max_iterations):
        api_messages = self._build_messages()  # 每次调用LLM都是完整的拼入 system prompt 和 历史消息记录(用户消息、LLM回复、工具结果...)
        response = self.client.chat(api_messages, tools=tools)
        assistant_msg = response.choices[0].message

        msg_dict = {"role": "assistant", "content": assistant_msg.content or ""}

        if assistant_msg.tool_calls:
            # LLM 想调用工具
            msg_dict["tool_calls"] = [...]  # 序列化工具调用信息
            self.messages.append(msg_dict)
            self._execute_tool_calls(assistant_msg.tool_calls, on_tool_call, on_tool_result)
            continue  # 继续循环，将工具结果发给 LLM
        else:
            # LLM 给出最终回答
            self.messages.append(msg_dict)
            return msg_dict["content"]

    return "[Agent reached maximum tool iterations without a final response.]"
```

其中工具执行逻辑在 `_execute_tool_calls()` 中：

```python
# agent/agent.py — Agent._execute_tool_calls()

def _execute_tool_calls(self, tool_calls, on_tool_call=None, on_tool_result=None):
    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)

        # 回调通知 CLI 展示工具调用信息
        if on_tool_call:
            on_tool_call(name, args)

        # 通过注册表调度执行
        result = registry.dispatch(name, args)

        # 追加工具结果到消息历史
        self.messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
```

---

## 工具注册与使用机制

v0.1 的工具系统采用了 **中心化注册表 + 自动发现** 的设计，核心只有两个文件：

- `tools/registry.py` — 注册表本身
- `tools/__init__.py` — 自动发现入口

### 注册表：ToolRegistry

`ToolRegistry` 是一个简单的字典容器，存储每个工具的 **schema**（描述）和 **handler**（处理函数）：

```python
# tools/registry.py

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, dict] = {}
        # 每个条目: {"schema": dict, "handler": Callable}

    def register(self, name: str, schema: dict, handler: Callable) -> None:
        """注册一个工具。在工具模块被导入时自动调用。"""
        self._tools[name] = {"schema": schema, "handler": handler}

    def get_definitions(self, names=None) -> List[dict]:
        """返回 OpenAI 格式的工具定义列表，用于传给 LLM API。"""
        result = []
        for name in sorted(names or self._tools.keys()):
            entry = self._tools[name]
            result.append({
                "type": "function",
                "function": entry["schema"],
            })
        return result

    def dispatch(self, name: str, args: dict) -> str:
        """根据名称执行对应的工具处理函数，返回结果字符串。"""
        entry = self._tools.get(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return entry["handler"](args)
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})
```

项目使用一个**全局单例** `registry`，所有模块都从这个单例导入：

```python
# tools/registry.py 末尾
registry = ToolRegistry()
```

### 自动发现机制

工具的注册不需要手动导入——`tools/__init__.py` 中的 `discover_tools()` 会自动扫描 `tools/` 目录下的所有 `.py` 文件并动态导入：

```python
# tools/registry.py — discover_tools()

def discover_tools(tools_dir=None) -> List[str]:
    tools_path = tools_dir or Path(__file__).resolve().parent
    module_names = [
        f"tools.{p.stem}"
        for p in sorted(tools_path.glob("*.py"))
        if p.name not in {"__init__.py", "registry.py"}  # 排除自身
    ]
    for mod_name in module_names:
        importlib.import_module(mod_name)  # 导入即注册
    return imported
```

```python
# tools/__init__.py — 包导入时自动执行
from tools.registry import registry, discover_tools
discover_tools()
```

**工作原理**：每个工具文件在模块末尾都调用了 `registry.register()`。当 `discover_tools()` 通过 `importlib.import_module()` 导入该模块时，模块级代码自动执行，工具就注册完成了。

这意味着**添加新工具只需要在 `tools/` 目录下新建一个 `.py` 文件**，无需修改任何其他代码。

### 调度执行

当 Agent 收到 LLM 的工具调用请求时，通过 `registry.dispatch(name, args)` 执行：

```python
# agent/agent.py — _execute_tool_calls() 中
result = registry.dispatch(name, args)
```

`dispatch` 方法根据工具名从注册表中找到对应的 handler 并调用，返回值始终是字符串（通常是 JSON 格式）。如果工具不存在或执行出错，会返回包含 `"error"` 字段的 JSON 字符串。

---

## 如何编写一个新工具

以 `tools/read_file.py` 为例，展示标准的三段式结构：

```python
# tools/read_file.py

import json
from pathlib import Path
from tools.registry import registry

# ---- 第一部分：SCHEMA ----
# 告诉 LLM 这个工具叫什么、做什么、需要什么参数
SCHEMA = {
    "name": "read_file",
    "description": "Read the contents of a file at the given path.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
        },
        "required": ["path"],
    },
}

# ---- 第二部分：handler ----
# 接收参数字典，返回结果字符串（JSON 格式）
def handler(args: dict) -> str:
    path = args.get("path", "")
    try:
        content = Path(path).read_text(encoding="utf-8")
        return json.dumps({"success": True, "content": content})
    except FileNotFoundError:
        return json.dumps({"success": False, "error": f"File not found: {path}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# ---- 第三部分：注册 ----
# 模块被导入时自动执行
registry.register("read_file", SCHEMA, handler)
```

**编写新工具只需要三步**：

1. 在 `tools/` 下新建 `.py` 文件
2. 定义 `SCHEMA`（遵循 OpenAI Function Calling 格式）和 `handler` 函数
3. 在文件末尾调用 `registry.register()`

保存后重启程序即可生效，无需改动其他文件。

---

## 内置工具一览

v0.1 内置了 4 个工具，覆盖基本的文件系统和命令操作：

| 工具名 | 文件 | 功能 | 必需参数 |
|--------|------|------|----------|
| `read_file` | `tools/read_file.py` | 读取文件内容 | `path` |
| `write_file` | `tools/write_file.py` | 写入文件（自动创建父目录） | `path`, `content` |
| `list_dir` | `tools/list_dir.py` | 列出目录内容（名称、类型、大小） | `path`（可选，默认当前目录） |
| `run_command` | `tools/run_command.py` | 执行 Shell 命令 | `command`（可选：`timeout`） |

所有工具的返回值都是 JSON 字符串，统一包含 `success` 字段表示是否成功：

```json
// 成功
{"success": true, "content": "file content here..."}

// 失败
{"success": false, "error": "File not found: xxx"}
```

---

## CLI 交互层

CLI 层（`agent/cli.py`）使用 [Rich](https://rich.readthedocs.io/) 库实现终端美化输出，是整个 Agent 的最外层。

### 运行流程

```
main.py → run_cli() → 创建 Agent 实例 → 进入交互循环
```

```python
# agent/cli.py — run_cli()（简化版）

def run_cli():
    _print_banner()         # 显示欢迎横幅
    agent = Agent()         # 创建 Agent

    while True:
        user_input = Prompt.ask("[bold green]You[/bold green]")
        # 处理内置命令或转发给 Agent
        response = agent.run_turn(
            user_input,
            on_tool_call=_print_tool_call,      # 工具调用时的回调
            on_tool_result=_print_tool_result,   # 工具结果时的回调
        )
        # 用 Rich Panel + Markdown 渲染回答
```

### 内置命令

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助信息 |
| `/clear` | 清空对话历史 |
| `/exit` | 退出程序 |
| `/tools` | 列出所有已注册的工具 |

### 回调机制

`run_turn()` 接受两个可选回调函数：

- `on_tool_call(name, args)` — 工具被调用时触发，CLI 用来在终端显示工具调用信息
- `on_tool_result(name, display)` — 工具返回结果时触发，CLI 用来显示执行结果（截断到 200 字符）

这种回调设计将 **Agent 逻辑** 和 **UI 展示** 解耦——Agent 不关心结果怎么显示，CLI 不关心工具怎么执行。

---

## 总结

v0.1 的架构可以用一句话概括：**一个循环 + 一个注册表**。

```
┌─────────────────────────────────────────────────┐
│                    CLI (cli.py)                  │
│              用户输入 / Rich 渲染输出              │
├─────────────────────────────────────────────────┤
│                Agent (agent.py)                  │
│     对话循环：LLM ↔ 工具调用，直到得到最终回答      │
├──────────────────┬──────────────────────────────┤
│  Client          │        Registry              │
│  (client.py)     │    (tools/registry.py)        │
│  API 通信        │  注册表 + 自动发现 + 调度       │
├──────────────────┼──────────────────────────────┤
│                  │  read_file / write_file /     │
│                  │  list_dir / run_command       │
└──────────────────┴──────────────────────────────┘
```

核心设计要点：

- **循环驱动**：Agent 通过 `for` 循环不断调用 LLM，直到 LLM 不再请求工具调用
- **注册表模式**：工具自注册、自动发现，新增工具零改动
- **回调解耦**：Agent 逻辑与 UI 展示通过回调函数分离
- **OpenAI 兼容协议**：使用标准的 Function Calling 格式，兼容所有 OpenAI API 兼容的 LLM 服务

这是构建更复杂 Agent 功能的基石。后续版本将在此基础上增加记忆管理、Skill 系统、社媒接入等能力，但核心的"循环 + 注册表"模式不会改变。
