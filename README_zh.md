# Agent-Base-Zero

**一个 commit 代表一个智能体版本，从零学会构建通用 AI Agent。**

Agent-Base-Zero 是一个开源项目，教你如何基于单个 LLM(deepseek) 由简到难构建通用智能体。
整个代码库通过 git commit 逐步演进 —— 每个版本引入一个新概念或功能，从最简对话循环到完整的自主智能体。

**学习方式：** `git log --oneline` 查看所有演进阶段，`git checkout <commit>` 跳转到任意版本阅读代码。

## 演进路线
| 版本 | 主题 | 核心概念 |
|------|------|---------|
| **v0.1** | 最小可用 Agent | Agent 循环、工具调用、CLI、工具注册表 |
| v0.2 | 流式输出 | SSE 流式传输、实时显示 |
| v0.3 | 记忆系统 | 会话持久化、长期记忆、上下文压缩 |
| v0.4 | 技能系统 | 技能注册、Prompt 模板、组合工具 |
| v0.5 | 规划与执行 | 任务分解、多步规划、自我反思 |
| v0.6 | 社交媒体接入 | API 集成、异步操作 |
| v0.7 | 多智能体协作 | Agent 协同、任务路由 |
当前进度为v0.1

## 快速开始

> **前置条件：** [uv](https://docs.astral.sh/uv/)（Python 包管理器）、Python >= 3.11、[DeepSeek API Key](https://platform.deepseek.com/)

```bash
# 1. 克隆仓库
git clone https://github.com/frecol/Agent-Base-Zero.git
cd Agent-Base-Zero

# 2. 安装依赖
uv sync

# 3. 设置 API Key
echo "DEEPSEEK_API_KEY=你的密钥" > .env

# 4. 启动
uv run genesis
```

## v0.1 — 最小可用 Agent

当前版本。约 300 行 Python 代码实现一个可运行的 Agent。

**功能：**
- 基于 Rich 的交互式 CLI 界面
- 多轮对话 + 工具调用循环
- 4 个内置工具：`read_file`、`write_file`、`list_dir`、`run_command`
- 自动发现的工具注册表（在 `tools/` 目录中放入文件即可注册）

**Agent 循环工作原理：**

```
用户输入 → Agent → LLM
                    ↓
               需要调用工具？
                ├─ 是 → 执行工具 → 追加结果 → 回到 LLM
                └─ 否 → 返回回复给用户
```

循环持续到 LLM 返回纯文本（不再请求工具），最多 50 轮迭代。

## 项目结构

```
Agent-Base-Zero/
├── agent/
│   ├── config.py          # Pydantic Settings 配置，读取 .env
│   ├── client.py          # DeepSeek API 客户端（基于 OpenAI SDK）
│   ├── agent.py           # 核心 Agent 循环（LLM ↔ 工具）
│   └── cli.py             # 交互式 CLI（Rich 美化）
├── tools/
│   ├── registry.py        # 工具注册表（注册 + 分发）
│   ├── read_file.py       # 读取文件内容
│   ├── write_file.py      # 写入文件
│   ├── list_dir.py        # 列出目录内容
│   └── run_command.py     # 执行 Shell 命令
├── main.py                # 入口（python main.py）
├── pyproject.toml         # 项目配置与依赖
└── .env                   # 你的 API Key（不纳入版本控制）
```

## 添加新工具

每个工具文件在导入时自动注册：

```python
# tools/my_tool.py
import json
from tools.registry import registry

SCHEMA = {
    "name": "my_tool",
    "description": "这个工具的功能描述",
    "parameters": {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "某个输入参数"},
        },
        "required": ["input"],
    },
}

def handler(args: dict) -> str:
    result = do_something(args["input"])
    return json.dumps({"success": True, "result": result})

registry.register("my_tool", SCHEMA, handler)
```

将文件放入 `tools/` 目录即可，启动时自动发现并注册。