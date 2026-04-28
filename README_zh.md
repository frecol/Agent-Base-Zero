# Agent-Base-Zero

**一个 commit 代表一个智能体版本，从零学会构建通用 AI Agent。**

Agent-Base-Zero 是一个开源项目，教你如何基于单个 LLM(deepseek) 由简到难构建通用智能体。
整个代码库通过 git commit 逐步演进 —— 每个版本引入一个新概念或功能，从最简对话循环到完整的自主智能体。

**学习方式：** `git log --oneline` 查看所有演进阶段，`git checkout <commit>` 跳转到任意版本阅读代码。

## 演进路线
| 版本 | 主题 | 核心概念 |
|------|------|---------|
| **v0.1** | 最小可用 Agent | Agent 循环、工具调用、CLI、工具注册表 |
| **v0.2** | 流式与思维链 | 流式输出、DeepSeek 思维链模式、联网搜索、时间工具 |
| **v0.3** | 记忆系统 | 会话持久化、长期记忆、上下文压缩 |
| **v0.4** | 技能系统 | 技能注册、PromptManager、LLM 自动激活、组合工具 |
| **v0.5** | Plan 模式 | Plan/Normal 双模式、只读探索、结构化规划、逐步执行 |
| v0.6 | 社交媒体接入 | API 集成、异步操作 |
| v0.7 | 多智能体协作 | Agent 协同、任务路由 |
当前进度为 v0.5

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

约 300 行 Python 代码实现一个可运行的 Agent。

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

## v0.2 — 流式输出、思维链与联网搜索

在 v0.1 基础上新增三大能力：

**流式输出** — 逐 token 实时渲染。新增 `run_turn_stream()` 方法，通过 Generator 产出结构化的 `StreamEvent` 事件（`thinking` / `content` / `done`），CLI 可按阶段分区显示。

**思维链模式** — 接入 DeepSeek Thinking API，LLM 在回答前先输出推理过程。`_build_messages()` 根据模式开关决定是否保留 `reasoning_content`，兼顾推理连续性与 token 开销。

**联网搜索工具** — 两个新工具突破本地限制：
- `web_search` — 基于 DuckDuckGo 的网页搜索，无需 API Key
- `current_time` — 当前时间查询，与 web_search 配合使用（如"今天的新闻"→ 获取日期 → 搜索）

其他改进：`max_tokens` 提升至 8192、新增 `/stream` `/think` `/status` CLI 命令、优化 System Prompt、工具调用展示升级为 Rich Panel 组件。

## v0.3 — 记忆系统

在 v0.2 基础上新增会话持久化、长期记忆和上下文压缩三大能力。

**会话持久化** — 每轮对话自动保存到 `.genesis/sessions/{session_id}.json`。会话文件包含元数据（标题、时间戳、消息数），采用原子写入防止数据损坏。

**会话管理命令：**
- `/sessions` — 列出所有已保存的会话及时间、标题
- `/resume <id>` — 恢复指定会话，继续之前的对话
- `/new` — 新建会话（长期记忆跨会话保留）

**长期记忆** — LLM 可通过 `memory_save` 工具自主保存关于用户的持久事实。记忆跨所有会话共享，在启动时作为 system prompt 注入。LLM 自行判断什么值得记住——用户偏好、环境细节、稳定约定。不需要配置，始终开启。

**会话搜索** — `session_search` 工具允许 LLM 按关键词搜索历史对话，无需手动翻文件。

**上下文压缩** — 当对话过长时（默认 >60,800 tokens），中间部分自动由 LLM 总结为结构化摘要。`/compact` 命令可手动触发。头部 3 条和尾部 20 条关键消息保持完整。

**Token 估算** — 基于 chars/4 启发式估算 token 用量，在 `/status` 中展示，帮助主动管理上下文，无需外部 tokenizer。

| 功能 | 说明 |
|------|------|
| `/status` | 现在显示 Session ID、消息数、Token 用量 |
| `/sessions` | 列出所有已保存会话 |
| `/resume <id>` | 继续之前的会话 |
| `/new` | 新建会话 |
| `/compact` | 手动压缩对话历史 |

## v0.4 — 技能系统

在 v0.3 基础上新增技能注册、PromptManager、LLM 自主激活技能和组合工具。

**技能系统** — 每个技能是 `skills/` 目录下的一个文件夹，包含一个 `SKILL.md` 文件（YAML frontmatter + Markdown 指令）。启动时自动发现。SKILL.md 格式兼容市面上标准 skill 格式（如 Claude Code）。

**PromptManager** — 替代 `client.py` 中硬编码的 `SYSTEM_PROMPT`。从 `agent/system_prompt.md` 加载基础提示词，启动时拼接技能索引（所有技能名称+描述）。系统提示词在整个会话中保持不变，保证提示词前缀缓存稳定。

**LLM 自主激活** — LLM 在系统提示词中看到技能索引，当任务匹配时自主调用 `activate_skill(name)` 激活技能。技能详细指令作为工具调用结果返回（不注入系统提示词），前缀缓存不破。LLM 也可调用 `deactivate_skill` 回到基础模式。

**组合工具** — 新增 `research_topic` 工具展示工具组合：handler 内部通过 `registry.dispatch()` 链式调用 `web_search` + `fetch_url`，提供一步到位的研究能力。

**缓存友好设计：**
```
系统提示词（稳定前缀）:
  基础提示词 + 技能索引 → 始终不变 → 缓存命中

技能详细指令:
  activate_skill 工具返回结果 → 进入对话历史
  系统提示词不变
```

**技能命令：**
- `/skills` — 列出所有可用技能（标记当前激活的，显示仅自动技能）
- `/skill <name>` — 手动激活技能（受 `user_invocable` 限制）
- `/unskill` — 取消激活当前技能

**示例技能：**
- `research` — 搜索研究专家，包含搜索策略和引用规范
- `code_assistant` — 代码助手专家，包含调试和代码风格指南

| 功能 | 说明 |
|------|------|
| `/skills` | 列出所有可用技能（标记仅自动技能） |
| `/skill <name>` | 激活指定技能（受 `user_invocable` 限制） |
| `/unskill` | 取消当前技能 |
| `/status` | 现在显示当前激活的技能 |

## v0.5 — Plan 模式（当前版本）

在 v0.4 基础上新增 Plan/Normal 双模式、工具分类、结构化规划、交互式审阅和逐步执行。

**Plan 模式** — 双模式工作流，将只读探索/规划与执行分离。按 **Shift+Tab** 或输入 `/plan` 切换 Normal 模式（所有工具可用）和 Plan 模式（仅只读工具）。在 Plan 模式下，Agent 先探索代码库、分析任务、生成结构化计划供用户审阅，确认后才执行变更。

**工具分类** — 所有工具在 `tools/registry.py` 中被标记为 `read_only` 或 `write`。Plan 模式下 LLM 只收到只读工具定义（read_file, list_dir, grep_search 等），执行层还设有硬防护，拦截任何写工具调用——即使模型从之前的 Normal 模式对话历史中"幻觉"出写工具调用。

**结构化规划** — LLM 按固定格式生成计划（Goal + 编号 Steps）。计划解析器支持多种格式变体（编号风格、分隔符）。无正式标题时自动回退到宽松解析。

**交互式审阅** — Agent 生成计划后，品红色边框面板展示目标和步骤。用户可以选择接受（执行）、修改（提供反馈修订，最多 3 轮）或取消。

**逐步执行** — 计划接受后，Agent 切换到 Normal 模式静默执行每个步骤。实时进度显示带有 Shimmer 动画，展示每个步骤的状态（Pending → In Progress → Done/Failed/Skipped）。失败时用户可选择继续或停止。

**视觉增强** — `prompt_toolkit` 输入处理器支持 Shift+Tab 检测、Rich Live 进度条 8fps 自动刷新、Shimmer 光扫描动画、完成摘要面板。

**Plan 模式命令：**
- `/plan` 或 **Shift+Tab** — 切换 Plan / Normal 模式
- `/status` — 现在显示当前模式（Normal / Plan）

| 功能 | 说明 |
|------|------|
| `/plan` | 切换 Plan / Normal 模式 |
| Shift+Tab | 输入时即时切换模式 |
| Plan 审阅 | 执行前可接受 / 修改 / 取消计划 |
| 实时进度 | Shimmer 动画展示步骤状态 |
| 写工具防护 | 执行层在 Plan 模式下阻止写工具 |

## 项目结构

```
Agent-Base-Zero/
├── agent/
│   ├── config.py          # Pydantic Settings 配置，读取 .env
│   ├── client.py          # DeepSeek API 客户端（流式 + 思维链）
│   ├── prompt.py          # PromptManager：基础提示词 + Plan 模式 + 技能索引 (v0.4+)
│   ├── prompts/           # 提示词模板目录 (v0.5)
│   │   ├── system_prompt.md   # 基础系统提示词 (v0.4)
│   │   └── plan_prompt.md     # Plan 模式专用提示词 (v0.5)
│   ├── plan.py            # Plan 数据结构：PlanPhase, StepStatus, PlanStep, Plan (v0.5)
│   ├── plan_parser.py     # 从 LLM 响应中解析结构化计划 (v0.5)
│   ├── plan_input.py      # InputHandler，支持 Shift+Tab 键绑定 (v0.5)
│   ├── plan_renderer.py   # Plan 审阅面板 + 实时进度条渲染 (v0.5)
│   ├── shimmer.py         # 共享 Shimmer 动画位置计算器 (v0.5)
│   ├── agent.py           # 核心 Agent 循环 + StreamEvent + Plan 模式
│   ├── session.py         # 会话持久化与恢复 (v0.3)
│   ├── memory.py          # 长期记忆存储 (v0.3)
│   ├── tokens.py          # Token 估算与上下文压缩 (v0.3)
│   └── cli.py             # 交互式 CLI（Rich + 流式渲染 + Plan 模式）
├── tools/
│   ├── registry.py        # 工具注册表（注册 + 分发 + read_only/write 分类）
│   ├── read_file.py       # 读取文件内容
│   ├── write_file.py      # 写入文件
│   ├── list_dir.py        # 列出目录内容
│   ├── run_command.py     # 执行 Shell 命令
│   ├── web_search.py      # DuckDuckGo 联网搜索
│   ├── current_time.py    # 当前时间查询
│   ├── edit_file.py       # 通过字符串替换编辑文件 (v0.3)
│   ├── grep_search.py     # 按正则/关键词搜索文件内容 (v0.3)
│   ├── fetch_url.py       # 抓取网页并提取纯文本 (v0.3)
│   ├── tree.py            # 递归展示目录树结构 (v0.3)
│   ├── find_file.py       # 按 glob 模式查找文件 (v0.3)
│   ├── file_delete.py     # 删除指定文件 (v0.3)
│   ├── system_info.py     # 获取系统运行环境信息 (v0.3)
│   ├── memory_save.py     # 保存事实到长期记忆 (v0.3)
│   ├── session_search.py  # 搜索历史对话 (v0.3)
│   └── research_topic.py  # 组合工具：搜索 + 抓取 (v0.4)
├── skills/
│   ├── registry.py        # 技能注册表、SKILL.md 解析、技能工具 (v0.4)
│   ├── research/          # 搜索研究技能 (v0.4)
│   │   └── SKILL.md
│   └── code_assistant/    # 代码助手技能 (v0.4)
│       └── SKILL.md
├── docs/
│   ├── v01/               # v0.1 架构文档（中文 + 英文）
│   ├── v02/               # v0.2 架构文档（中文 + 英文）
│   ├── v03/               # v0.3 架构文档（中文 + 英文）
│   ├── v04/               # v0.4 架构文档（中文 + 英文）
│   └── v05/               # v0.5 架构文档（中文 + 英文）
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

## 添加新技能

每个技能是一个包含 `SKILL.md` 文件的文件夹：

```markdown
<!-- skills/my_skill/SKILL.md -->
---
name: my_skill
description: "技能的功能描述和激活时机。"
user_invocable: true
---

# 我的技能

LLM 激活此技能后的详细指令...
```

将文件夹放入 `skills/` 即可，启动时自动发现。`name` 和 `description` 会出现在系统提示词的技能索引中，LLM 可通过调用 `activate_skill("my_skill")` 激活该技能。设置 `user_invocable: false` 可限制为仅 LLM 自动激活（`/skill` 命令拒绝手动激活，`/skills` 中显示 `(auto-only)`）。