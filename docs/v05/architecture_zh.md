# v0.5 更新详解：Plan 模式 — 规划与执行分离

v0.5 在 v0.4 的"技能系统 + PromptManager + 组合工具"基础上，引入 **Plan 模式（Plan Mode）**。核心思想是将 Agent 的工作分为两个阶段：**只读探索 + 规划** 和 **逐步执行**。用户先让 Agent 在受控环境下分析任务、生成结构化计划，确认后再按步骤执行。六大新增组件：**Plan 数据结构**、**工具分类与只读过滤**、**计划解析器**、**Plan 模式提示词**、**交互式输入处理器**、**可视化渲染器**。

---

## 目录

- [变更总览](#变更总览)
- [新增能力一：Plan 模式入口与切换](#新增能力一plan-模式入口与切换)
  - [Shift+Tab 键绑定](#shifttab-键绑定)
  - [/plan 命令](#plan-命令)
  - [模式状态同步](#模式状态同步)
- [新增能力二：工具分类与只读过滤](#新增能力二工具分类与只读过滤)
  - [ToolRegistry 中的工具分类](#toolregistry-中的工具分类)
  - [Agent._get_tools 过滤机制](#agent_get_tools-过滤机制)
  - [执行层硬防护](#执行层硬防护)
- [新增能力三：Plan 数据结构与生命周期](#新增能力三plan-数据结构与生命周期)
  - [PlanPhase 与 StepStatus](#planphase-与-stepstatus)
  - [PlanStep 与 Plan](#planstep-与-plan)
- [新增能力四：计划解析器](#新增能力四计划解析器)
  - [LLM 输出格式](#llm-输出格式)
  - [parse_plan 解析流程](#parse_plan-解析流程)
  - [宽松解析回退](#宽松解析回退)
- [新增能力五：Plan 模式提示词](#新增能力五plan-模式提示词)
  - [PromptManager 条件注入](#promptmanager-条件注入)
  - [plan_prompt.md 内容](#plan_promptmd-内容)
- [新增能力六：可视化渲染](#新增能力六可视化渲染)
  - [Plan Review 面板](#plan-review-面板)
  - [执行进度实时显示](#执行进度实时显示)
  - [Shimmer 动画](#shimmer-动画)
  - [完成摘要](#完成摘要)
- [Plan 模式完整流程](#plan-模式完整流程)
  - [1. 规划阶段](#1-规划阶段)
  - [2. 审查阶段](#2-审查阶段)
  - [3. 执行阶段](#3-执行阶段)
  - [4. 完成阶段](#4-完成阶段)
- [CLI 变更](#cli-变更)
- [文件变更清单](#文件变更清单)

---

## 变更总览

| 维度 | v0.4 | v0.5 |
|------|------|------|
| 工作模式 | 单一模式（直接执行） | Normal + Plan 双模式（规划后执行） |
| 工具安全 | 所有工具始终可用 | Plan 模式仅暴露只读工具 + 执行层硬防护 |
| 提示词 | base + skills index | base + **plan prompt**（条件注入） + skills index |
| 任务执行 | LLM 一次性完成 | 结构化计划 → 用户确认 → 逐步执行 |
| 输入处理 | 基础 input() | prompt_toolkit 输入，支持 Shift+Tab 切换 |
| 可视化 | Rich 面板 | 新增 Plan Review 面板、实时进度条、Shimmer 动画 |
| 新增文件 | 0 | 6 个（5 核心 + 1 提示词） |
| 新增依赖 | 0 | 1 个（`prompt-toolkit`） |

---

## 新增能力一：Plan 模式入口与切换

### Shift+Tab 键绑定

v0.5 引入 `prompt_toolkit` 替代标准 `input()` 来捕获用户输入，从而支持特殊按键检测：

```python
# agent/plan_input.py
class InputHandler:
    def _setup_keybindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("s-tab")
        def _toggle_mode(event):
            self._plan_mode = not self._plan_mode
            event.app.exit(result="")  # 立即退出提示，CLI 检测到切换

        return bindings
```

按下 Shift+Tab 时，`_plan_mode` 翻转，`get_input()` 返回空字符串。CLI 主循环检测到模式变化后同步 Agent 状态。

### /plan 命令

CLI 中也可通过 `/plan` 命令切换模式：

```python
elif cmd == "/plan":
    input_handler.plan_mode = not input_handler.plan_mode
    agent.set_plan_mode(input_handler.plan_mode)
```

### 模式状态同步

模式切换时需同步三个位置：

1. `InputHandler._plan_mode` — UI 层（决定提示符样式）
2. `Agent._plan_mode` — Agent 层（决定工具过滤 + 提示词注入）
3. `PromptManager._plan_mode_enabled` — 提示词层（条件注入 plan prompt）

```python
# agent/agent.py
def set_plan_mode(self, enabled: bool) -> None:
    self._plan_mode = enabled
    self.prompt_manager.set_plan_mode(enabled)  # 触发提示词更新
```

**提示符样式：**
- Normal 模式：`You> `（绿色）
- Plan 模式：`Plan> `（品红色）

---

## 新增能力二：工具分类与只读过滤

### ToolRegistry 中的工具分类

`tools/registry.py` 新增 `_TOOL_CATEGORIES` 字典，将所有工具分为两类：

```python
_TOOL_CATEGORIES: Dict[str, str] = {
    # Read-only tools — Plan 模式可用
    "read_file": "read_only",
    "list_dir": "read_only",
    "tree": "read_only",
    "find_file": "read_only",
    "grep_search": "read_only",
    "web_search": "read_only",
    "fetch_url": "read_only",
    "research_topic": "read_only",
    "current_time": "read_only",
    "system_info": "read_only",
    "session_search": "read_only",
    # Write tools — Plan 模式不可用
    "write_file": "write",
    "edit_file": "write",
    "file_delete": "write",
    "run_command": "write",
    "memory_save": "write",
    "activate_skill": "write",
    "deactivate_skill": "write",
}
```

新增方法：
- `get_read_only_names()` — 返回所有只读工具名称（Plan 模式用）
- `is_write_tool(name)` — 判断是否为写工具（执行层防护用）

### Agent._get_tools 过滤机制

```python
def _get_tools(self) -> Optional[List[dict]]:
    if self._plan_mode:
        names = registry.get_read_only_names()  # 只返回只读工具
    else:
        names = registry.get_all_names()         # 返回所有工具
    return registry.get_definitions(names) if names else None
```

API 层面：Plan 模式下不向 LLM 传递写工具定义。

### 执行层硬防护

即使 LLM 在 Plan 模式下"幻觉"出写工具调用（例如 Normal 模式的对话历史中包含写工具的 `tool_calls`，模型可能仍然尝试调用），`_execute_tool_calls()` 在执行层拦截：

```python
if self._plan_mode and registry.is_write_tool(name):
    rejected = json.dumps({"error": f"Tool '{name}' is not available in Plan Mode."})
    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": rejected})
    continue  # 跳过执行，返回错误给 LLM
```

**双重防护**：API 层面不提供写工具定义 + 执行层面拒绝写工具调用，确保 Plan 模式的只读安全性。

---

## 新增能力三：Plan 数据结构与生命周期

### PlanPhase 与 StepStatus

```python
class PlanPhase(Enum):
    PLANNING    = auto()  # Agent 使用只读工具探索并生成计划
    REVIEWING   = auto()  # 用户审阅计划，可接受/修改/取消
    EXECUTING   = auto()  # 计划被接受，逐步执行中
    COMPLETED   = auto()  # 所有步骤完成（成功或失败）

class StepStatus(Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    DONE        = "done"
    FAILED      = "failed"
    SKIPPED     = "skipped"
```

### PlanStep 与 Plan

```python
@dataclass
class PlanStep:
    index: int              # 1-based 步骤编号
    title: str              # 简短标题（一行）
    description: str        # 详细描述
    status: StepStatus = StepStatus.PENDING

@dataclass
class Plan:
    goal: str                               # 用户原始目标
    steps: List[PlanStep]                   # 步骤列表
    phase: PlanPhase = PlanPhase.PLANNING   # 当前阶段
    raw_plan_text: str = ""                 # LLM 原始输出
```

**生命周期：**
```
用户输入（Plan 模式）
    ↓
PLANNING: Agent 用只读工具探索代码库，生成结构化计划
    ↓
REVIEWING: parse_plan() 解析步骤 → render_plan_review() 展示计划面板
    ↓                              用户选择 Accept / Modify / Cancel
    ↓ (Accept)
EXECUTING: 切换到 Normal 模式 → 逐步执行每个步骤 → PlanProgressLive 实时更新
    ↓
COMPLETED: render_plan_summary() 展示最终统计
```

---

## 新增能力四：计划解析器

### LLM 输出格式

LLM 在 Plan 模式下被指示生成以下格式：

```markdown
## Plan
**Goal**: <一句话目标>

### Steps
1. **<标题>** -- <详细描述>
2. **<标题>** -- <详细描述>
...
```

### parse_plan 解析流程

```python
# agent/plan_parser.py
def parse_plan(llm_output: str, user_goal: str = "") -> Optional[Plan]:
    # 1. 查找 "## Plan" 或 "## Execution Plan" 区块
    plan_match = re.search(r"(?:##\s*Plan|##\s*Execution\s*Plan)\s*\n(.*?)(?:\n##\s|\Z)", ...)

    if not plan_match:
        return _try_parse_loose(llm_output, user_goal)

    # 2. 提取 Goal（如有）
    goal_match = re.search(r"\*\*Goal\*\*:\s*(.+?)(?:\n|$)", ...)

    # 3. 提取 Steps
    steps = _parse_steps(plan_text)
```

**步骤解析正则** 支持多种格式变体：
- 编号：`1.` 或 `1)`
- 标题：`**bold**` 或纯文本
- 分隔符：`--`、`-`、`:`、`—`（em dash）、`–`（en dash）

### 宽松解析回退

如果未找到正式的 `## Plan` 区块，`_try_parse_loose()` 会在全文中搜索至少 2 个编号步骤，尝试构建 Plan 对象。需要至少 2 个步骤才能成功解析（单个编号行不算计划）。

---

## 新增能力五：Plan 模式提示词

### PromptManager 条件注入

v0.5 扩展了 `PromptManager`，支持 Plan 模式提示词的条件注入：

```python
class PromptManager:
    def __init__(self):
        self._base_prompt = read("agent/prompts/system_prompt.md")
        self._skills_index = ""
        # v0.5: Plan Mode prompt
        self._plan_mode_enabled = False
        self._plan_mode_prompt = read("agent/prompts/plan_prompt.md")

    def get_system_prompt(self) -> str:
        parts = [self._base_prompt]
        if self._plan_mode_enabled and self._plan_mode_prompt:
            parts.append(self._plan_mode_prompt)  # 条件注入
        if self._skills_index:
            parts.append(self._skills_index)
        return "\n\n".join(parts)
```

**提示词层叠结构：**
```
┌──────────────────────┐
│ Base Prompt          │ ← 始终存在
├──────────────────────┤
│ Plan Mode Prompt     │ ← 仅 Plan 模式时注入
├──────────────────────┤
│ Skills Index         │ ← 始终存在
└──────────────────────┘
```

### plan_prompt.md 内容

Plan 模式提示词强制执行两阶段流程和信息完整性要求：

**阶段一 — 探索与信息收集：**
- 使用只读工具（read_file, list_dir, grep_search 等）充分探索代码库
- 读取所有相关文件，搜索已有模式和工具函数，追踪依赖关系
- 记录精确的文件路径、行号、函数名和变量名
- 识别每个需要创建或修改的文件
- 如果用户请求存在歧义，基于代码上下文做出合理推断

**阶段二 — 生成计划：**
- 仅在阶段一完成后才输出计划
- 每个步骤描述必须包含：具体文件路径、行号或函数名、具体变更内容、预期结果
- 禁止出现"确认"、"询问用户"、"根据用户选择"、"待定"等需要运行时交互的措辞
- 步骤数 3-10 个，按依赖排序，标题不超过 60 字符
- 结尾包含验证步骤
- 如果信息确实不足，明确标注假设前提，而非推迟到执行时询问用户

---

## 新增能力六：交互式审阅与可视化

### Plan Review 面板

计划生成后，`render_plan_review()` 展示品红色边框的面板：

```
┌─────────── Plan Mode ───────────┐
│ Goal: 重构认证模块              │
│                                  │
│ Proposed Steps:                  │
│   1. 分析现有认证代码结构       │
│      读取 auth/ 目录下的所有文件 │
│   2. 提取认证逻辑为独立模块     │
│      创建 auth/handler.py        │
│   ...                            │
└──────────────────────────────────┘
```

用户通过方向键选择 **Accept / Modify / Cancel**。

### 修改流程

选择 Modify 后，用户输入反馈，原始计划 + 反馈发送给 LLM 生成修订版。支持最多 **3 轮修改**（防止无限循环）。

### 执行进度实时显示

执行阶段使用 `PlanProgressLive` 上下文管理器，基于 Rich Live 实现 8fps 自动刷新：

```
○  Step 1: Analyze project structure       Pending
●  Step 2: Review existing code            In Progress...
✓  Step 3: Implement changes               Done
```

**状态图标：**
- ○ Pending（暗色）
- ● In Progress（黄色脉冲动画 + Shimmer 光扫效果）
- ✓ Done（绿色）
- ✗ Failed（红色）
- — Skipped（暗色）

**步骤失败处理：** 用户可选择 Continue（跳过继续）或 Stop（跳过所有剩余步骤）。

### Shimmer 动画

`agent/shimmer.py` 提供共享的光扫描动画效果。一个亮点从左到右扫过文本：

```python
def shimmer_positions(text: str, frame: int) -> list[int]:
    """返回每个字符位置的亮度级别（0-3），3 为最高亮度。"""
    cycle = length + 8
    pos = frame % cycle - 4
    return [max(0, 3 - abs(i - pos)) for i in range(length)]
```

在 CLI 中有两种使用场景：
1. **流式输出的 Thinking 动画** — ANSI 转义码实现
2. **进度条 In Progress 文字** — Rich Text 样式实现

### 完成摘要

执行结束后 `render_plan_summary()` 展示统计：

```
┌────────── Plan Complete ──────────┐
│ 3 completed, 0 failed (of 3 total)│
└───────────────────────────────────┘
```

边框颜色：无失败为绿色，有失败为黄色。

---

## Plan 模式完整流程

### 1. 规划阶段

```
用户: /plan（或 Shift+Tab）
  → Agent 进入 Plan Mode
  → PromptManager 注入 plan_prompt.md
  → _get_tools() 仅返回只读工具
  → _execute_tool_calls() 拦截写工具调用

用户: "重构认证模块"
  → Agent 用只读工具探索代码库（read_file, grep_search, tree 等）
  → LLM 生成结构化计划（Goal + Steps）
```

### 2. 审查阶段

```
  → parse_plan() 解析 LLM 输出为 Plan 对象
  → render_plan_review() 展示计划面板
  → 用户选择:
     ├─ Accept → 进入执行阶段
     ├─ Modify → 用户输入反馈 → LLM 修订 → 重新审查（最多 3 轮）
     └─ Cancel → 取消计划
```

### 3. 执行阶段

```
  → 切换到 Normal 模式（所有工具可用）
  → PlanProgressLive 启动实时进度显示
  → 对每个 Step:
     → 构造 step_prompt（包含步骤标题、描述、上下文）
     → agent.run_turn(step_prompt) 静默执行
     → 更新步骤状态（PENDING → IN_PROGRESS → DONE/FAILED）
     → 失败时用户选择 Continue 或 Stop
```

### 4. 完成阶段

```
  → render_plan_summary() 展示统计
  → plan.phase = COMPLETED
```

---

## CLI 变更

### 新增命令

| 命令 | 说明 |
|------|------|
| `/plan` | 切换 Plan / Normal 模式 |
| Shift+Tab | 同上（在输入时即时切换） |

### 变更命令

| 命令 | 变更 |
|------|------|
| `/status` | 新增 Plan Mode 行显示当前模式 |
| `/help` | 新增 `/plan` 说明 |

### Banner 增强

Plan 模式激活时 Banner 显示 `Mode: PLAN (read-only)`。

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `agent/plan.py` | **新增** | Plan 数据结构（PlanPhase, StepStatus, PlanStep, Plan） |
| `agent/plan_parser.py` | **新增** | 计划解析器，从 LLM 输出中提取结构化计划 |
| `agent/prompts/plan_prompt.md` | **新增** | Plan 模式专用提示词（指导 LLM 生成计划格式） |
| `agent/prompts/system_prompt.md` | 移动 | 迁移至 prompts/ 目录（原在 agent/ 下） |
| `agent/plan_input.py` | **新增** | InputHandler，prompt_toolkit 输入处理 + Shift+Tab 切换 |
| `agent/plan_renderer.py` | **新增** | Plan Review 渲染器 + 实时进度条 |
| `agent/shimmer.py` | **新增** | 共享光扫描动画位置计算器 |
| `agent/agent.py` | 修改 | 新增 `_plan_mode` 状态、`set_plan_mode()` 方法、执行层写工具拦截 |
| `agent/cli.py` | 修改 | `/plan` 命令、Plan 模式提示符、计划审阅与执行流程 |
| `agent/prompt.py` | 修改 | `set_plan_mode()` 条件注入 plan_prompt |
| `tools/registry.py` | 修改 | `_TOOL_CATEGORIES` 字典、`get_read_only_names()`、`is_write_tool()` |
| `pyproject.toml` | 修改 | 版本号 0.5.0，新增 `prompt-toolkit` 依赖 |
| `main.py` | 修改 | 版本注释更新为 v0.5 |
