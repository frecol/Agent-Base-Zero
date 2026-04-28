# v0.4 更新详解：技能系统、PromptManager 与组合工具

v0.4 在 v0.3 的"会话持久化 + 长期记忆 + 上下文压缩"基础上，引入 **技能系统（Skill System）**。核心思想是让 Agent 具备"角色切换"能力——不同任务激活不同专业技能，每个技能自带专属指令和领域知识。三大能力：**Skill 自动发现与注册**、**PromptManager 动态提示词管理**、**LLM 自主激活技能 + 组合工具**。

---

## 目录

- [变更总览](#变更总览)
- [新增能力一：Skill 自动发现与注册](#新增能力一skill-自动发现与注册)
  - [SKILL.md 标准格式](#skillmd-标准格式)
  - [SkillInfo 数据类](#skillinfo-数据类)
  - [SkillRegistry 注册表](#skillregistry-注册表)
  - [discover_skills 发现流程](#discover_skills-发现流程)
  - [build_skills_index 索引构建](#build_skills_index-索引构建)
- [新增能力二：PromptManager](#新增能力二promptmanager)
  - [从硬编码到文件管理](#从硬编码到文件管理)
  - [提示词组装逻辑](#提示词组装逻辑)
  - [缓存友好的设计](#缓存友好的设计)
- [新增能力三：LLM 自主激活技能](#新增能力三llm-自主激活技能)
  - [activate_skill 工具](#activate_skill-工具)
  - [deactivate_skill 工具](#deactivate_skill-工具)
  - [激活流程详解](#激活流程详解)
- [新增能力四：组合工具](#新增能力四组合工具)
  - [research_topic 组合工具](#research_topic-组合工具)
  - [工具编排模式](#工具编排模式)
- [示例技能](#示例技能)
  - [research — 搜索研究专家](#research--搜索研究专家)
  - [code_assistant — 代码助手专家](#code_assistant--代码助手专家)
- [CLI 新增命令](#cli-新增命令)
- [文件变更清单](#文件变更清单)

---

## 变更总览

| 维度 | v0.3 | v0.4 |
|------|------|------|
| 提示词管理 | 硬编码在 `client.py` | `PromptManager` 从文件加载，支持动态拼接 |
| 技能系统 | 无 | SKILL.md 标准格式，自动发现，LLM 自主激活 |
| 组合工具 | 无 | `research_topic` 链式编排多个工具 |
| CLI 命令 | 11 个 | 新增 3 个（`/skills`、`/skill`、`/unskill`） |
| /status | Session + Token | 增加 Active Skill 显示 |
| 新增目录 | 0 | 1 个（`skills/`） |
| 新增文件 | 0 | 6 个（2 核心 + 2 技能 + 1 工具 + 1 提示词） |

---

## 新增能力一：Skill 自动发现与注册

### SKILL.md 标准格式

每个 Skill 是 `skills/` 目录下的一个子文件夹，包含一个 `SKILL.md` 文件。采用与 Claude Code 等生态兼容的 **YAML frontmatter** 格式（`---` 包裹）：

```markdown
---
name: research
description: "搜索研究专家，擅长网络搜索、信息提取和综合分析。"
user_invocable: true
---

You are now operating in Research Mode...
（详细指令内容）
```

**frontmatter 字段：**
- `name`（必需）：技能标识符
- `description`（必需）：技能描述，会写入 system prompt 供 LLM 判断是否激活
- `user_invocable`（可选，默认 true）：用户是否可直接调用

**frontmatter 之后的内容**是技能的详细指令（Markdown 格式），在技能激活时通过 `activate_skill` 工具返回给 LLM。

### SkillInfo 数据类

```python
# skills/registry.py
@dataclass
class SkillInfo:
    name: str
    description: str
    prompt_text: str         # SKILL.md 中 frontmatter 之后的完整内容
    user_invocable: bool
    skill_dir: Optional[Path]
```

### SkillRegistry 注册表

遵循项目已有的注册表模式（与 `ToolRegistry` 对称）：

```python
class SkillRegistry:
    _skills: Dict[str, SkillInfo]
    _active_skill: Optional[str]
    _on_activate: Optional[Callable]

    def register(skill_info)           # 注册一个 skill
    def activate(name) -> str          # 激活，返回 JSON（含 prompt_text）
    def deactivate() -> str            # 取消激活
    def get_active() -> SkillInfo?     # 获取当前激活的 skill
    def get_all_names() -> List[str]   # 所有已注册 skill 名称
    def get_skill(name) -> SkillInfo?  # 按名称查询
    def set_on_activate(callback)      # 注册状态变更回调
```

全局单例 `skill_registry`。

### discover_skills 发现流程

```python
def discover_skills(skills_dir=None) -> List[str]:
    skills_path = skills_dir or Path(__file__).resolve().parent
    for item in sorted(skills_path.iterdir()):
        if not item.is_dir(): continue
        if item.name.startswith(("_", ".")): continue

        skill_md = item / "SKILL.md"
        if not skill_md.exists(): continue

        info = _parse_skill_md(skill_md)   # 解析 frontmatter + 内容
        skill_registry.register(info)

    _register_skill_tools()  # 注册 activate_skill / deactivate_skill
```

**frontmatter 解析**：使用正则 `^---\s*\n(.*?)\n---\s*\n` 提取 YAML 区块，逐行解析 `key: value` 对。支持引号包裹和布尔值，无需 `pyyaml` 依赖。

### build_skills_index 索引构建

```python
def build_skills_index() -> str:
    # 输出格式：
    # ## Available Skills
    # When a task matches a skill below, call activate_skill(name)...
    #
    # - research: 搜索研究专家...
    # - code_assistant: 代码助手专家...
```

在启动时调用，拼接到 system prompt 末尾。

---

## 新增能力二：PromptManager

### 从硬编码到文件管理

v0.3 的 `SYSTEM_PROMPT` 是一个 Python 字符串常量写在 `agent/client.py` 中。v0.4 将其提取到独立文件：

```
agent/client.py (SYSTEM_PROMPT 常量) → agent/system_prompt.md (独立文件)
                                  + agent/prompt.py (PromptManager 类)
```

`client.py` 不再持有任何提示词知识，变成纯粹的 API wrapper。

### 提示词组装逻辑

```python
class PromptManager:
    def __init__(self, base_prompt_path=None):
        self._base_prompt = read("agent/system_prompt.md")
        self._skills_index = ""

    def update_skills_index(self, skills_index):
        # 启动时调用一次
        self._skills_index = skills_index

    def get_system_prompt(self) -> str:
        return self._base_prompt + "\n\n" + self._skills_index
```

**组装结果：**
```
[base prompt from system_prompt.md]

## Available Skills
When a task matches a skill below, call activate_skill(name)...

- research: 搜索研究专家...
- code_assistant: 代码助手专家...
```

### 缓存友好的设计

**关键原则：system prompt 前缀在整个会话中保持不变。**

1. **启动时**：`base prompt` + `skills index` 拼接完成 → 前缀固定
2. **激活技能时**：技能详细指令作为 `activate_skill` 的 **tool result** 返回，进入对话历史
3. **system prompt 不变** → 提示词前缀缓存始终命中

```
system prompt（稳定前缀，不随 skill 切换而变化）:
┌─────────────────────────────┐
│ base prompt                 │ ← agent/system_prompt.md
│ + skills index（名称+描述） │ ← 启动时加载
├─────────────────────────────┤ ← 缓存边界
│ memory system message       │ ← v0.3 已有
│ + conversation history      │ ← 动态增长
│ + skill tool results        │ ← 技能指令通过这里进入
└─────────────────────────────┘
```

---

## 新增能力三：LLM 自主激活技能

### activate_skill 工具

```python
SCHEMA = {
    "name": "activate_skill",
    "description": "Activate a skill by name. Returns the skill's detailed instructions.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill_name": {"type": "string", "description": "技能名称"}
        },
        "required": ["skill_name"],
    },
}
```

**handler 逻辑：**
1. 调用 `skill_registry.activate(name)` 设置激活状态
2. 触发回调通知 Agent（状态变更）
3. 返回技能的完整 `prompt_text` 作为 **tool result**

### deactivate_skill 工具

```python
SCHEMA = {
    "name": "deactivate_skill",
    "description": "Deactivate the currently active skill and return to base mode.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
```

### 激活流程详解

```
用户: "帮我搜索 Python 最新版本"
  → _build_messages(): system prompt 不变（包含 skills index）
  → LLM 看到 skills index 中 "research: 搜索研究专家..."
  → LLM 调用 activate_skill("research")
  → handler 返回 research/SKILL.md 的完整内容作为 tool result
  → tool result 进入对话历史（system prompt 不变，缓存不破）
  → LLM 按照收到的 skill 指令调用 web_search 等工具
  → 最终回复用户
```

**回调机制：** 工具 handler 调用 `skill_registry._on_activate(name, prompt_text)`，Agent 注册此回调。当前实现中回调体为空（pass），因为 `_get_tools()` 每次调用时自动读取 `skill_registry` 的最新状态。但回调机制为未来版本（如工具过滤）保留了扩展点。

**CLI 手动激活：** 用户通过 `/skill <name>` 手动激活时，`Agent.activate_skill()` 会构造一个模拟的 assistant tool_call + tool result 消息对注入 `self.messages`，确保 LLM 收到技能指令。这与 LLM 自主调用 `activate_skill` 工具的效果完全一致——指令都通过 tool result 进入对话历史，system prompt 保持不变。此外，`user_invocable: false` 的技能会被 CLI 拒绝手动激活，仅允许 LLM 自动激活。

---

## 新增能力四：组合工具

### research_topic 组合工具

组合工具是 `tools/` 目录中的普通工具，其 handler 通过 `registry.dispatch()` 编排多个已有工具：

```python
# tools/research_topic.py
def handler(args: dict) -> str:
    query = args.get("query", "")
    max_sources = args.get("max_sources", 3)

    # Step 1: 搜索
    search_raw = registry.dispatch("web_search", {"query": query})
    search_result = json.loads(search_raw)

    # Step 2: 获取全文
    for r in search_result.get("results", [])[:max_sources]:
        fetch_raw = registry.dispatch("fetch_url", {"url": r["href"]})
        # ... 整合结果

    return json.dumps({"success": True, "sources": fetched, ...})
```

### 工具编排模式

组合工具的核心模式：

1. 定义 `SCHEMA` 和 `handler`，与普通工具完全一致
2. handler 内部通过 `registry.dispatch("tool_name", args)` 调用其他工具
3. 将多个工具的结果整合为更高层级的返回值
4. 通过 `registry.register()` 自注册，对 LLM 来说就是一个新工具

**优势：** 无需新的抽象或框架，复用现有工具注册和分发机制。开发者只需写一个 handler 函数来编排已有工具。

---

## 示例技能

### research — 搜索研究专家

```
skills/research/
└── SKILL.md
```

**指令要点：**
- 搜索策略（先广后窄、不重复查询）
- 信息提取和综合分析方法
- 引用来源规范
- 判断何时搜索 vs 已有知识
- 推荐使用 `research_topic` 组合工具处理复杂研究任务

### code_assistant — 代码助手专家

```
skills/code_assistant/
└── SKILL.md
```

**指令要点：**
- 先读后改的工作流
- 调试指南（理解错误 → 定位代码 → 验证修复）
- 代码风格（跟随项目现有风格）
- 工具偏好（edit_file > write_file，tree 了解结构等）

---

## CLI 新增命令

| 命令 | 说明 |
|------|------|
| `/skills` | 列出所有可用技能（标记当前激活的，`user_invocable: false` 显示为 `(auto-only)`） |
| `/skill <name>` | 手动激活指定技能（`user_invocable: false` 的技能拒绝手动激活） |
| `/unskill` | 取消当前技能，回到基础模式 |

`/status` 增强：新增 `Active Skill` 行显示当前激活的技能名称。

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `agent/prompt.py` | **新增** | PromptManager，从文件加载 base prompt + 拼接 skills index |
| `agent/system_prompt.md` | **新增** | 基础系统提示词（从 client.py 提取） |
| `skills/__init__.py` | **新增** | 自动发现入口，调用 discover_skills() |
| `skills/registry.py` | **新增** | SkillRegistry、SKILL.md 解析、activate_skill/deactivate_skill 工具 |
| `skills/research/SKILL.md` | **新增** | 搜索研究技能 |
| `skills/code_assistant/SKILL.md` | **新增** | 代码助手技能 |
| `tools/research_topic.py` | **新增** | 组合工具：web_search + fetch_url 链式编排 |
| `agent/client.py` | 修改 | 删除 SYSTEM_PROMPT 常量 |
| `agent/agent.py` | 修改 | PromptManager 集成、skill 回调、activate_skill/deactivate_skill 方法（CLI 激活注入指令到对话） |
| `agent/cli.py` | 修改 | 新增 /skills /skill /unskill 命令、/status 增加 Active Skill、版本 v0.4 |
| `main.py` | 修改 | 新增 import tools / import skills 自动发现 |
| `pyproject.toml` | 修改 | 版本号 0.4.0，包发现新增 skills* |
