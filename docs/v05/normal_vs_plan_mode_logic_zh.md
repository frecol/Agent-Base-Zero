# Agent 运行逻辑详解

本文档详细描述 Agent-Base-Zero v0.5 的运行逻辑，包括主循环流程、模式切换机制、Normal / Plan 双模式的回复路径。

---

## 主循环流程

CLI 入口为 `agent/cli.py` 的 `run_cli()` 函数。启动时初始化三个核心对象：

```
run_cli() 启动
  ├── PromptManager    加载 base prompt + plan_prompt.md + skills index
  ├── Agent            核心 Agent，持有对话历史和 LLM 客户端
  ├── InputHandler     输入处理器，持有 _plan_mode 状态（默认 False = Normal）
  │
  ├── stream_enabled   = settings.stream_enabled   （默认 True）
  ├── thinking_enabled = settings.thinking_enabled  （默认 True）
  │
  └── while True:  主循环
```

### 每轮循环的 6 个步骤

```
while True:

  ① 记录当前模式
     mode_before = input_handler.plan_mode

  ② 等待用户输入
     user_input = input_handler.get_input()
       ├── prompt_toolkit 渲染提示符:
       │     Normal → "You> "  (绿色)
       │     Plan   → "Plan> " (品红色)
       └── 如果用户按了 Shift+Tab:
             _plan_mode 翻转
             get_input() 返回 ""

  ③ 检测模式变化（Shift+Tab 触发）
     if input_handler.plan_mode != mode_before:
       → agent.set_plan_mode(...)
       → 打印模式切换提示
       → continue  ← 跳过本轮，不处理任何输入

  ④ 空输入检查
     if not user_input: continue

  ⑤ 内置命令处理
     /exit, /help, /plan, /status ... 各自处理并 continue

  ⑥ 对话处理（核心分支，见下文）
```

**模式检测不是主动轮询**，而是 Shift+Tab 让 `get_input()` 提前返回空串，第③步发现模式变了就跳过本轮。

---

## 工具过滤机制

`Agent._get_tools()` 根据当前模式决定传递给 LLM 的工具集：

```
_get_tools()
  │
  ├── _plan_mode = True  (Plan 模式)
  │   └── registry.get_read_only_names()
  │       → read_file, list_dir, tree, find_file, grep_search
  │       → web_search, fetch_url, research_topic
  │       → current_time, system_info, session_search
  │       （共 11 个只读工具）
  │
  └── _plan_mode = False (Normal 模式)
      └── registry.get_all_names()
          → 以上 11 个 + write_file, edit_file, file_delete
          → run_command, memory_save, activate_skill, deactivate_skill
          （共 18 个全部工具）
```

**双重防护**：除了 API 层面不传递写工具定义外，`_execute_tool_calls()` 在执行层还有硬防护 —— Plan 模式下遇到写工具调用直接拦截并返回错误信息给 LLM。

---

## 提示词组装

`PromptManager.get_system_prompt()` 根据模式条件组装系统提示词：

```
Normal 模式:
  [base prompt from system_prompt.md]
  [skills index]

Plan 模式:
  [base prompt from system_prompt.md]
  [plan_prompt.md]              ← 条件注入，指导 LLM 生成结构化计划
  [skills index]
```

---

## Normal 模式回复逻辑

参考代码：`cli.py` 第 639-660 行

```
Normal Mode
  │
  ├── stream_enabled = True:
  │   └── _stream_response(agent, user_input, thinking_enabled)
  │       │
  │       └── agent.run_turn_stream(user_input, ...)
  │           │
  │           │  前置:
  │           │  ├── messages.append({"role": "user", "content": user_input})
  │           │  ├── tools = _get_tools()            → 所有 18 个工具
  │           │  └── api_messages = _build_messages() → base prompt + skills index
  │           │
  │           │  进入迭代循环 (最多 50 次):
  │           │  │
  │           │  ├── client.chat_stream(api_messages, tools, thinking_enabled)
  │           │  │
  │           │  │  累积流式 chunks:
  │           │  │  ├── reasoning_content  → yield thinking event
  │           │  │  │   CLI 显示: shimmer "Thinking..." 动画
  │           │  │  │
  │           │  │  ├── content            → yield content event
  │           │  │  │   CLI 逐 token 输出到终端
  │           │  │  │
  │           │  │  └── tool_calls deltas  → 累积完整 tool call
  │           │  │
  │           │  ├── 流结束后检查:
  │           │  │   │
  │           │  │   ├── 有 tool_calls:
  │           │  │   │   ├── messages.append(assistant_msg + tool_calls)
  │           │  │   │   ├── _execute_tool_calls(...)
  │           │  │   │   │   对每个 tool_call:
  │           │  │   │   │   ├── on_tool_call(name, args)  → CLI 显示工具调用面板
  │           │  │   │   │   ├── registry.dispatch(name, args) → 执行工具
  │           │  │   │   │   └── messages.append(tool_result)
  │           │  │   │   └── continue → 回到迭代循环开头，把工具结果发给 LLM
  │           │  │   │
  │           │  │   └── 无 tool_calls (纯文本):
  │           │  │       ├── messages.append(assistant_msg)
  │           │  │       ├── _post_turn() → 压缩检查 + 会话保存
  │           │  │       └── yield done event → 结束
  │           │
  │           └── 达到 50 次上限:
  │               └── yield done (带最大迭代提示)
  │
  └── stream_enabled = False:
      └── agent.run_turn(user_input, ...)
          │  阻塞式调用 client.chat()
          │  同样的 tool_calls 循环逻辑，但非流式
          └── 打印 Panel(Markdown(response), title="Assistant", border="blue")
```

---

## Plan 模式回复逻辑

参考代码：`cli.py` 第 603-638 行

Plan 模式分三个阶段：**规划** → **审阅** → **执行**。

### 阶段一：规划（Plan mode，只读探索）

```
Plan Mode — 规划阶段
  │
  ├── stream_enabled = True:
  │   └── _stream_response(agent, user_input, thinking_enabled)
  │       └── agent.run_turn_stream(user_input, ...)
  │           │  tools = _get_tools()            → 只有 11 个只读工具
  │           │  api_messages = _build_messages() → base + plan_prompt + skills index
  │           │  _execute_tool_calls 硬防护      → 写工具调用被拦截
  │           │
  │           └── LLM 用 read_file, grep_search 等探索代码库
  │              最终输出包含结构化计划的文本
  │
  │   response = _get_last_assistant_content(agent.messages)
  │
  ├── stream_enabled = False:
  │   └── agent.run_turn(user_input, ...)
  │       └── 打印 Panel(Markdown(response), title="Plan", border="magenta")
  │
  └── 解析计划:
      plan = parse_plan(response, user_input)
```

### 阶段二：审阅（用户交互）

```
parse_plan 有结果?
  │
  ├── Yes (解析出带步骤的结构化计划):
  │   └── _handle_plan_review(plan, input_handler, agent, ...)
  │       │
  │       ├── render_plan_review(plan, console)
  │       │   → 展示品红色 Plan 面板（Goal + Steps）
  │       │
  │       └── select_option(["Accept", "Modify", "Cancel"])
  │           │
  │           ├── Accept (0):
  │           │   └── return plan  → 进入执行阶段
  │           │
  │           ├── Modify (1):
  │           │   ├── 获取用户反馈文本
  │           │   ├── 构造 revise_prompt = 原始计划 + 用户反馈
  │           │   ├── agent.run_turn(revise_prompt)
  │           │   │   → LLM 生成修订版计划
  │           │   ├── parse_plan(修订版响应)
  │           │   └── 递归 _handle_plan_review(depth + 1)
  │           │       → 最多 3 轮修改，防止无限循环
  │           │
  │           └── Cancel (2) / Ctrl+C:
  │               └── return None  → 计划丢弃
  │
  └── No (解析不出计划):
      └── 静默跳过（不显示警告）
```

### 阶段三：执行（切回 Normal mode）

```
plan 被接受?
  │
  ├── Yes:
  │   └── _execute_plan(plan, agent, thinking_enabled, input_handler)
  │       │
  │       ├── 切换回 Normal 模式:
  │       │   input_handler.plan_mode = False
  │       │   agent.set_plan_mode(False)    → 恢复所有工具
  │       │   plan.phase = EXECUTING
  │       │
  │       ├── 打印 "Executing Plan" 绿色分隔线
  │       │
  │       └── with PlanProgressLive(plan):  ← Rich Live 8fps 进度条
  │           │
  │           └── for step in plan.steps:
  │               │
  │               ├── step.status = IN_PROGRESS
  │               │
  │               ├── 构造 step_prompt:
  │               │   "Execute Step {n}: {title}
  │               │    Details: {description}
  │               │    Context: step {n} of {total} in plan to '{goal}'
  │               │    Perform ONLY this step. Be concise."
  │               │
  │               ├── agent.run_turn(step_prompt, thinking_enabled)
  │               │   ← 注意: 无 on_tool_call / on_tool_result 回调
  │               │   ← 工具调用静默执行，用户只看到进度条
  │               │   ← _get_tools() 已恢复返回所有 18 个工具
  │               │
  │               ├── 成功:
  │               │   step.status = DONE
  │               │   打印一行结果摘要（截断到 120 字符）
  │               │
  │               └── 失败:
  │                   step.status = FAILED
  │                   select_option(["Continue", "Stop"])
  │                   ├── Continue → 跳过当前步，继续下一步
  │                   └── Stop → 所有 PENDING 步骤标记 SKIPPED, break
  │
  └── render_plan_summary(plan, console)
      → "X completed, Y failed, Z skipped (of N total)"
      → 无失败: 绿色边框
      → 有失败: 黄色边框
```

---

## 总流程图

```
                       ┌──────────────────────┐
                       │    while True 循环     │
                       └──────────┬───────────┘
                                  │
                     ① mode_before = plan_mode
                     ② user_input = get_input()
                                  │
                     ③ 模式变了? ──Yes──→ 同步 Agent 状态, continue
                                  │ No
                     ④ 空输入? ────Yes──→ continue
                                  │ No
                     ⑤ /命令? ────Yes──→ 处理命令, continue
                                  │ No
                     ⑥ plan_mode?
                       ┌──────────┴──────────┐
                       │                     │
                    False                  True
                       │                     │
              ┌────────┴────────┐   ┌────────┴─────────────────┐
              │   Normal Mode   │   │   Plan Mode               │
              │                 │   │                           │
              │  _get_tools():  │   │  _get_tools():            │
              │  所有 18 个工具  │   │  只有 11 个只读工具        │
              │                 │   │  + 执行层写工具硬防护       │
              │  提示词:        │   │                           │
              │  base + skills  │   │  提示词:                  │
              │                 │   │  base + plan_prompt        │
              │                 │   │  + skills                  │
              │                 │   │                           │
              │  run_turn /     │   │  run_turn / stream         │
              │  stream         │   │  (LLM 只读探索 + 生成计划)  │
              │                 │   │         ↓                  │
              │  实时展示:      │   │  parse_plan()              │
              │  thinking 动画  │   │         ↓                  │
              │  工具调用面板    │   │  review: Accept / Modify   │
              │  逐 token 输出  │   │         ↓ (Accept)         │
              │  最终响应面板    │   │  切回 Normal Mode           │
              │                 │   │  _get_tools() → 全部工具    │
              │                 │   │         ↓                  │
              │                 │   │  for step in plan.steps:    │
              │                 │   │    run_turn(step) 静默执行   │
              │                 │   │    进度条实时更新            │
              │                 │   │         ↓                  │
              │                 │   │  summary 面板              │
              └─────────────────┘   └────────────────────────────┘
```

---

## 关键设计要点

1. **模式切换时机**: Shift+Tab 不是主动轮询检测，而是让 `get_input()` 提前返回空串，下一轮循环开头的第③步捕获变化
2. **Plan 模式的只读安全**: API 层面不传写工具定义 + 执行层硬防护拦截写工具调用，双重保障
3. **执行阶段静默**: Plan 执行时每个 step 的 `run_turn()` 不传 `on_tool_call` 回调，工具调用不可见，用户只看到进度条
4. **执行后自动恢复**: `_execute_plan()` 开头将模式切回 Normal，执行期间所有工具可用
5. **迭代上限**: 单次 `run_turn` 内最多 50 轮工具调用循环，防止无限循环
6. **Plan 修改轮数上限**: 审阅阶段最多 3 轮修改，防止无限修改循环
