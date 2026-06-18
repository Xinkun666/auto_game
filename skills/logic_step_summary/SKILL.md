---
name: logic_step_summary
description: Use when the user asks to summarize code logic, business flow,方案逻辑, branch paths, retry/fallback behavior, or says to explain in plain language without losing details.
---

# logic_step_summary

## Skill Purpose

Use this skill to turn code logic, algorithm flow, automation scripts, game AI behavior, test flows, or discussed方案 into a clear step-by-step summary.

The output should feel like plain Chinese explanation, but it must keep the important technical details: trigger conditions, thresholds, states, variables, function names, parameters, branch order, retries, fallback handling, and final success/failure outcomes.

## When to Use

Use this skill when the user says things like:

- “总结一下这部分逻辑”
- “把这段代码流程整理一下”
- “按白话但不丢细节的方式总结”
- “把这个方案按步骤重新整理”
- “把分支逻辑闭环整理出来”
- “像之前搜房/进房逻辑那样整理”
- “白话一点，但细节别丢”

Also use it for:

- Code execution flow.
- Business/algorithm branch logic.
- Game AI, automation script, test process, image-processing process.
- Final scheme整理 after multi-turn discussion.
- Retry, fallback, failure, and state-transition summaries.

## Required Output Shape

Start naturally, for example:

```text
收到，我按“白话但不丢细节”的方式，把这部分逻辑重新整理一遍：
```

or:

```text
收到，这样分支基本闭环了。我现在按最终版重新整理：
```

Then divide the logic into business-readable stages using bold “第几段” headings:

```markdown
**第一段：靠近目标**

1. ...
2. ...

**第二段：进入核心判断范围**

1. ...
2. ...
```

Headings must describe business meaning. Do not use only function names as headings.

Bad:

```markdown
**第一段：process_target()**
```

Good:

```markdown
**第一段：靠近目标并进入处理范围**

1. 这部分主要对应 `process_target()`。
```

## Step Format

Inside each stage, use numbered steps:

```markdown
1. 当距离 `>15` 时：
   - 继续普通导航。
   - 不进入特殊处理。
2. 当距离进入 `<=15` 时：
   - 停止自动前进。
   - 重新计算朝向。
   - 进入靠近目标逻辑。
```

Use sub-bullets for conditions, branches, or grouped actions. Avoid long paragraphs when a branch list would be clearer.

## Details That Must Be Preserved

Keep important code details and wrap them in backticks:

- Variables: `house_scene`, `fail_count`
- States/enums: `NEAR_WALL`, `INDOOR`, `OUTDOOR`, `SCENE_ENTRY`
- Functions: `refresh_frame()`, `process_target()`
- Parameters: `y_bias=-350, dura=100, wait=3000`
- Thresholds: `<=4`, `15~10`, `>3`
- Return values: `True / False`, `"success"`, `"retry"`
- Counters and limits: `max_steps=3`, `large_backoff_count > 3`

Do not replace precise logic with vague phrases.

Bad:

```markdown
1. 距离比较近时，往前走一下。
2. 失败后再处理。
```

Good:

```markdown
1. 当距离进入 `<=4` 时：
   - 锁定当前目标。
   - 状态切到 `SCENE_ENTRY`。
2. 前推使用 `y_bias=-350, dura=100, wait=3000`。
3. 如果 `house_scene == NEAR_WALL`：
   - 认为撞墙。
   - 进入小后拉或大后拉兜底。
```

## Explain the Intent, Not Just the Code

Do not only translate code line by line. Explain why the step exists.

Bad:

```markdown
调用 `refresh_frame()`，然后判断 `house_scene`。
```

Good:

```markdown
前推后先 `refresh_frame()`，是为了拿到最新画面，避免用旧画面误判。然后再判断 `house_scene`：
- 如果已经是 `INDOOR`：说明进房成功。
- 如果还是 `OUTDOOR`：继续找门/窗或走兜底逻辑。
```

## Branch Coverage Rules

When there is `if / elif / else`, loop, retry, fallback, timeout, or state transition, cover all meaningful branches:

1. Normal success path.
2. Condition not met path.
3. Failure/error path.
4. Retry behavior.
5. Retry limit.
6. What happens after exceeding the limit.
7. Whether the current target/state is marked success or failure.
8. Whether the logic switches to the next target/stage.

If branch order matters, state the order explicitly:

```markdown
判断顺序是：
1. 先判断是否已经成功。
2. 再判断是否还能继续尝试。
3. 最后才进入失败兜底。
```

If a priority is highest, say it directly:

```markdown
任何时候只要检测到 `INDOOR`，最高优先级：直接认为进房成功，启动搜房。
```

## Suggested Summary Workflow

When summarizing:

1. Identify the entry point:
   - Which function, state, user action, event, or timer starts this logic.
2. Identify the main stages:
   - Approach/precondition.
   - Core decision range.
   - Main branch selection.
   - Normal success path.
   - Failure/fallback path.
   - Retry/final exit.
3. Extract exact details:
   - Thresholds, states, function calls, parameters, counters, return values.
4. Convert each branch into plain-language steps:
   - “当 `xxx` 时...”
   - “如果 `xxx`，说明...”
   - “失败后先...再...超过后...”
5. Add uncertainty notes where needed.
6. End with a short “整体理解” if the logic is complex.

## Handling Uncertainty

Do not invent behavior that is not visible in the code or provided context.

Use wording like:

```markdown
这里代码里没有完全展开，暂时只能理解为：
- `xxx()` 可能负责具体的重试。
- 具体是否会标记失败，还需要继续看 `xxx()` 内部实现。
```

or:

```markdown
这一点需要待确认：当前上下文里没有看到超过次数后的最终处理。
```

If the user asks for a final summary but some details are uncertain, still summarize the known flow and clearly mark uncertain parts.

## Output Template

Use this as the default shape, adjusting stage names to the actual logic:

```markdown
收到，我按“白话但不丢细节”的方式重新整理：

**第一段：xxx**

1. xxx。
2. 当 `xxx` 时：
   - xxx。
   - xxx。
3. 如果 `xxx`：
   - xxx。
   - xxx。

**第二段：xxx**

1. xxx。
2. 判断顺序是：
   - 情况 A：xxx。
   - 情况 B：xxx。
   - 情况 C：xxx。

**第三段：异常/失败兜底**

1. xxx。
2. 如果失败：
   - 先 xxx。
   - 再 xxx。
   - 超过 `N` 次后 xxx。

**整体理解：**

这套逻辑本质上是：xxx。
```

## Positive Example

```markdown
收到，这样分支基本闭环了。我现在按最终版重新整理：

**第一段：靠近房子**

1. 人物和某个房子标记点距离 `>15`：继续普通导航。
2. 距离进入 `<=15`：
   - 停止自动前进。
   - 先朝“当前位置 -> 房子标记点”的角度对准房子。
3. 距离 `15~10`：
   - 快速推进，推进时间加长。
   - 每次推进后优先判断是否已经在室内。
4. 距离 `10~5`：
   - 慢速推进，推进时间也加长。
   - 如果 `house_scene == NEAR_WALL`，认为撞墙。
5. 推进中撞墙：
   - 如果已经在室内：直接启动搜房。
   - 如果在室外：走室外避障/后拉/绕行逻辑。
```

## Negative Example

Avoid summaries that hide important branches:

```markdown
这段代码就是先靠近房子，然后找门窗，能进去就进去，不行就换一个。
```

Why this is bad:

- No thresholds.
- No states.
- No retry/failure logic.
- No branch order.
- No function/parameter details.
- The user cannot tell what actually triggers each step.

## Final Quality Checklist

Before answering, check:

- Did the summary use “第几段” business-readable headings?
- Are steps numbered with `1. 2. 3.`?
- Are key variables, states, thresholds, functions, and parameters preserved?
- Are success, failure, retry, and final exit branches covered?
- Is priority/order explicit where it matters?
- Did you explain intent in plain language?
- Did you avoid inventing uncertain behavior?
- Is there a short “整体理解” when the logic is complex?
