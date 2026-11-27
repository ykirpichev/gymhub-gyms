# Planner/executor boundary

A planner model wrote a plan; your executor runs it. Write `solve(plan, tools, max_steps)`.

- `plan`: a list of steps `{"id": str, "tool": str, "args": dict}`. Ids are non-empty strings of letters, digits and `_`.
- `tools`: a dict from tool name to a Python function, called as `tools[name](**args)`.
- `max_steps`: an `int`, the largest plan the executor accepts.

**References.** A string that is exactly `"$<id>.output"` anywhere inside `args` (an argument value, a list item, or a dict value, at any depth) stands for the output of step `<id>`. Any other string, such as `"total: $a.output"`, is a literal.

**Validate first** and raise `ValueError` before running anything if `len(plan) > max_steps`, two steps share an id, or a reference names a step that does not appear **earlier** in the plan (unknown, the step itself, or a later step).

**Execute** the steps in plan order. Each step adds one entry to the trace:

| Situation | Trace entry |
| --- | --- |
| a referenced step's status is not `"ok"` | `{"id", "tool", "status": "skipped", "blocked_by": [...]}`: the ids of those referenced steps, sorted, without duplicates. The tool is not called. |
| `tool` is not in `tools` | `{"id", "tool", "status": "error", "error": "unknown tool: <tool>"}` |
| the tool raises `e` | `{"id", "tool", "status": "error", "error": f"{type(e).__name__}: {e}"}` |
| otherwise | `{"id", "tool", "status": "ok", "output": <return value>}` |

Steps with no failed references still run after a failure (partial failure). Return the trace, a list with one entry per step in plan order.

```
plan = [{"id": "s1", "tool": "fetch", "args": {"url": "a.com"}},
        {"id": "s2", "tool": "words", "args": {"text": "$s1.output"}},
        {"id": "s3", "tool": "fetch", "args": {"url": "c.com"}},
        {"id": "s4", "tool": "count", "args": {"items": "$s3.output"}},
        {"id": "s5", "tool": "count", "args": {"items": "$s2.output"}},
        {"id": "s6", "tool": "add", "args": {"a": "$s5.output", "b": "$s4.output"}}]
# fetch("a.com") returns "alpha beta", fetch("c.com") raises LookupError("404 c.com"),
# words(text) = text.split(), count(items) = len(items), add(a, b) = a + b
solve(plan, tools, 6) ->
[{"id": "s1", "tool": "fetch", "status": "ok", "output": "alpha beta"},
 {"id": "s2", "tool": "words", "status": "ok", "output": ["alpha", "beta"]},
 {"id": "s3", "tool": "fetch", "status": "error", "error": "LookupError: 404 c.com"},
 {"id": "s4", "tool": "count", "status": "skipped", "blocked_by": ["s3"]},
 {"id": "s5", "tool": "count", "status": "ok", "output": 2},
 {"id": "s6", "tool": "add", "status": "skipped", "blocked_by": ["s4"]}]
```

The tests also record every tool call, so calling a tool for a skipped step fails even if the trace looks right.

**Interview focus:** validating the whole plan before acting, a bounded step count, explicit data dependencies, partial failure that does not cascade into independent steps, and a structured trace.

**Follow-ups** to answer out loud once your solution passes:

- How would you run independent steps in parallel?
- A step fails halfway through. When should the agent re-plan, and when stop?
- How do you stop a plan from calling a dangerous tool?
