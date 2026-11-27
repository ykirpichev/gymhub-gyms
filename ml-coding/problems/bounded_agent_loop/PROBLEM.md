# Bounded agent tool-use loop

Write `solve(script, tools, max_steps, loop_limit)`.

`script` is a list of model responses, taken in order, and it does not branch on observations:

- `{"type": "tool", "tool": str, "args": dict}` proposes one call, made as `tools[name](**args)`
- `{"type": "final", "answer": <JSON value>}` finishes. The answer may be `None`.

`max_steps` is the most responses you may consume. `loop_limit` is an int `N >= 2`.

Return exactly this dict. `answer` is `None` unless the reason is `"final"`:

```
{"answer": <value or None>, "stop_reason": "final" | "step_budget" | "loop_detected" | "model_exhausted", "trace": [...]}
```

On every iteration, do the first rule that applies:

1. `max_steps` responses already consumed: stop `"step_budget"`.
2. No response left: stop `"model_exhausted"`.
3. Consume one response and append `{"kind": "model", "response": <that dict>}`.
4. A final response: stop `"final"` with its answer.
5. This proposal is the Nth in a row with the same tool name and `args ==` the previous ones: stop `"loop_detected"` and do not call the tool. Equality is Python's: dict key order does not matter, list order does. The streak counts every proposal, including an unknown tool, and resets when the tool or the args differ.
6. The name is not in `tools`: append `{"kind": "observation", "tool": name, "ok": False, "error": "unknown tool: <name>"}` and continue. Do not call it. The next iteration consumes the next scripted response.
7. Call the tool. Success appends `{"kind": "observation", "tool": name, "ok": True, "output": <return value>}`. If it raises `e`, append `{"kind": "observation", "tool": name, "ok": False, "error": f"{type(e).__name__}: {e}"}`. Either way, continue.

Trace entries have only the keys shown above.

```
# search(query) returns [query]. max_steps=4, loop_limit=3.
script = [
    {"type": "tool", "tool": "search", "args": {"query": "gpu"}},
    {"type": "tool", "tool": "missing", "args": {}},
    {"type": "final", "answer": 2}]
solve(script, tools, 4, 3) ->
{"answer": 2, "stop_reason": "final", "trace": [
    {"kind": "model", "response": {"type": "tool", "tool": "search", "args": {"query": "gpu"}}},
    {"kind": "observation", "tool": "search", "ok": True, "output": ["gpu"]},
    {"kind": "model", "response": {"type": "tool", "tool": "missing", "args": {}}},
    {"kind": "observation", "tool": "missing", "ok": False, "error": "unknown tool: missing"},
    {"kind": "model", "response": {"type": "final", "answer": 2}}]}
```

The tests also record every real tool call. The proposal that trips the loop is not one of them, and neither is an unknown tool.

**Interview focus:** a hard step budget, detecting a repeated identical tool call, feeding an unknown tool back once as an observation, and a trace a reviewer can replay.

**Follow-ups** to answer out loud once your solution passes:

- How would you budget by tokens or wall time instead of steps?
- The model calls the same tool with slightly different arguments forever. How do you detect it?
- What would you log so a failed run can be replayed and debugged?
