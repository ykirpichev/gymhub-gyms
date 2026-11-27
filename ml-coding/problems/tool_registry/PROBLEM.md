# Permission-aware tool registry

Write a class `ToolRegistry`. An agent calls tools by name with JSON arguments chosen by a model; the registry is the boundary that decides what actually runs.

| Method | Behavior |
| --- | --- |
| `__init__()` | empty registry |
| `register(name, fn, schema, side_effect=False)` | add a tool; returns `None` |
| `call(name, args, approved=False)` | returns a result dict (below); never raises |

`fn` is a plain Python function with ordinary named parameters (no `*args`, `**kwargs` or positional-only ones). `schema` maps each argument the model may pass to a type name: `"string"`, `"integer"`, `"number"`, `"boolean"`, `"array"` or `"object"`.

**`register` raises `ValueError`** when the name is already registered, a schema key is not a parameter of `fn`, a parameter of `fn` without a default is missing from the schema, or a type name is not one of the six.

**`call` checks in this order** and returns the first result that applies:

1. Unknown name: `{"ok": False, "error": "unknown_tool"}`.
2. Invalid arguments: `{"ok": False, "error": "invalid_arguments", "details": [...]}`. Check every name in `args` plus every required parameter (one without a default), sorted by name; each gets at most one message:
   - `missing required argument: <name>`: required, absent from `args`;
   - `unexpected argument: <name>`: present but not in the schema;
   - `<name>: expected <type>, got <json type>`: wrong type. Types follow JSON: `bool` is **not** an integer or a number, an `int` **is** a number, `"array"` is `list`, `"object"` is `dict`. The JSON type of a value is `null`, `boolean`, `integer` (`int`), `number` (`float`), `string`, `array` or `object`.
3. The tool has `side_effect=True` and `approved` is false: `{"ok": False, "error": "approval_required"}`. The function must **not** run.
4. Run `fn(**args)`; parameters the model left out take their defaults. If it raises an exception `e`: `{"ok": False, "error": "tool_error", "message": f"{type(e).__name__}: {e}"}`.
5. Otherwise `{"ok": True, "result": <return value>}`.

The tests replay a list of operations against one registry and also record every side effect the tools perform, so a tool that runs without approval fails even if the returned dict looks right.

```
reg = ToolRegistry()
reg.register("add", add, {"a": "integer", "b": "integer"})       # def add(a, b=0)
reg.register("send_email", send_email, {"to": "string", "subject": "string", "body": "string"},
             side_effect=True)                                     # def send_email(to, subject, body="")
reg.call("add", {"a": 2, "b": 3})               -> {"ok": True, "result": 5}
reg.call("add", {"a": "2", "c": 1})             -> {"ok": False, "error": "invalid_arguments",
                                                    "details": ["a: expected integer, got string", "unexpected argument: c"]}
reg.call("send_email", {"to": "ann@example.com", "subject": "hi"})
                                                -> {"ok": False, "error": "approval_required"}
reg.call("send_email", {"to": "ann@example.com", "subject": "hi"}, approved=True)
                                                -> {"ok": True, "result": {"id": 1}}
reg.call("rm", {})                              -> {"ok": False, "error": "unknown_tool"}
```

**Interview focus:** introspecting the signature at registration, validating before binding, the approval boundary for side effects (checked before anything runs), and turning every failure into a structured result instead of an exception.

**Follow-ups** to answer out loud once your solution passes:

- How would you version tools when their signatures change?
- Which tools should need human approval, and how do you enforce it?
- How do you keep a tool's description short enough for the context and still precise?
