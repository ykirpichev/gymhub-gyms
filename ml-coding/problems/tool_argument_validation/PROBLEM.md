# Validate JSON tool arguments

Write `solve(schema, args)`. A model proposed a tool call with arguments `args`; check them against the tool's `schema` before anything runs.

- `schema`: a dict, a subset of JSON Schema:
  - `"type": "object"`.
  - `"properties"`: maps each argument name to a spec with a `"type"` (one of `"string"`, `"integer"`, `"number"`, `"boolean"`, `"array"`), an `"enum"` (a list of allowed values), or both.
  - `"required"` (optional, default `[]`): names that must be present. Every required name is also in `properties`.
  - `"additionalProperties"` (optional, default `true`): when `false`, names not in `properties` are errors; otherwise they are ignored.
- `args`: the decoded JSON arguments, normally a dict.

Return a `list[str]` of errors, empty when `args` is valid.

**Types** follow JSON, not Python: `"string"` is `str`; `"integer"` is `int` but **not** `bool`; `"number"` is `int` or `float`, not `bool` (so `3` is a valid number, `3.0` is not an integer); `"boolean"` is `bool`; `"array"` is `list`. The JSON type name of a value is `null` (`None`), `boolean`, `integer` (`int`), `number` (`float`), `string`, `array` (`list`) or `object` (`dict`).

**Enum:** a value matches an entry when `value == entry` and either both or neither are `bool` (so `true` does not match `1`). When a spec has both, check the type first and the enum only if the type is right.

**Error strings**, exactly:

| Problem | Message |
| --- | --- |
| required name absent | `missing required argument: <name>` |
| name not in `properties`, `additionalProperties` is `false` | `unexpected argument: <name>` |
| wrong type | `<name>: expected <type>, got <JSON type name>` |
| not in the enum | `<name>: <json.dumps(value)> is not one of <json.dumps(enum)>` |

Each name gets at most one error. Sort the list by argument name (plain string order). If `args` is not a dict, return just `["arguments: expected object, got <JSON type name>"]`.

```
schema = {"type": "object",
          "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "units": {"enum": ["C", "F"]}},
          "required": ["city", "days"], "additionalProperties": False}
solve(schema, {"days": True, "units": "K", "lang": "en"})
-> ['missing required argument: city',
    'days: expected integer, got boolean',
    'unexpected argument: lang',
    'units: "K" is not one of ["C", "F"]']
```

**Interview focus:** required fields, JSON types versus Python types (`bool` is an `int` subclass), enums, rejecting unexpected arguments, and deterministic error output.

**Follow-ups** to answer out loud once your solution passes:

- How would you return errors so the model can correct its call?
- How do you validate nested objects and arrays?
- Why is `True` a dangerous value for an integer field in Python?
