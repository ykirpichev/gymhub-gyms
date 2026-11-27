## Hint 1
Walk the union of `required` names and the names in `args`, sorted. For each name decide which single problem applies: absent and required, present but undeclared, or present and declared (then check type, then enum).

## Hint 2
`isinstance(True, int)` is `True` in Python, and `True == 1`. Test for `bool` before `int` both when checking types and when comparing against enum entries.

## Hint 3
Write a small `json_type(value)` that returns the JSON type name (check `None`, then `bool`, then `int`, `float`, `str`, `list`, `dict`), and a table from schema type to an accept function. Format enum errors with `json.dumps`.
