## Hint 1
Do the expensive checks once, in `register`: `inspect.signature(fn).parameters` gives each parameter's name and `default` (`inspect.Parameter.empty` when it has none). Store the required names with the tool.

## Hint 2
In `call`, collect validation messages for `sorted(set(args) | required)`. Check `bool` before `int`: `isinstance(True, int)` is `True`.

## Hint 3
Keep the order strict: unknown, invalid, approval, then run inside `try/except Exception`. Return early at each stage so a side-effecting function is never reached without approval.
