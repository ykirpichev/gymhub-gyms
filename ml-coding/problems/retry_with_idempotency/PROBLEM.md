# Retry with idempotency

Write a class `IdempotentExecutor`. A call may fail transiently and be retried; the same idempotency key must never run the operation twice.

| Method | Behavior |
| --- | --- |
| `__init__(sleep, base, cap, max_attempts)` | `sleep(delay)` is injected. `base` and `cap` are numbers, `max_attempts` is an int ≥ 1. |
| `run(key, fn, transient)` | call `fn()` under the string `key`. `transient` is an exception class. |

**Backoff.** Number the attempts `i = 0, 1, ...`. When attempt `i` raises a transient error and another attempt is allowed, call `sleep(min(cap, base * 2 ** i))`. Sleep even if that delay is `0`. Do not sleep after success, after a permanent error, or after the last allowed attempt.

**Transient means `isinstance(exc, transient)`.** A subclass is retried. Any other exception is permanent: stop, and do not sleep for it.

`max_attempts` counts every call to `fn`, including the first. After that many transient failures, stop and keep the last one.

**Result of `run`:**

```
{"ok": True, "value": <what fn returned>, "attempts": <int>, "delays": [<delay>, ...]}
{"ok": False, "error": "<ExceptionName>: <message>", "attempts": <int>, "delays": [<delay>, ...]}
```

`attempts` is how many times `fn` ran. `delays` are the numbers passed to `sleep`, in order. `error` is `f"{type(exc).__name__}: {exc}"` for the exception that stopped the run.

**Idempotency.** Store that dict on the first completed run, success or failure. A later `run` with the same key returns it and does not call `fn` or `sleep`, ignoring the new `fn` and the new `transient`. Other keys have their own stored results.

```
# sleep is injected, base=1, cap=5, max_attempts=4, transient=TimeoutError
# fn raises TimeoutError("slow") twice, then records one side effect and returns {"id": 7}
run("charge", fn, TimeoutError) -> {"ok": True, "value": {"id": 7}, "attempts": 3, "delays": [1, 2]}
run("charge", other_fn, ConnectionError) -> the same dict; other_fn is not called
```

The tests record every delay passed to `sleep`, every call to `fn`, and every side effect `fn` records before it returns or raises.

**Interview focus:** separating transient errors from permanent ones, capped exponential backoff, and suppressing a duplicate key so the side effect happens once.

**Follow-ups** to answer out loud once your solution passes:

- Why add random jitter to the backoff?
- Where should the idempotency keys be stored, and for how long?
- Which HTTP errors would you retry, and which never?
