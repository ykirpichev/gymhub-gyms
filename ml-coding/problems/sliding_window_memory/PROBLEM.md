# Sliding-window conversation memory

Write a class `SlidingWindowMemory`. It turns a conversation into the messages that fit in a token budget.

| Method | Behavior |
| --- | --- |
| `__init__(count_tokens, summarize, budget)` | `count_tokens(message) -> int` and `summarize(messages) -> message` are injected. `budget` is an int. |
| `build(messages) -> list` | return the messages to send, or raise `ValueError`. |

Every message is a dict with `"pinned": bool`. Both injected functions are pure. Call them; do not reimplement a particular counter or summarizer, and do not modify the dicts they give you.

Let `pinned` be the pinned messages in their original order, and `rest` the others in their original order.

- If `count_tokens` over `pinned` is greater than `budget`, raise `ValueError`. Pinned messages are never dropped and never passed to `summarize`.
- Otherwise `room = budget - that sum`. If `count_tokens` over `rest` is `<= room`, return `pinned + rest` and do not summarize.
- Otherwise find the smallest `k >= 1` such that `count_tokens(summarize(rest[:k])) + count_tokens` over `rest[k:]` is `<= room`. Return `pinned + [that summary] + rest[k:]`.
- If no such `k` exists, raise `ValueError`. The returned messages never exceed the budget.

`k` is the number of oldest unpinned messages evicted. One summary replaces that prefix, and it occupies the budget too. Output order is pinned messages, then the summary if any, then the kept recent messages.

```
count_tokens(m) = m["tokens"]
summarize(evicted) = {"role": "system", "pinned": False,
                      "content": "summary:" + "|".join(m["content"] for m in evicted),
                      "tokens": 1 + len(evicted)}
budget = 12
messages = [
    {"role": "system", "content": "rules", "pinned": True, "tokens": 3},
    {"role": "user", "content": "ancient", "pinned": False, "tokens": 4},
    {"role": "assistant", "content": "ancient-reply", "pinned": False, "tokens": 4},
    {"role": "system", "content": "format", "pinned": True, "tokens": 2},
    {"role": "user", "content": "recent", "pinned": False, "tokens": 3},
    {"role": "assistant", "content": "recent-reply", "pinned": False, "tokens": 2}]
# pinned costs 5, so room is 7. The unpinned costs are 4, 4, 3, 2.
# k=1 costs 2+9, k=2 costs 3+5, k=3 costs 4+2 and fits.
build(messages) -> [
    {"role": "system", "content": "rules", "pinned": True, "tokens": 3},
    {"role": "system", "content": "format", "pinned": True, "tokens": 2},
    {"role": "system", "pinned": False, "content": "summary:ancient|ancient-reply|recent", "tokens": 4},
    {"role": "assistant", "content": "recent-reply", "pinned": False, "tokens": 2}]
```

Each `build` depends only on its argument. An empty `messages` returns `[]` when the pinned cost is 0.

**Interview focus:** pinned instructions stay, the oldest unpinned messages are the ones evicted, and the summary that replaces them spends tokens from the same budget.

**Follow-ups** to answer out loud once your solution passes:

- How would you decide what goes into the summary?
- When is retrieval over past messages better than a summary?
- How do you count tokens consistently with the model's tokenizer?
