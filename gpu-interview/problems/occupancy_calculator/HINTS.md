## Hint 1

`warps = (threads_per_block + 31) // 32`, then `threads_alloc = warps * 32`.

## Hint 2

Use `float("inf")` for an unused resource, then take the minimum over a list of `(count, name)` pairs in the tie-break order.
