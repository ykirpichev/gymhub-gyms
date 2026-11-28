## Hint 1

A read of `size` bytes at `a` covers sectors `a // 32` to `(a + size - 1) // 32`.

## Hint 2

Collect requested bytes in a set to count duplicates once.
