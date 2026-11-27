## Hint 1

Sector of a byte: `byte // 32`. A read of `size` bytes from `a` covers sectors `a // 32` through `(a + size - 1) // 32`.

## Hint 2

Keep two sets: sectors, and individual bytes.
