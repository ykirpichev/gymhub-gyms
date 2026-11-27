## Hint 1
`max_attempts` is the size of the loop, not the number of sleeps. There is one fewer sleep than there are attempts, and only when each of those earlier attempts was transient.

## Hint 2
After attempt `i` (starting at 0) fails transiently, the delay is `min(cap, base * 2 ** i)`. Append that same number you pass to `sleep`. `except transient` already matches subclasses; a later `except Exception` handles the permanent case.

## Hint 3
Write the result dict once, store it under the key, and return it. The next `run` for that key returns the stored dict before it touches `fn` or `sleep`.
