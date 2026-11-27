# Shared-memory bank conflicts

Shared memory has 32 banks, each 4 bytes wide: 4-byte word `w` lives in bank `w % 32`. Write `solve(words)` for one warp-wide access of 4-byte words.

- `words`: a list of up to 32 word indices (non-negative integers), one per lane, or `None` for an inactive lane.

Lanes that read the **same word** are served by one broadcast. Lanes that read **different words in the same bank** are serialized. Return the **conflict degree**: the largest number of distinct words any one bank must serve. A conflict-free access returns 1; no active lane returns 0.

```
solve([t for t in range(32)])        ->  1    # one word per bank
solve([32 * t for t in range(32)])   ->  32   # column of a [32][32] tile: all in bank 0
solve([33 * t for t in range(32)])   ->  1    # column of a [32][33] tile
```

#### What interviewers look for

- Banks are by *word*, `% 32`, and same-word reads are free.
- Being able to say which tile layout causes a conflict, and the two fixes: padding, or an XOR swizzle of the column index.

#### Follow-ups

- A warp reads `double`s (8 bytes) from consecutive addresses. Is that a conflict? (Hint: the hardware splits it into two half-warp phases.)
- When is a swizzle better than padding? (Think of `float4` reads and of the memory padding wastes.)
- Where do bank conflicts show up in a tiled matmul, and in which of the two operand tiles?
