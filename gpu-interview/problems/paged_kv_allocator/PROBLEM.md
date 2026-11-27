# A paged KV-cache allocator

A server's KV cache is a pool of `num_blocks` blocks, numbered `0` to `num_blocks − 1`, each holding `block_size` tokens. Each sequence has a **block table**: the list of blocks holding its tokens, in order. Write `solve(block_size, num_blocks, ops)` that runs a list of operations:

- `["append", seq, n]`: add `n ≥ 1` tokens to sequence `seq` (a string), creating it if new. The sequence needs `ceil(length / block_size)` blocks for its new length. Allocate the extra blocks it needs, **lowest-numbered free blocks first**, appended to its table in that order. If there are not enough free blocks, the operation is rejected as a whole: nothing changes, and its index in `ops` is recorded.
- `["free", seq]`: return all of `seq`'s blocks to the pool and forget the sequence. Raise `ValueError` if `seq` is unknown.

Return a dict:

- `"tables"`: `{seq: [block, ...]}` for every live sequence.
- `"lengths"`: `{seq: tokens}` for every live sequence.
- `"free"`: the number of free blocks.
- `"rejected"`: the list of indices of rejected operations, in order.

```
solve(4, 3, [["append", "a", 5], ["append", "b", 3], ["append", "b", 2], ["free", "a"], ["append", "b", 2]])
-> {"tables": {"b": [2, 0]}, "lengths": {"b": 5}, "free": 1, "rejected": [2]}
```

`a` takes blocks 0 and 1 for its 5 tokens, and `b` takes block 2 for 3 tokens. Growing `b` to 5 tokens needs a second block and none is free, so op 2 is rejected and `b` stays at 3 tokens. Freeing `a` returns blocks 0 and 1. Op 4 grows `b` to 5 tokens and takes the lowest free block, 0.

#### What interviewers look for

- Allocating only the *difference* in blocks a sequence needs, with a ceiling division.
- All-or-nothing appends: never leaving a sequence half-grown when memory runs out.
- Talking about what a real server does on rejection: preempt a sequence (swap its blocks to CPU memory or recompute them later) rather than fail the request.

#### Follow-ups

- Two requests share a 1,000-token system prompt. How would reference counts let them share blocks, and what happens when one of them appends (copy-on-write)?
- How does the attention kernel use the block table to find a token's keys and values?
- What block size would you pick, and what does a larger one waste?
