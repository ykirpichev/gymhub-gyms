#### The idea
A free list that always hands out the lowest-numbered block is a min-heap. For an append, compute how many blocks the
new length needs (`ceil(length / block_size)`) minus how many the sequence has; if the heap holds fewer, reject the
whole operation without touching anything.

#### A solution
```python
import heapq


def solve(block_size, num_blocks, ops):
    free = list(range(num_blocks))
    heapq.heapify(free)
    tables, lengths, rejected = {}, {}, []
    for index, op in enumerate(ops):
        if op[0] == "append":
            seq, n = op[1], op[2]
            new_length = lengths.get(seq, 0) + n
            needed = -(-new_length // block_size) - len(tables.get(seq, []))   # ceil division
            if needed > len(free):
                rejected.append(index)
                continue
            tables.setdefault(seq, []).extend(heapq.heappop(free) for _ in range(needed))
            lengths[seq] = new_length
        elif op[0] == "free":
            seq = op[1]
            if seq not in tables:
                raise ValueError(f"unknown sequence {seq!r}")
            for block in tables.pop(seq):
                heapq.heappush(free, block)
            del lengths[seq]
        else:
            raise ValueError(f"unknown op {op[0]!r}")
    return {"tables": tables, "lengths": lengths, "free": len(free), "rejected": rejected}
```
Each allocation or free is `O(log num_blocks)` per block.

#### In a real server
This is vLLM's block manager in miniature (see the diagram in the serving lesson). The attention kernel reads a
sequence's keys and values through its block table, so the blocks need not be contiguous. Fragmentation is at most
one partly filled block per sequence. Two features build on it: **prefix sharing** (sequences with the same system
prompt point at the same blocks, with reference counts and copy-on-write) and **preemption** (when no block is free,
the scheduler pauses a sequence and frees or swaps out its blocks, instead of failing the request).

#### Pitfalls
- Creating the sequence's table before knowing the append fits: a rejected first append must leave no trace.
- Allocating per token instead of per block: a sequence of 5 tokens with block size 4 needs 2 blocks, not 5.
