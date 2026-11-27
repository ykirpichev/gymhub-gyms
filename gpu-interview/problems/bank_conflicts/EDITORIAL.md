#### The idea
Group the active lanes by bank (`word % 32`). Inside a bank, lanes reading the *same* word share one broadcast, so
what costs a replay is each extra *distinct* word. The conflict degree is the largest number of distinct words in any
one bank.

#### A solution
```python
def solve(words):
    banks = {}
    for word in words:
        if word is not None:
            banks.setdefault(word % 32, set()).add(word)
    return max((len(distinct) for distinct in banks.values()), default=0)
```
One pass over at most 32 lanes.

#### On a real GPU
The hardware does the same grouping. A `k`-way conflict splits the instruction into `k` passes, so a shared-memory
load that should take one cycle takes `k`. The classic cases to recognize on sight:

| Access by lane `t` | Degree | Why |
|---|---|---|
| `tile[t]` (4-byte words) | 1 | one word per bank |
| `tile[t][c]` in `float tile[32][32]` | 32 | word `32t + c`, all in bank `c` |
| `tile[t][c]` in `float tile[32][33]` | 1 | word `33t + c`, bank `(t + c) % 32` |
| `x[2 * t]` | 2 | even words only, two per bank |
| `x[0]` for every lane | 1 | one broadcast |

Nsight Compute reports this as *shared memory bank conflicts* per load and store instruction.

#### Pitfalls
- Counting lanes per bank instead of distinct words: a broadcast is not a conflict.
- Forgetting inactive lanes (`None`), and returning 1 instead of 0 when no lane is active.
