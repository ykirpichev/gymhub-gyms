# Warp shuffle butterfly

A warp can sum 32 values without shared memory using `__shfl_xor_sync`: each step, lane `i` adds the value held by lane `i XOR offset`, for `offset = 16, 8, 4, 2, 1`. After five steps every lane holds the total.

Write `solve(values, steps)` that simulates the first `steps` steps.

- `values`: 32 numbers, the value in lane 0 to lane 31.
- `steps`: an integer from 0 to 5.

All lanes exchange at the same time: each step reads every lane's value from *before* the step. Return the 32 lane values after `steps` steps, as a list.

```
solve([1] * 32, 5)   ->  [32, 32, ..., 32]
```

#### What interviewers look for

- Simultaneous exchange: computing the new values from a snapshot, not in place.
- Knowing why the XOR pattern ends with the total in *every* lane, while `__shfl_down_sync` leaves it only in lane 0.

#### Follow-ups

- Change the sum to a max. What about an argmax that returns the index too?
- What does the `0xffffffff` mask in `__shfl_xor_sync(0xffffffff, v, offset)` mean, and when must it be different?
- How would you reduce 1,024 values in a block using shuffles and one small shared array?
