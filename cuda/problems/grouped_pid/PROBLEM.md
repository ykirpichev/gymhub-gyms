# Grouped tile order for L2 reuse

Triton's matmul tutorial launches one program per output tile and maps the program id `pid` to a tile `(pid_m, pid_n)`
in **grouped** order, so that programs that run at the same time share rows of A and columns of B in the L2 cache:

```python
num_pid_in_group = group_m * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * group_m
group_size_m = min(num_pid_m - first_pid_m, group_m)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

Write `solve(num_pid_m, num_pid_n, group_m)` that returns the list of `[pid_m, pid_n]` for every `pid` from 0 to
`num_pid_m * num_pid_n - 1`.

Example: `solve(2, 2, 2)` → `[[0, 0], [1, 0], [0, 1], [1, 1]]` (down a column of tiles first). With `group_m = 1` the
order is row by row: `solve(2, 2, 1)` → `[[0, 0], [0, 1], [1, 0], [1, 1]]`.
