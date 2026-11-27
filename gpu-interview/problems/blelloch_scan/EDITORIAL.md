#### The idea
Two sweeps over an implicit binary tree laid out in the array:
- **Up-sweep:** each node (at index `i`) adds its left child's subtree total (at `i − stride`). The root, the last
  element, ends with the grand total.
- **Down-sweep:** clear the root, then walk back down: each node passes its value to its left child and adds the old
  left value to itself. Each position ends with the sum of everything before it.

#### A solution
```python
def solve(values):
    a = list(values)
    n = len(a)
    if n == 0 or n & (n - 1):
        raise ValueError("length must be a power of two")
    stride = 1
    while stride < n:                                   # up-sweep
        for i in range(2 * stride - 1, n, 2 * stride):
            a[i] += a[i - stride]
        stride *= 2
    tree = list(a)
    a[n - 1] = 0
    stride = n // 2
    while stride >= 1:                                  # down-sweep
        for i in range(2 * stride - 1, n, 2 * stride):
            a[i - stride], a[i] = a[i], a[i] + a[i - stride]
        stride //= 2
    return (tree, a)
```
`n & (n - 1)` is zero exactly for powers of two.

#### Cost
`n − 1` additions up and `n − 1` down: `O(n)` work, against `O(n log n)` for the simpler Hillis–Steele scan. Both
take `O(log n)` parallel steps. On a GPU each inner loop is one step for a block of threads with a
`__syncthreads()` between steps. The strides make shared-memory accesses conflict badly at the deep levels;
implementations pad the index (`i + i / 32`) to spread them over banks.

#### Pitfalls
- An inclusive scan is the exclusive one shifted left plus the total; the task wants exclusive.
- The swap in the down-sweep needs the *old* left value; the tuple assignment above evaluates the right side first.
