# Log-probability of the right class

Write `solve(logits, targets)` for `logits` of shape `(N, C)` and integer `targets` of shape `(N,)`. Return a tuple:

- `picked`: the `N` values `log_softmax(logits)[i, targets[i]]`, as a NumPy array.
- `loss`: the mean of `-picked`, a float: the cross-entropy loss.

Compute `log_softmax` stably: `x - max - log(sum(exp(x - max)))` per row. Use indexing, not a loop.

```
solve([[2, 0], [0, 0]], [0, 1])  ->  (array([-0.1269, -0.6931]), 0.41)
```

#### What interviewers look for

- `x[np.arange(N), targets]`, or `take_along_axis`.
- The stable log-softmax, never `log(softmax(x))`.

#### Follow-ups

- Add `ignore_index`: targets equal to `-100` contribute nothing and are not counted in the mean.
- Add label smoothing with `ε = 0.1`. What changes?
- Why do frameworks fuse log-softmax and the loss into one operation?
