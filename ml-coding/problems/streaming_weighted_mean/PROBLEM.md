# Streaming weighted mean

Write a class `StreamingWeightedMean`. There is no `solve`. The tests build the class and call its methods.

```python
class StreamingWeightedMean:
    def update(self, values, weights) -> None: ...
    def merge(self, other) -> None: ...
    def result(self): ...
```

- `update(values, weights)` folds in one batch. `values` and `weights` are lists of numbers of the same length. A weight is added to `sum(weights)` and `value * weight` is added to `sum(weight * value)`. Also count how many values you have seen, including zeros.
- `merge(other)` adds another `StreamingWeightedMean`'s count, weight sum and weighted sum into this one. Merging two shards must match updating with every sample from both.
- `result()` returns the weighted mean `weighted_sum / weight_sum` as a Python float. If the weight sum is 0 (a new object, an empty batch, or only zero weights), return `None`.

Raise `ValueError` if `len(values) != len(weights)` or any weight is negative. A failed `update` must not change the totals. Values may be negative. Weights may be zero; they add nothing to either sum.

Do not average the means of batches. `0` with total weight 4 and `10` with total weight 1 average to `2`, not `5`.

```
m = StreamingWeightedMean()
m.update([0.0, 0.0], [1.0, 3.0])
m.update([10.0], [1.0])
m.result() -> 2.0
```

**Interview focus:** Weight by sample count, mergeable state, empty state.

**Follow-ups** to answer out loud once your solution passes:

- Extend it to a streaming variance. (Welford's algorithm.)
- How would you merge states computed on different machines?
- How would you make recent values count more? (An exponential moving average.)
