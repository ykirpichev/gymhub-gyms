#### The idea
Keep **sufficient statistics**, not the mean: the weight sum and the weighted sum. Both are plain sums, so updates add
to them and merging two shards adds their totals. The mean is computed only when asked. That is the same reason
distributed metrics report sums and counts, and why averaging per-batch means is wrong when batches differ in weight.

Validate the whole batch before touching the totals, so a bad weight halfway through leaves the object unchanged.

#### A solution
```python
class StreamingWeightedMean:
    def __init__(self):
        self.count = 0
        self.weight_sum = 0.0
        self.weighted_sum = 0.0

    def update(self, values, weights):
        if len(values) != len(weights):
            raise ValueError("values and weights must have the same length")
        batch = []
        for value, weight in zip(values, weights):
            value = float(value)
            weight = float(weight)
            if weight < 0:
                raise ValueError("weights must be non-negative")
            batch.append((value, weight))
        for value, weight in batch:
            self.count += 1
            self.weight_sum += weight
            self.weighted_sum += weight * value

    def merge(self, other):
        self.count += other.count
        self.weight_sum += other.weight_sum
        self.weighted_sum += other.weighted_sum

    def result(self):
        if self.weight_sum == 0:
            return None
        return self.weighted_sum / self.weight_sum
```

#### What it costs
`O(batch)` per update, `O(1)` per merge and per result, `O(1)` memory. For very long streams of floats, naive summation
loses precision; Kahan summation or `math.fsum` per batch keeps it. A streaming **variance** uses the same pattern with
a third statistic (Welford's algorithm, or Chan's formula to merge).

#### Pitfalls
- Storing a running mean and averaging it with each new batch mean.
- Updating the totals while validating, so a failed update leaves half a batch behind.
- `result()` dividing by zero instead of returning `None`.
- Counting only non-zero weights in `count`.
