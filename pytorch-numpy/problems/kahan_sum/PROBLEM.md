# Kahan summation in float32

Write `solve(values)` that sums a list of numbers **in float32** two ways and returns `(naive, kahan)` as Python floats:

- `naive`: a running `np.float32` sum, adding one value at a time in order.
- `kahan`: Kahan's compensated sum, all in `np.float32`:

```
total = 0, c = 0
for x in values:
    y = x - c
    t = total + y
    c = (t - total) - y
    total = t
```

Convert each value to `np.float32` before using it, and do every operation on `np.float32` scalars.

```
solve([16777216.0, 1.0, 1.0])  ->  (16777216.0, 16777218.0)
```

`16777216 = 2^24`, where the gap between float32 numbers becomes 2: adding 1 rounds back each time, but Kahan carries the lost 1 in `c` until it adds up to 2.

#### What interviewers look for

- The compensation term: `c` holds what the last addition rounded away, and the next addition takes it back.
- Doing everything in float32 on purpose; in Python floats (float64) the effect is hidden.

#### Follow-ups

- What does NumPy's `np.sum` do instead of a naive running sum? (Pairwise summation.)
- Why does a GPU reduction usually beat a naive CPU loop on accuracy, without trying to?
- Where does summation error bite in ML? (Long FP16 or BF16 accumulations, loss averaging over huge batches, optimizer updates much smaller than the weights.)
