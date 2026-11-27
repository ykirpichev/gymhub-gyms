#### The idea
The perceptron is online: it looks at one sample at a time and updates only on a mistake. A mistake moves `w` toward
`y·x`, which raises that sample's margin by `‖x‖² + 1` (the `+1` is the bias). The order of samples matters, and a
zero margin counts as wrong, which is what gets the first update going from `w = 0`.

#### A solution
```python
import numpy as np


def solve(X, y, epochs):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all((y == 1) | (y == -1)):
        raise ValueError("labels must be -1 or +1")
    w = np.zeros(X.shape[1])
    b = 0.0
    updates = 0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:  # a zero margin is a mistake
                w += yi * xi
                b += yi
                updates += 1
    return w, b, updates
```

#### What it costs
`O(epochs·N·D)` time. On linearly separable data with margin `γ` and radius `R = max ‖x‖`, Novikoff's theorem bounds
the total number of mistakes by `(R/γ)²`, independent of `N`. On non-separable data it never settles, which is why
people use the averaged perceptron or move to a hinge loss (the SVM in the next exercise).

#### Pitfalls
- Using `< 0` instead of `<= 0`: from `w = 0` every margin is 0 and nothing is ever learned.
- Vectorising the epoch (updating with all mistakes at once): that is a different algorithm, batch gradient descent
  on the perceptron loss, and gives different numbers.
- Stopping early when an epoch is clean; the statement asks for every epoch (the result is the same, but the habit of
  reading the spec matters in an interview).
- Forgetting the bias update.
