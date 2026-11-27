## Hint 1

Copy the input to a new list and work in place on it, exactly as the formulas say. Take a copy of the list after the up-sweep for `tree`.

## Hint 2

`for i in range(2 * stride - 1, n, 2 * stride)` visits the right indices for both phases.
