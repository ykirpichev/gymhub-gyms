## Hint 1
The mean is `sum(weight * value) / sum(weight)`. Store those two sums, plus the number of values seen. `result()` divides them, and returns `None` when the weight sum is 0.

## Hint 2
`update` adds each pair into the three numbers. Check the lengths and reject a negative weight before changing anything, so a bad batch leaves the old totals in place.

## Hint 3
`merge` adds the other object's three numbers to yours. That matches one pass over both streams. Averaging the means of the batches does not: a batch with total weight 4 and a batch with total weight 1 are not equal votes.
