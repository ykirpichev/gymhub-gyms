## Hint 1
Compute the mean and standard deviation of each column from `train` only, then apply those two vectors to `train` and to `test`.

## Hint 2
Use the population standard deviation: divide the sum of squared deviations by `N`, not `N - 1`. One training row has standard deviation 0.

## Hint 3
If a training column has standard deviation 0, write 0 for every train and test row in that column. Do not divide by zero, and do not let a different test value change the scale.
