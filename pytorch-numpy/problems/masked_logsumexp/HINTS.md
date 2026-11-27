## Hint 1

Replace a row max of `-inf` with 0 before subtracting, then the sum is 0 and `log(0) = -inf`. Silence the warning with `np.errstate(divide="ignore")`.
