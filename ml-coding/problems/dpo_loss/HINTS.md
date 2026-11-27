## Hint 1
The implicit reward of a response is `beta` times how much more likely the policy makes it than the reference does. The loss pushes the chosen reward above the rejected one, so it should be small when `z` is large and positive.

## Hint 2
`log(1 / (1 + exp(-z)))` breaks for `z = -1000`: `exp(1000)` overflows to `inf` and the log gives `-inf`. Rewrite `log σ(z)` so the argument of `exp` is never positive.

## Hint 3
`log σ(z) = min(z, 0) - log1p(exp(-|z|))`. (`-np.logaddexp(0, -z)` is equivalent.) Compute the rewards, take `z`, average `-log σ(z)`, and compare the reward arrays for the accuracy.
