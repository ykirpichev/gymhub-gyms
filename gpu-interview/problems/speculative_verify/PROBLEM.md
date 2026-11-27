# Speculative decoding: the verify step

A draft model proposed `k` tokens `draft[0..k-1]`, sampling token `i` from its distribution `q[i]`. The target model, in one forward pass, gave distributions `p[0..k]` (one more than the draft: `p[k]` is for the position after the last draft token). Each distribution is a list of probabilities over the same vocabulary of token ids `0..V-1`.

Write `solve(draft, q, p, u, u_resample)` that applies the speculative sampling rule with the random numbers given:

1. For `i = 0, 1, …, k − 1`: let `x = draft[i]`. If `u[i] < p[i][x] / q[i][x]`, accept `x` and continue.
2. Otherwise reject it: build the residual `r_j = max(0, p[i][j] − q[i][j])`, normalize it to sum to 1, sample a token from it with `u_resample`, append that token and stop.
3. If all `k` tokens were accepted, sample one more token from `p[k]` with `u_resample` and append it.

To sample from a distribution `d` with a uniform number `v`: return the smallest `j` such that `v < d[0] + d[1] + … + d[j]` (if rounding leaves no such `j`, return the last token with nonzero probability).

Return the list of tokens produced: between 1 and `k + 1` of them.

```
solve([2], [[0.0, 0.5, 0.5]], [[0.2, 0.2, 0.6], [1.0, 0.0, 0.0]], [0.9], 0.3)  ->  [2, 0]
```

`p/q` for token 2 is `0.6 / 0.5 ≥ 1`, so it is accepted; then token 0 is sampled from `p[1]`.

#### What interviewers look for

- The acceptance ratio `p/q` (capped at 1) and sampling the correction from the *residual* `max(0, p − q)`, not from `p`.
- Knowing why it is exact: the accept and resample paths together give exactly `p`.
- The payoff: with acceptance rate `α` per token, each target pass yields on average $(1 - \alpha^{k+1}) / (1 - \alpha)$ tokens.

#### Follow-ups

- With greedy decoding (`temperature = 0`), what does this rule reduce to?
- Why does the bonus token come for free?
- How would you choose `k`, and what makes the draft model a good one? (Self-speculation, Medusa heads, EAGLE.)
