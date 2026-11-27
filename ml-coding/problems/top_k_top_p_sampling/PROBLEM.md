# Temperature, top-k, and top-p sampling

Write `solve(logits, temperature, k, p, u)`.

- `logits`: a length-`V` list of finite floats, `V ≥ 1`.
- `temperature`: a finite number `≥ 0`. `temperature < 0` raises `ValueError`.
- `k`: an int. `k ≤ 0` raises `ValueError`. `k > V` keeps every token.
- `p`: a finite number in `(0, 1]`. Otherwise raise `ValueError`.
- `u`: a finite number in `[0, 1)`. Otherwise raise `ValueError`.

Return `(probs, token)`:

- `probs`: a float64 array of shape `(V,)` that sums to 1.
- `token`: a Python int, one draw from `probs`.

Do the steps in this order. Validate `temperature`, `k`, `p`, and `u` before the shortcut in step 1.

1. If `temperature == 0`, return a one-hot vector at `argmax(logits)`. Ties go to the lowest index. `k`, `p`, and `u` do not change that vector or the token.
2. Otherwise `probs = softmax(logits / temperature)`. Subtract the max before the exponential so large logits stay finite.
3. **Top-k.** Rank tokens by probability, highest first. Equal probabilities keep the lower index first. Keep the first `min(k, V)` tokens, set the rest to 0, and renormalize so the kept values sum to 1.
4. **Top-p** uses that renormalized distribution, not the original softmax. Rank the tokens that still have a positive probability the same way. Keep the shortest prefix whose cumulative probability is at least `p`, and always keep at least one token. Set the rest to 0 and renormalize again.
5. **Draw** in index order, zeros included. `token` is the smallest `i` with `cumsum(probs)[i] > u`. If rounding leaves no such index, use the last index whose probability is positive.

```
solve([1.0, 2.0, 3.0], 2.0, 2, 0.8, 0.0)
-> (array([0.    , 0.3775, 0.6225]), 1)
```

Dividing by 2 gives `[0.5, 1.0, 1.5]`, whose softmax is about `[0.1863, 0.3072, 0.5065]`. Top-2 drops index 0. Renormalizing the other two is `softmax([1.0, 1.5])`, about `[0.3775, 0.6225]`. The larger mass is `0.6225 < 0.8`, so top-p keeps both. `u = 0` selects index 1, the first token that survived.

**Interview focus:** Filtering order, renormalization, deterministic tests.

## This version and the other one

This is the CPU interview version for one row: temperature, both filters, and the draw itself, so it returns `(probs, token)`. The LLM gym's `sampling_filter` is the GPU version: it filters a whole batch of logit rows on CUDA and returns filtered logits, leaving the draw to the caller.

**Follow-ups** to answer out loud once your solution passes:

- Apply temperature before or after the filters? Does the order matter?
- What does min-p sampling do differently?
- How would you sample a whole batch at once on a GPU?
