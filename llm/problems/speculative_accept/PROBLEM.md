# Speculative decoding: accept and resample

**Level:** hard   **Teaches:** speculative decoding, rejection sampling, first-failure index with tensor ops, inverse-CDF sampling

## Task

A small draft model proposed `K` tokens per sequence. The large target model scored all of them in one forward pass. Decide, for a whole batch at once, how many draft tokens to keep and which token comes next, so that the generated text has exactly the target model's distribution. All randomness is passed in as uniform numbers, so the result is deterministic.

## Arguments

| Name           | Type           | Shape            | Role                                                        |
| -------------- | -------------- | ---------------- | ----------------------------------------------------------- |
| `draft_tokens` | `torch.Tensor` | `[B, K]`, int64  | the proposed tokens                                         |
| `draft_probs`  | `torch.Tensor` | `[B, K, V]`      | `q_i`: the draft model's distribution at each position      |
| `target_probs` | `torch.Tensor` | `[B, K+1, V]`    | `p_i`: the target's distribution at each position, plus one more after the last draft token |
| `u_accept`     | `torch.Tensor` | `[B, K]`         | uniform numbers in `[0, 1)` for the accept tests            |
| `u_sample`     | `torch.Tensor` | `[B]`            | a uniform number in `[0, 1)` for the next token             |
| returns        | `tuple`        |                  | `(num_accepted, next_token)`, both `[B]` int64              |

## Rules

1. Walk the draft positions `i = 0 .. K−1` in order. With `x = draft_tokens[b, i]`, **accept** when `u_accept[b, i] * q_i(x) < p_i(x)` (accept with probability `min(1, p/q)`). Stop at the first rejection. `num_accepted[b]` is the number of tokens accepted before it (`K` if none is rejected).
2. **Next token.** If position `j = num_accepted[b] < K` was rejected, sample from the residual `r = max(p_j − q_j, 0)`. If all `K` were accepted, sample from `r = p_K` (the bonus token).
3. **Sampling:** the next token is the smallest index `t` with `cumsum(r)[t] > u_sample[b] · sum(r)`.

Every draft token has `q_i(x) > 0`, because the draft model sampled it.

## Example

```
B = 1, K = 2, V = 3
draft_tokens = [[0, 2]]
q_0 = [0.5, 0.25, 0.25]   p_0 = [0.25, 0.5, 0.25]   u_accept[0] = 0.4:  0.4 * 0.5 = 0.2  < 0.25  accept token 0
q_1 = [0.25, 0.25, 0.5]   p_1 = [0.5, 0.25, 0.25]   u_accept[1] = 0.6:  0.6 * 0.5 = 0.3 >= 0.25  reject
residual r = max(p_1 - q_1, 0) = [0.25, 0, 0]      u_sample = 0.7: cumsum [0.25, 0.25, 0.25] > 0.175 at index 0
num_accepted = [1]   next_token = [0]
```

## How to think about it

Do the accept test for all `B × K` positions at once with `gather`, then find the first rejection per row: `cumprod` of the accept mask is 1 up to the first `False` and 0 after, so its sum is `num_accepted`. Then gather the one distribution row you need per sequence (residual or bonus) and sample it with `cumsum` and a comparison, or `searchsorted`.

## Tests

All accepted (draft equals target), the first token rejected, a draft token the target gives probability 0, `K = 1`, and random batches up to `B = 32, K = 5, V = 1000`. Benchmark input: `B = 256, K = 5, V = 32,000`.

## This version and the other one

This is the batched GPU version. The GPU interview course's `speculative_verify` is one sequence in plain Python, for the logic alone.
