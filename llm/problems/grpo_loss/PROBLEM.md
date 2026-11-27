# GRPO loss

**Level:** hard   **Teaches:** policy-gradient objectives for LLMs, group-normalised advantages, PPO clipping, masked token means

## Task

Group Relative Policy Optimization (GRPO, from DeepSeekMath) trains a model on its own sampled answers. For each prompt it samples a **group** of `G` completions, scores each with a reward, and pushes up the completions that beat their group's average. Compute the GRPO loss for a batch.

## Arguments

| Name         | Type           | Shape          | Role                                                         |
| ------------ | -------------- | -------------- | ------------------------------------------------------------ |
| `logp_new`   | `torch.Tensor` | `[N, T]`       | per-token log-probs under the policy being trained, float32, CUDA |
| `logp_old`   | `torch.Tensor` | `[N, T]`       | per-token log-probs under the policy that sampled the completions |
| `logp_ref`   | `torch.Tensor` | `[N, T]`       | per-token log-probs under the frozen reference model          |
| `rewards`    | `torch.Tensor` | `[N]`          | one reward per completion                                     |
| `mask`       | `torch.Tensor` | `[N, T]`, bool | `True` on completion tokens; `False` on padding               |
| `group_size` | `int`          |                | `G >= 2`; rows `g·G .. g·G + G − 1` answer the same prompt    |
| `clip_eps`   | `float`        |                | the PPO clip range                                            |
| `beta`       | `float`        |                | the KL penalty weight                                         |
| returns      | `tuple`        |                | `(loss, advantages)`: a 0-dim tensor and an `[N]` tensor      |

`N` is a multiple of `G`. Every row has at least one `True` in `mask`. Masked-out positions hold arbitrary finite values.

## Formula

```
advantage A_i  = (r_i − mean of its group) / (std of its group + 1e-4)     std with G − 1 in the denominator
ratio          = exp(logp_new − logp_old)
surrogate      = min(ratio · A, clamp(ratio, 1 − clip_eps, 1 + clip_eps) · A)
kl             = exp(logp_ref − logp_new) − (logp_ref − logp_new) − 1      per token, always >= 0
token_loss     = −(surrogate − beta · kl)
row_loss       = mean of token_loss over the row's True positions
loss           = mean of row_loss over the N rows
```

## Example

```
N = 2, T = 2, G = 2, clip_eps = 0.2, beta = 0
rewards = [1, 0]: mean 0.5, std 0.7071, A = [0.7070, -0.7070]
logp_new = logp_old everywhere: ratio 1, surrogate = A on every token
mask = [[True, True], [True, False]]
row_loss = [-0.7070, 0.7070]   loss = 0.0
```

## How to think about it

The advantage is one number per completion, broadcast over its tokens. Reshape the rewards to `[N / G, G]` to get each group's mean and std. The clip stops a single update from moving a token's probability ratio far outside `[1 − ε, 1 + ε]`. The `kl` term is the low-variance "k3" estimator of the KL divergence to the reference. Average over real tokens only: a plain `mean()` counts padding.

## Tests

Identical rewards in a group (advantage 0), ratios inside and outside the clip range in both directions, `beta = 0`, rows of different lengths, junk in masked positions, and random batches. Benchmark input: `N = 512` (64 prompts × 8 completions), `T = 2048`.
