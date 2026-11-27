# Hints for speculative decoding: accept and resample

## Hint 1

`p_x = target_probs[:, :K].gather(-1, draft_tokens[..., None]).squeeze(-1)` gives `p_i(x_i)` for every position; do the same for `q`. Then `accept = u_accept * q_x < p_x` is `[B, K]`.

## Hint 2

`accept.long().cumprod(dim=1).sum(dim=1)` is the number of leading `True`s: the accepted count.

## Hint 3

`rows = torch.arange(B)`. The residual row uses position `j = min(num_accepted, K − 1)`: `r = (target_probs[rows, j] - draft_probs[rows, j]).clamp(min=0)`. Where `num_accepted == K`, replace it with `target_probs[:, K]` using `torch.where`. Then `(r.cumsum(-1) > u_sample[:, None] * r.sum(-1, keepdim=True))` and `argmax` over the last axis finds the first `True`.
