# Hints for the beam search step

## Hint 1

`beam_scores[:, :, None] + logprobs` is `[B, W, V]`: every candidate's total score.

## Hint 2

`.reshape(B, W * V)` puts each input's candidates in one row, then `torch.topk(row, W, dim=-1)` gives values and flat indices, sorted best first.

## Hint 3

`parents = flat // V` and `tokens = flat % V`. `topk` returns int64 indices already.
