# Hints for the mixture-of-experts layer

## Hint 1

`val, idx = torch.topk(logits, top_k, dim=-1)` then `gates = torch.softmax(val, dim=-1)`. Softmax over the `k` chosen logits, not over all `E`.

## Hint 2

Flatten: `flat_expert = idx.reshape(-1)` (length `T·k`), `flat_token = torch.arange(T).repeat_interleave(k)`, `flat_gate = gates.reshape(-1)`. For expert `e`, `sel = flat_expert == e` picks its assignments.

## Hint 3

For each expert: `tokens = flat_token[sel]`, `h = expert(x[tokens])`, then `y.index_add_(0, tokens, h * flat_gate[sel, None])`. Skip experts with no tokens. Sorting `flat_expert` once with `argsort` and slicing by counts (`torch.bincount`) avoids `E` boolean scans.
