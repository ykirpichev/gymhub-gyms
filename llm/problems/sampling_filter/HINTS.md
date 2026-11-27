# Hints for top-k / top-p filtering

## Hint 1

Top-k: `kth = torch.topk(logits, k, dim=-1).values[:, -1:]`, then `logits.masked_fill(logits < kth, float("-inf"))`. Guard the disabled cases (`k == 0`, `k >= V`) first.

## Hint 2

Top-p: `sorted_logits, order = torch.sort(x, dim=-1, descending=True)`, `p = softmax(sorted_logits)`, `remove_sorted = (p.cumsum(-1) - p) >= top_p`. Then map back: `remove = torch.zeros_like(remove_sorted).scatter(-1, order, remove_sorted)`. Position 0 of the sorted row has nothing before it, so it is never removed.

## Hint 3

For speed with top-k active: `vals, idx = torch.topk(logits, k)` already returns the survivors sorted descending. Run the top-p logic on those `k` columns only, then scatter the surviving values into a row full of `-inf`. That replaces a sort of 128k values per row with one `topk`.
