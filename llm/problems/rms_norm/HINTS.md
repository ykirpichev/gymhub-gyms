# Hints for RMSNorm

## Hint 1

`x.pow(2).mean(dim=-1, keepdim=True)` gives one value per row with a trailing dimension of 1, which broadcasts back against `x`.

## Hint 2

`torch.rsqrt(t)` is `1 / sqrt(t)` in one op. Multiplying by it is cheaper than dividing by `sqrt`.

## Hint 3

For speed, fewer passes over memory wins. Each op you chain reads and writes the full tensor once. Try `torch.compile` on your function, or write the whole thing as one Triton kernel.
