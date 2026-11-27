# Symmetric contrastive loss

Write `solve(img, txt, temperature)`: the symmetric InfoNCE loss used to train CLIP on a batch of `N` matching pairs.

- `img`, `txt`: `N × d` embeddings (list of lists or array). Row `i` of `img` and row `i` of `txt` are a positive pair; every other row in the batch is a negative. `N ≥ 1`, `d ≥ 1`.
- `temperature`: a float `> 0`.

Steps:

1. L2-normalize every row: `u / max(‖u‖, 1e-12)`. A zero row stays zero.
2. `logits = (img_n @ txt_nᵀ) / temperature`, an `N × N` matrix.
3. `loss_i2t` = mean cross-entropy of the rows of `logits` with target `i` for row `i` (image → text).
4. `loss_t2i` = the same over the columns, i.e. the rows of `logitsᵀ` (text → image).
5. Return the Python float `(loss_i2t + loss_t2i) / 2`.

Raise `ValueError` if `img` and `txt` are not 2-D with the same shape, or if `temperature ≤ 0`.

`temperature` can be as small as `1e-4`, which makes the logits as large as `1e4`; the result must stay finite and accurate.

```
solve(img=[[1, 0], [0, 1]], txt=[[2, 0], [1, 1]], temperature=1.0)
-> 0.4912
```

Here `logits = [[1, 0.7071], [0, 0.7071]]`: image → text gives `0.4791`, text → image gives `0.5032`. Note `[2, 0]` normalizes to `[1, 0]`.

**Interview focus:** normalizing the embeddings, the temperature, using the rest of the batch as negatives, a stable log-softmax, and including both directions.

**Follow-ups** to answer out loud once your solution passes:

- Why does a larger batch usually help this loss?
- What does the temperature do, and why is it sometimes learned?
- How do hard negatives change training, and what is the risk of false negatives?
