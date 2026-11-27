## Hint 1
Start with one `L × L` causal mask: `np.tril(np.ones((L, L), dtype=bool))` is True exactly where `j <= i` (row `i`, column `j`). The diagonal must be True: a token sees itself.

## Hint 2
Padding acts along both axes. `valid[:, None, :]` (shape `B × 1 × L`) masks keys, i.e. columns; `valid[:, :, None]` (shape `B × L × 1`) masks queries, i.e. rows. Combine all three with `&` and let broadcasting produce `B × L × L`.
