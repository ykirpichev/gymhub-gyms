# The cost of divergence

Model how a warp executes an `if`/`else`. You get `taken`, a list of booleans (one per thread of the block, in
`threadIdx.x` order, length a multiple of 32), and the number of instructions in each side, `then_cost` and
`else_cost`.

For each warp: if any lane is `True`, the warp issues the `then` side (`then_cost` instructions); if any lane is `False`,
it issues the `else` side (`else_cost`). Return `[issued, useful]`, where `issued` is the total number of warp
instructions issued over all warps, and `useful` is the number of lane-instructions that did real work (each active
lane counts once per instruction of its side).

Example: one warp with lanes 0–15 `True` and 16–31 `False`, `then_cost = 3`, `else_cost = 2`: the warp issues 3 + 2 =
5 instructions; useful work is 16 × 3 + 16 × 2 = 80. So `[5, 80]`; efficiency is 80 / (5 × 32) = 50%.
