## Hint 1
Write down the shapes first. `x Aᵀ` is `N × r`, and multiplying that by `Bᵀ` (`r × out`) gives `N × out`, the same shape as the base output `x Wᵀ`.

## Hint 2
The scale is `alpha / r`, not `alpha`. It keeps the update's size roughly constant when you change the rank, so you do not have to retune the learning rate.

## Hint 3
`(x Aᵀ) Bᵀ = x (B A)ᵀ`, so the adapter folds into one weight: `W_merged = W + s · B @ A`, which is `out × in`. Read `in`, `out` and `r` from the shapes, check that they agree, then count `r * (in + out)` and `out * in`.
