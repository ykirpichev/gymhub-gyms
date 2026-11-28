## Hint 1

Loop over tile rows `i0`, tile columns `j0`, and K steps `k0`; then over `i`, `j`, `k` inside the tile.

## Hint 2

Loads are `(n / t) * (m / t) * (k / t) * 2 * t * t`; count them as you go to check yourself.
