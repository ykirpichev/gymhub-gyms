# Hints for Minimum shipping capacity

## Hint 1

Write `days_needed(C)`: walk the packages, start a new day whenever the next one does not fit. Is `days_needed` increasing or decreasing as `C` grows?

## Hint 2

`days_needed` never increases when `C` grows. So the capacities that work form a range `[answer, ∞)`. Binary-search its left end.

## Hint 3

Search between `lo = max(weights)` (anything smaller is impossible) and `hi = sum(weights)` (one day). While `lo < hi`: let `mid = (lo + hi) // 2`; if `days_needed(mid) <= days`, set `hi = mid`, else `lo = mid + 1`.
