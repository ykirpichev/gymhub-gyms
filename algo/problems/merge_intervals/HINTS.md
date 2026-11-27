# Hints for Merge intervals

## Hint 1

Sort the intervals by their start. After that, an interval can only overlap the run of intervals directly before it.

## Hint 2

Walk the sorted list while keeping a result list. Compare each interval's start with the end of the last interval in the result. Use `<=`, not `<`, so touching intervals merge.

## Hint 3

When they merge, set the last end to `max(last_end, end)`. A nested interval like `[2, 3]` inside `[1, 10]` must not shrink the end.
