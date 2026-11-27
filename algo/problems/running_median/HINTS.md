# Hints for Running median

## Hint 1

You only ever need the one or two middle values. Split the numbers seen so far into a lower half and an upper half.

## Hint 2

Python's `heapq` is a min-heap. Store the lower half negated to get a max-heap.

## Hint 3

Push the new number into the lower half, then move the lower half's largest into the upper half. If the upper half is now bigger, move its smallest back. The lower half always holds the extra element on an odd count.
