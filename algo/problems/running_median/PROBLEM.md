# Running median

**Level:** medium   **Teaches:** two heaps, keeping an invariant under inserts, streaming statistics

## Task

Numbers arrive one at a time. After each one, report the median of every number seen so far. With an odd count the median is the middle value; with an even count it is the average of the two middle values.

## Arguments

| Name      | Type          | Role                                            |
| --------- | ------------- | ----------------------------------------------- |
| `nums`    | `list[int]`   | the stream, in arrival order; may be empty      |
| returns   | `list[float]` | `len(nums)` medians, one after each arrival     |

## Example

```
nums = [5, 15, 1, 3]
after 5:            [5]            -> 5.0
after 15:           [5, 15]        -> 10.0
after 1:            [1, 5, 15]     -> 5.0
after 3:            [1, 3, 5, 15]  -> 4.0
result = [5.0, 10.0, 5.0, 4.0]
```

## How to think about it

Keep the smaller half of the numbers in a max-heap and the larger half in a min-heap, with sizes equal or the lower half one larger. The median is then the top of the lower half, or the average of both tops. Each new number goes into one heap, and at most one number moves across to restore the balance.

## Tests

Empty input, one number, duplicates, negatives, increasing and decreasing streams, and random streams up to 3,000 numbers. Benchmark input: 100,000 random numbers. There is no time limit; re-sorting after every arrival is O(n² log n) and very slow.
