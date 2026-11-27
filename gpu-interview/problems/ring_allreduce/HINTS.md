## Hint 1

Work with a list of chunks per rank: `chunks[r][c]` is a list of `L / N` numbers.

## Hint 2

For each step, first collect `(destination, chunk index, values)` for every rank, then apply them all.
