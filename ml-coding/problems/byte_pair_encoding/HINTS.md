## Hint 1
Keep a dict from each distinct word to its current list of symbols, plus a `Counter` of word frequencies. Each round, rebuild a `Counter` of pairs with `zip(symbols, symbols[1:])`, adding the word's frequency.

## Hint 2
"Highest count, then smallest pair" is one key: `min(pairs, key=lambda p: (-pairs[p], p))`. `Counter.most_common` breaks ties by insertion order, which is not what is asked.

## Hint 3
Write one helper `merge(symbols, pair)` that walks the list with an index, joining and skipping two symbols on a match. Use it both while learning (on every word) and while encoding (once per learned merge, in order).
