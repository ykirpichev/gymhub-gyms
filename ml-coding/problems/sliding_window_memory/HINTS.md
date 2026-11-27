## Hint 1
Split the list into pinned and unpinned once, preserving order inside each. The budget check on the pinned messages happens before any summary.

## Hint 2
If the unpinned messages already fit in the remaining room, return them unchanged. Summarize only a prefix you actually drop.

## Hint 3
Try prefixes from shortest to longest. For each prefix, call `summarize` on it and `count_tokens` on that result plus the suffix you would keep. Take the first prefix that fits, which is the one that keeps the most recent messages.
